import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:math';
import 'dart:ui';

import 'package:flutterface/constants/model_file.dart';
import 'package:flutterface/services/ai_model.dart';
import 'package:flutterface/services/face_detection/anchors.dart';
import 'package:flutterface/services/face_detection/generate_anchors.dart';
import 'package:flutterface/services/face_detection/non_maximum_suppression.dart';
import 'package:flutterface/services/face_detection/options.dart';
import 'package:flutterface/services/face_detection/process.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

// ignore: must_be_immutable
class FaceDetection extends AIModel {
  FaceDetection({this.interpreter}) {
    loadModel();
  }

  final int inputSize = 128;
  final double threshold = 0.7;

  @override
  Interpreter? interpreter;

  @override
  List<Object> get props => [];

  @override
  int get getAddress => interpreter!.address;

  late List<Anchor> _anchors;
  late int originalImageWidth;
  late int originalImageHeight;

  @override
  Future<void> loadModel() async {
    final anchorOption = AnchorOption(
      inputSizeHeight: 128,
      inputSizeWidth: 128,
      minScale: 0.1484375,
      maxScale: 0.75,
      anchorOffsetX: 0.5,
      anchorOffsetY: 0.5,
      numLayers: 4,
      featureMapHeight: [],
      featureMapWidth: [],
      strides: [8, 16, 16, 16],
      aspectRatios: [1.0],
      reduceBoxesInLowestLayer: false,
      interpolatedScaleAspectRatio: 1.0,
      fixedAnchorSize: true,
    );
    try {
      final interpreterOptions = InterpreterOptions();

      // Use XNNPACK Delegate
      if (Platform.isAndroid) {
        interpreterOptions.addDelegate(XNNPackDelegate());
      }

      // Use GPU Delegate
      // doesn't work on emulator
      // if (Platform.isAndroid) {
      //   options.addDelegate(GpuDelegateV2());
      // }

      // Use Metal Delegate
      if (Platform.isIOS) {
        interpreterOptions.addDelegate(GpuDelegate());
      }

      // Create anchor boxes for BlazeFace
      _anchors = generateAnchors(anchorOption);

      // Load model from assets
      interpreter = interpreter ??
          await Interpreter.fromAsset(
            ModelFile.faceDetectionShortRange,
            options: interpreterOptions,
          );

      // Get tensor input shape [1, 128, 128, 3]
      final inputTensors = interpreter!.getInputTensors().first;
      devtools.log('Input Tensors: $inputTensors');
      // Get tensour output shape [1, 896, 16]
      final outputTensors = interpreter!.getOutputTensors();
      final outputTensor = outputTensors.first;
      devtools.log('Output Tensors: $outputTensor');

      for (var tensor in outputTensors) {
        outputShapes.add(tensor.shape);
        outputTypes.add(tensor.type);
      }
      devtools.log('Model interpreter loaded successfully');
    } catch (e) {
      devtools.log('Error while creating interpreter: $e');
    }
  }

  static num normalizePixel(int pixelValue) {
    return (pixelValue / 127.5) - 1;
  }

  @override
  List<List<List<num>>> getPreprocessedImage(String imagePath) {
    assert(interpreter != null);
    assert(imagePath.isNotEmpty);

    // Read image bytes from file
    final imageData = File(imagePath).readAsBytesSync();

    // Decode image using package:image/image.dart (https://pub.dev/image)
    final image_lib.Image? image = image_lib.decodeImage(imageData);
    if (image == null) {
      throw Exception('Image not found');
    }

    // if (Platform.isAndroid) {
    //   image = image_lib.copyRotate(image, angle: -90);
    //   image = image_lib.flipHorizontal(image);
    // }

    // Resize image for model input
    originalImageWidth = image.width;
    originalImageHeight = image.height;
    image_lib.Image imageInput = image_lib.copyResize(
      image,
      width: inputSize,
      height: inputSize,
      interpolation: image_lib
          .Interpolation.cubic, // if this is too slow, change to linear
    );

    // Normalize the image to range [-1, 1]
    imageInput = image_lib.normalize(imageInput, min: -1, max: 1);

    // Get image matrix representation [128, 128, 3]
    final imageMatrix = List.generate(
      imageInput.height,
      (y) => List.generate(
        imageInput.width,
        (x) {
          final pixel = imageInput.getPixel(x, y);
          return [pixel.r, pixel.g, pixel.b];
        },
      ),
    );

    return imageMatrix;
  }

  // TODO: Make the predict function asynchronous with use of isolate-interpreter
  @override
  List<Map<String, dynamic>> predict(String imagePath) {
    assert(interpreter != null);

    final options = OptionsFace(
      numClasses: 1,
      numBoxes: 896,
      numCoords: 16,
      keypointCoordOffset: 4,
      ignoreClasses: [],
      scoreClippingThresh: 100.0,
      minScoreThresh: 0.75,
      numKeypoints: 6,
      numValuesPerKeypoint: 2,
      reverseOutputOrder: true,
      boxCoordOffset: 0,
      xScale: 128,
      yScale: 128,
      hScale: 128,
      wScale: 128,
    );

    final inputImageMatrix = getPreprocessedImage(imagePath);
    final input = [inputImageMatrix];
    final output = [
      List<double>.filled(1, 896 * 16),
      List<double>.filled(1, 896 * 1)
    ];

    // Run inference
    interpreter!.run(input, output);

    // Get output tensors
    final rawBoxes = output[0];
    final rawScores = output[1];

    var detections = process(
      options: options,
      rawScores: rawScores,
      rawBoxes: rawBoxes,
      anchors: _anchors,
    );

    devtools.log(
      'Detections: ${detections.sublist(0, min(10, detections.length))}',
    );

    detections = nonMaximumSuppression(detections, threshold);
    if (detections.isEmpty) {
      devtools.log('[log] No face detected');
      return [
        {'bbox': Rect.zero, 'score': 0.0}
      ];
    }

    final rectFaces = <Map<String, dynamic>>[];

    for (var detection in detections) {
      Rect? bbox;
      final score = detection.score;
      if (score > threshold) {
        bbox = Rect.fromLTRB(
          originalImageWidth * detection.xMin,
          originalImageHeight * detection.yMin,
          originalImageWidth * detection.width,
          originalImageHeight * detection.height,
        );
      }
      rectFaces.add({'bbox': bbox, 'score': score});
    }
    rectFaces.sort((a, b) => b['score'].compareTo(a['score']));

    return rectFaces;
  }
}

// Map<String, dynamic>? runFaceDetector(Map<String, dynamic> params) {
//   try {
//     devtools.log('[log] runFaceDetector');
//     final faceDetection = FaceDetection(
//         // interpreter: Interpreter.fromAddress(params['detectorAddress']),);
//     final image = ImageUtils.convertCameraImage(params['cameraImage'])!;
//     final result = faceDetection.predict(image);
//     devtools.log('[log] Prediction complete: $result');
//     return result;
//   } catch (e) {
//     devtools.log('Error during face detection: $e');
//   }
// }
