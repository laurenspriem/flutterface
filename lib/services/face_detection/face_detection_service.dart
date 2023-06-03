import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:math';

import 'package:flutter/services.dart';
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
  FaceDetection._();

  static Future<FaceDetection> create() async {
    final faceDetector = FaceDetection._();
    await faceDetector.loadModel();
    return faceDetector;
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

  // static Future<Uint8List> loadImageData(String imagePath) async {
  //   final ByteData imageData = await rootBundle.load(imagePath);
  //   return imageData.buffer.asUint8List();
  // }

  static List createNestedList(List<int> shape) {
    if (shape.length < 2 || shape.length > 3) {
      throw ArgumentError('Shape must have length 2 or 3');
    }
    if (shape.length == 2) {
      return List.generate(shape[0], (_) => List.filled(shape[1], 0.0));
    } else {
      return List.generate(
        shape[0],
        (_) => List.generate(shape[1], (_) => List.filled(shape[2], 0.0)),
      );
    }
  }

  @override
  Future<void> loadModel() async {
    devtools.log('loadModel is called');
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
      devtools.log('loadModel is finished');
    } catch (e) {
      devtools.log('Error while creating interpreter: $e');
    }
  }

  static num normalizePixel(int pixelValue) {
    return (pixelValue / 127.5) - 1;
  }

  @override
  Future<List<List<List<num>>>> getPreprocessedImage(String imagePath) async {
    devtools.log('Preprocessing is called');
    assert(imagePath.isNotEmpty);

    // Read image bytes from file
    final imageData = File(imagePath).readAsBytesSync();
    // final imageData = await loadImageData(imagePath);

    // Decode image using package:image/image.dart (https://pub.dev/image)
    var image = image_lib.decodeImage(imageData);
    if (image == null) {
      throw Exception('Image not found');
    }

    if (Platform.isAndroid) {
      image = image_lib.copyRotate(image, angle: -90);
      image = image_lib.flipHorizontal(image);
    }

    originalImageWidth = image.width;
    originalImageHeight = image.height;

    // Resize image for model input
    final image_lib.Image imageInput = image_lib.copyResize(
      image,
      width: inputSize,
      height: inputSize,
      interpolation: image_lib
          .Interpolation.cubic, // if this is too slow, change to linear
    );

    // Get image matrix representation [128, 128, 3]
    final imageMatrix = List.generate(
      imageInput.height,
      (y) => List.generate(
        imageInput.width,
        (x) {
          final pixel = imageInput.getPixel(x, y);
          // return [pixel.r, pixel.g, pixel.b];
          return [
            pixel.r / 127.5 - 1, // Normalize the image to range [-1, 1]
            pixel.g / 127.5 - 1, // Normalize the image to range [-1, 1]
            pixel.b / 127.5 - 1, // Normalize the image to range [-1, 1]
          ];
        },
      ),
    );
    devtools.log('Preprocessing is finished');

    // Check the content of imageMatrix for anything suspicious!
    // for (var i = 0; i < imageMatrix.length; i++) {
    //   for (var j = 0; j < imageMatrix[i].length; j++) {
    //     devtools.log('Pixel at [$i, $j]: ${imageMatrix[i][j]}');
    //   }
    // }

    return imageMatrix;
  }

  // TODO: Make the predict function asynchronous with use of isolate-interpreter
  @override
  List<Map<String, dynamic>> predict(List<List<List<num>>> inputImageMatrix) {
    assert(interpreter != null);

    final options = OptionsFace(
      numClasses: 1,
      numBoxes: 896,
      numCoords: 16,
      keypointCoordOffset: 4,
      ignoreClasses: [],
      scoreClippingThresh: 100.0,
      minScoreThresh: 0.70,
      numKeypoints: 6,
      numValuesPerKeypoint: 2,
      reverseOutputOrder: true,
      boxCoordOffset: 0,
      xScale: 128,
      yScale: 128,
      hScale: 128,
      wScale: 128,
    );

    devtools.log('outputShapes: $outputShapes');

    final input = [inputImageMatrix];

    final outputFaces = createNestedList(outputShapes[0]);
    final outputScores = createNestedList(outputShapes[1]);
    final outputs = <int, List>{
      0: outputFaces,
      1: outputScores,
    };

    devtools.log('Input of shape ${input.shape}');
    devtools
        .log('Outputs: of shape ${outputs[0]?.shape} and ${outputs[1]?.shape}');

    devtools.log('Interpreter.run is called');
    // Run inference
    interpreter!.runForMultipleInputs([input], outputs);
    devtools.log('Interpreter.run is finished');

    // Get output tensors
    final rawBoxes = outputs[0]!;
    final rawScores = outputs[1]!;

    final List<dynamic> flatScores = List.filled(896, 0);
    for (var i = 0; i < rawScores[0].length; i++) {
      flatScores[i] = rawScores[0][i][0];
    }
    final flatScoresSorted = flatScores;
    flatScoresSorted.sort();
    devtools.log('Ten highest scores: ${flatScoresSorted.sublist(886)}');

    var detections = process(
      options: options,
      rawScores: rawScores[0],
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
