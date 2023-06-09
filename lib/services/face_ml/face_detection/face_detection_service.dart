import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:typed_data' show Uint8List;

import 'package:flutterface/services/face_ml/face_detection/anchors.dart';
import 'package:flutterface/services/face_ml/face_detection/blazeface_model_config.dart';
import 'package:flutterface/services/face_ml/face_detection/detection.dart';
import 'package:flutterface/services/face_ml/face_detection/face_detection_exceptions.dart';
import 'package:flutterface/services/face_ml/face_detection/filter_extract_detections.dart';
import 'package:flutterface/services/face_ml/face_detection/generate_anchors.dart';
import 'package:flutterface/services/face_ml/face_detection/naive_non_max_suppression.dart';
import 'package:flutterface/utils/image.dart';
import 'package:flutterface/utils/ml_input_output.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

class FaceDetection {
  Interpreter? _interpreter;
  int get getAddress => _interpreter!.address;

  final outputShapes = <List<int>>[];
  final outputTypes = <TensorType>[];

  late List<Anchor> _anchors;
  late int originalImageWidth;
  late int originalImageHeight;

  final BlazeFaceModelConfig config;
  // singleton pattern
  FaceDetection._privateConstructor({required this.config});

  /// Use this instance to access the FaceDetection service. Make sure to call `init()` before using it.
  /// e.g. `await FaceDetection.instance.init();`
  ///
  /// Then you can use `predict()` to get the bounding boxes of the faces, so `FaceDetection.instance.predict(imageData)`
  ///
  /// config options: faceDetectionFront // faceDetectionBackWeb // faceDetectionShortRange //faceDetectionFullRangeSparse; // faceDetectionFullRangeDense (faster than web while still accurate)
  static final instance = FaceDetection._privateConstructor(config: faceDetectionBackWeb);

  /// Check if the interpreter is initialized, if not initialize it with `loadModel()`
  Future<void> init() async {
    if (_interpreter == null) {
      await loadModel();
    }
  }

  /// Initialize the interpreter by loading the model file.
  Future<void> loadModel() async {
    devtools.log('BlazeFace.loadModel is called');

    final anchorOption = config.anchorOptions;

    try {
      final interpreterOptions = InterpreterOptions();

      // Use XNNPACK Delegate (CPU)
      if (Platform.isAndroid) {
        interpreterOptions.addDelegate(XNNPackDelegate());
      }

      // Use GPU Delegate (GPU)
      // doesn't work on emulator
      if (Platform.isAndroid) {
        interpreterOptions.addDelegate(GpuDelegateV2());
      }

      // Use Metal Delegate
      if (Platform.isIOS) {
        interpreterOptions.addDelegate(GpuDelegate());
      }

      // Create anchor boxes for BlazeFace
      _anchors = generateAnchors(anchorOption);

      // Load model from assets
      _interpreter = _interpreter ??
          await Interpreter.fromAsset(
            config.modelPath,
            options: interpreterOptions,
          );

      // Get tensor input shape [1, 128, 128, 3]
      final inputTensors = _interpreter!.getInputTensors().first;
      devtools.log('BlazeFace Input Tensors: $inputTensors');
      // Get tensour output shape [1, 896, 16]
      final outputTensors = _interpreter!.getOutputTensors();
      final outputTensor = outputTensors.first;
      devtools.log('BlazeFace Output Tensors: $outputTensor');

      for (var tensor in outputTensors) {
        outputShapes.add(tensor.shape);
        outputTypes.add(tensor.type);
      }
      devtools.log('BlazeFace.loadModel is finished');
    // ignore: avoid_catches_without_on_clauses
    } catch (e) {
      devtools.log('BlazeFace Error while creating interpreter: $e');
      throw BlazeFaceInterpreterInitializationException();
    }
  }

  List<List<List<num>>> getPreprocessedImage(
    image_lib.Image image,
  ) {
    devtools.log('BlazeFace preprocessing is called');
    final faceOptions = config.faceOptions;

    originalImageWidth = image.width;
    originalImageHeight = image.height;
    devtools.log(
      'originalImageWidth: $originalImageWidth, originalImageHeight: $originalImageHeight',
    );

    // Resize image for model input
    final image_lib.Image imageInput = image_lib.copyResize(
      image,
      width: faceOptions.inputWidth,
      height: faceOptions.inputHeight,
      interpolation: image_lib.Interpolation
          .linear, // linear interpolation is less accurate than cubic, but faster!
    );

    // Get image matrix representation [inputWidt, inputHeight, 3]
    final imageMatrix = createInputMatrixFromImage(imageInput, normalize: true);

    devtools.log('BlazeFace preprocessing is finished');

    return imageMatrix;
  }

  // TODO: Make the predict function asynchronous with use of isolate-interpreter: https://github.com/tensorflow/flutter-tflite/issues/52
  List<FaceDetectionAbsolute> predict(Uint8List imageData) {
    assert(_interpreter != null);

    final image = convertUint8ListToImagePackageImage(imageData);

    final faceOptions = config.faceOptions;
    devtools.log('BlazeFace outputShapes: $outputShapes');

    final stopwatch = Stopwatch()..start();

    final inputImageMatrix =
        getPreprocessedImage(image); // [inputWidt, inputHeight, 3]
    final input = [inputImageMatrix];

    final outputFaces = createEmptyOutputMatrix(outputShapes[0]);
    final outputScores = createEmptyOutputMatrix(outputShapes[1]);
    final outputs = <int, List>{
      0: outputFaces,
      1: outputScores,
    };

    devtools.log('Input of shape ${input.shape}');
    devtools
        .log('Outputs: of shape ${outputs[0]?.shape} and ${outputs[1]?.shape}');

    devtools.log('BlazeFace interpreter.run is called');
    // Run inference
    final secondStopwatch = Stopwatch()..start();
    try {
      _interpreter!.runForMultipleInputs([input], outputs);
    // ignore: avoid_catches_without_on_clauses
    } catch (e) {
      devtools.log('BlazeFace Error while running inference: $e');
      throw BlazeFaceInterpreterRunException();
    }
    secondStopwatch.stop();
    devtools.log(
        'BlazeFace interpreter.run is finished, in ${secondStopwatch.elapsedMilliseconds} ms');

    // Get output tensors
    final rawBoxes = outputs[0]![0]; // Nested List of shape [896, 16]
    final rawScores = outputs[1]![0]; // Nested List of shape [896, 1]

    // // Visually inspecting the raw scores
    // final List<dynamic> flatScores = List.filled(896, 0);
    // for (var i = 0; i < rawScores.length; i++) {
    //   flatScores[i] = rawScores[i][0];
    // }
    // final flatScoresSorted = flatScores;
    // flatScoresSorted.sort();
    // devtools.log('Ten highest (raw) scores: ${flatScoresSorted.sublist(886)}');

    // // Visually inspecting the raw boxes
    // final List<dynamic> flatBoxesFirstCoordinates = List.filled(896, 0);
    // final List<dynamic> flatBoxesSecondCoordinates = List.filled(896, 0);
    // final List<dynamic> flatBoxesThirdCoordinates = List.filled(896, 0);
    // final List<dynamic> flatBoxesFourthCoordinates = List.filled(896, 0);
    // for (var i = 0; i < rawBoxes[0].length; i++) {
    //   flatBoxesFirstCoordinates[i] = rawBoxes[i][0];
    //   flatBoxesSecondCoordinates[i] = rawBoxes[i][1];
    //   flatBoxesThirdCoordinates[i] = rawBoxes[i][2];
    //   flatBoxesFourthCoordinates[i] = rawBoxes[i][3];
    // }
    // devtools.log('rawBoxesFirstCoordinates: $flatBoxesFirstCoordinates');

    var relativeDetections = filterExtractDetections(
      options: faceOptions,
      rawScores: rawScores,
      rawBoxes: rawBoxes,
      anchors: _anchors,
    );

    relativeDetections = naiveNonMaxSuppression(
      detections: relativeDetections,
      iouThreshold: faceOptions.iouThreshold,
    );

    if (relativeDetections.isEmpty) {
      devtools.log('No face detected');
      return [
        FaceDetectionAbsolute.zero(),
      ];
    }

    final absoluteDetections = relativeToAbsoluteDetections(
      detections: relativeDetections,
      originalWidth: originalImageWidth,
      originalHeight: originalImageHeight,
    );

    stopwatch.stop();
    devtools.log(
        'BlazeFace.predict() face detection executed in ${stopwatch.elapsedMilliseconds}ms',);

    return absoluteDetections;
  }
}
