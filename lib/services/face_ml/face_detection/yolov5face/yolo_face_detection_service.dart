import 'dart:io';
import 'dart:typed_data' show Uint8List;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutterface/services/face_ml/face_detection/detection.dart';
import 'package:flutterface/services/face_ml/face_detection/naive_non_max_suppression.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_face_detection_exceptions.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_face_detection_options.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_filter_extract_detections.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_model_config.dart';
import 'package:flutterface/utils/image_ml_isolate.dart';
import 'package:flutterface/utils/image_ml_util.dart';
import 'package:logging/logging.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class YOLOFaceDetection {
  final _logger = Logger('YOLOFaceDetectionService');

  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  int get getAddress => _interpreter!.address;

  final outputShapes = <List<int>>[];
  final outputTypes = <TensorType>[];

  late final FaceDetectionOptionsYOLO _faceOptions;

  final YOLOModelConfig config;
  // singleton pattern
  YOLOFaceDetection._privateConstructor({required this.config});

  /// Use this instance to access the FaceDetection service. Make sure to call `init()` before using it.
  /// e.g. `await FaceDetection.instance.init();`
  ///
  /// Then you can use `predict()` to get the bounding boxes of the faces, so `FaceDetection.instance.predict(imageData)`
  ///
  /// config options: yoloV5FaceN //
  static final instance =
      YOLOFaceDetection._privateConstructor(config: yoloV5FaceS480x640tflite);
  factory YOLOFaceDetection() => instance;

  /// Check if the interpreter is initialized, if not initialize it with `loadModel()`
  Future<void> init() async {
    if (_interpreter == null || _isolateInterpreter == null) {
      await _loadModel();
    }
  }

  /// Detects faces in the given image data.
  Future<List<FaceDetectionRelative>> predict(Uint8List imageData) async {
    assert(_interpreter != null && _isolateInterpreter != null);

    final stopwatch = Stopwatch()..start();

    final stopwatchDecoding = Stopwatch()..start();
    final (inputImageMatrix, originalSize, newSize) =
        await ImageMlIsolate.instance.preprocessImageYOLOtflite(
      imageData,
      normalize: true,
      requiredWidth: _faceOptions.inputWidth,
      requiredHeight: _faceOptions.inputHeight,
      maintainAspectRatio: true,
      quality: FilterQuality.medium,
    );
    final input = [inputImageMatrix];
    stopwatchDecoding.stop();
    _logger.info(
      'Image decoding and preprocessing is finished, in ${stopwatchDecoding.elapsedMilliseconds}ms',
    );
    _logger.info('original size: $originalSize \n new size: $newSize');

    final layer1 = createEmptyOutputMatrix(outputShapes[0]);
    final layer2 = createEmptyOutputMatrix(outputShapes[1]);
    final layer3 = createEmptyOutputMatrix(outputShapes[2]);
    final outputs = <int, List>{
      0: layer1,
      1: layer2,
      2: layer3,
    };

    _logger.info('interpreter.run is called');
    // Run inference
    final stopwatchInterpreter = Stopwatch()..start();
    try {
      await _isolateInterpreter!.runForMultipleInputs([input], outputs);
    } catch (e, s) {
      _logger.severe('Error while running inference: $e \n $s');
      throw YOLOInterpreterRunException();
    }
    stopwatchInterpreter.stop();
    _logger.info(
      'interpreter.run is finished, in ${stopwatchInterpreter.elapsedMilliseconds} ms',
    );

    // Get output tensors
    final stride32 = outputs[0]![0]; // Nested List of shape [8, 10, 16, 3]
    final stride16 = outputs[1]![0]; // Nested List of shape [16, 20, 16, 3]
    final stride8 = outputs[2]![0]; // Nested List of shape [32, 40, 16, 3]

    var relativeDetections = filterExtractDetectionsYOLOtfliteDebug(
      options: _faceOptions,
      stride32: stride32,
      stride16: stride16,
      stride8: stride8,
    );

    // Account for the fact that the aspect ratio was maintained
    for (final faceDetection in relativeDetections) {
      faceDetection.correctForMaintainedAspectRatio(
        Size(
          _faceOptions.inputWidth.toDouble(),
          _faceOptions.inputHeight.toDouble(),
        ),
        newSize,
      );
    }

    relativeDetections = naiveNonMaxSuppression(
      detections: relativeDetections,
      iouThreshold: _faceOptions.iouThreshold,
    );

    if (relativeDetections.isEmpty) {
      _logger.info('No face detected');
      return <FaceDetectionRelative>[];
    }

    stopwatch.stop();
    _logger.info(
      'predict() face detection executed in ${stopwatch.elapsedMilliseconds}ms',
    );

    return relativeDetections;
  }

  /// Initialize the interpreter by loading the model file.
  Future<void> _loadModel() async {
    _logger.info('loadModel is called');

    _faceOptions = config.faceOptions;

    try {
      final interpreterOptions = InterpreterOptions();

      // Android Delegates
      // TODO: Make sure this works on both platforms: Android and iOS
      if (Platform.isAndroid) {
        // Use GPU Delegate (GPU). WARNING: It doesn't work on emulator. And doesn't speed up current version of BlazeFace used.
        interpreterOptions.addDelegate(GpuDelegateV2());
        // Use XNNPACK Delegate (CPU)
        interpreterOptions.addDelegate(XNNPackDelegate());
      }

      // iOS Delegates
      if (Platform.isIOS) {
        // Use Metal Delegate (GPU)
        interpreterOptions.addDelegate(GpuDelegate());
      }

      // Load model from assets
      _interpreter ??= await Interpreter.fromAsset(
        config.modelPath,
        options: interpreterOptions,
      );
      _isolateInterpreter ??=
          IsolateInterpreter(address: _interpreter!.address);

      _logger.info('Interpreter created from asset: ${config.modelPath}');

      // Get tensor input shape [1, 128, 128, 3]
      final inputTensors = _interpreter!.getInputTensors().first;
      _logger.info('Input Tensors: $inputTensors');
      // Get tensour output shape [1, 896, 16]
      final outputTensors = _interpreter!.getOutputTensors();
      final outputTensor = outputTensors.first;
      _logger.info('Output Tensors: $outputTensor');

      for (var tensor in outputTensors) {
        outputShapes.add(tensor.shape);
        outputTypes.add(tensor.type);
      }
      _logger.info('outputShapes: $outputShapes');
      _logger.info('loadModel is finished');
      // ignore: avoid_catches_without_on_clauses
    } catch (e, s) {
      _logger.severe('Error while initializing YOLO interpreter: $e \n $s');
      throw YOLOInterpreterInitializationException();
    }
  }
}
