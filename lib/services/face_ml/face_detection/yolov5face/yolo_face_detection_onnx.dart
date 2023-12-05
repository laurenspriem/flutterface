import 'dart:typed_data' show Uint8List;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutterface/services/face_ml/face_detection/detection.dart';
import 'package:flutterface/services/face_ml/face_detection/naive_non_max_suppression.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_face_detection_exceptions.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_face_detection_options.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_filter_extract_detections.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_model_config.dart';
import 'package:flutterface/utils/image_ml_isolate.dart';
import 'package:logging/logging.dart';
import 'package:onnxruntime/onnxruntime.dart';

class YOLOFaceDetectionONNX {
  final _logger = Logger('YOLOFaceDetectionService');

  // Interpreter? _interpreter;
  // IsolateInterpreter? _isolateInterpreter;
  // int get getAddress => _interpreter!.address;

  // final outputShapes = <List<int>>[];
  // final outputTypes = <TensorType>[];

  OrtSessionOptions? _sessionOptions;
  OrtSession? _session;

  late final FaceDetectionOptionsYOLO _faceOptions;

  bool _isInitialized = false;

  final YOLOModelConfig config;
  // singleton pattern
  YOLOFaceDetectionONNX._privateConstructor({required this.config});

  /// Use this instance to access the FaceDetection service. Make sure to call `init()` before using it.
  /// e.g. `await FaceDetection.instance.init();`
  ///
  /// Then you can use `predict()` to get the bounding boxes of the faces, so `FaceDetection.instance.predict(imageData)`
  ///
  /// config options: yoloV5FaceN //
  static final instance = YOLOFaceDetectionONNX._privateConstructor(
    config: yoloV5FaceS480x640onnx,
  );
  factory YOLOFaceDetectionONNX() {
    OrtEnv.instance.init();
    return instance;
  }

  /// Check if the interpreter is initialized, if not initialize it with `loadModel()`
  Future<void> init() async {
    if (!_isInitialized) {
      await _loadModel();
    }
  }

  Future<void> dispose() async {
    if (_isInitialized) {
      _sessionOptions?.release();
      _sessionOptions = null;
      _session?.release();
      _session = null;
      OrtEnv.instance.release();

      _isInitialized = false;
    }
  }

  /// Detects faces in the given image data.
  Future<List<FaceDetectionRelative>> predict(Uint8List imageData) async {
    assert(_isInitialized && _session != null && _sessionOptions != null);

    final stopwatch = Stopwatch()..start();

    final stopwatchDecoding = Stopwatch()..start();
    final (inputImageList, originalSize, newSize) =
        await ImageMlIsolate.instance.preprocessImageYOLOonnx(
      imageData,
      normalize: true,
      requiredWidth: _faceOptions.inputWidth,
      requiredHeight: _faceOptions.inputHeight,
      maintainAspectRatio: true,
      quality: FilterQuality.medium,
    );
    // final input = [inputImageList];
    final inputShape = [
      1,
      3,
      _faceOptions.inputHeight,
      _faceOptions.inputWidth,
    ];
    final inputOrt = OrtValueTensor.createTensorWithDataList(
      inputImageList,
      inputShape,
    );
    final inputs = {'data': inputOrt};
    stopwatchDecoding.stop();
    _logger.info(
      'Image decoding and preprocessing is finished, in ${stopwatchDecoding.elapsedMilliseconds}ms',
    );
    _logger.info('original size: $originalSize \n new size: $newSize');

    _logger.info('interpreter.run is called');
    // Run inference
    final stopwatchInterpreter = Stopwatch()..start();
    List<OrtValue?>? outputs;
    try {
      final runOptions = OrtRunOptions();
      outputs = await _session?.runAsync(runOptions, inputs);
      inputOrt.release();
      runOptions.release();
    } catch (e, s) {
      _logger.severe('Error while running inference: $e \n $s');
      throw YOLOInterpreterRunException();
    }
    stopwatchInterpreter.stop();
    _logger.info(
      'interpreter.run is finished, in ${stopwatchInterpreter.elapsedMilliseconds} ms',
    );

    _logger.info('outputs: $outputs');

    // Get output tensors
    final stride_8 = outputs?[0]?.value
        as List<List<List<List<List<double>>>>>; // [1, 3, 60, 80, 16]
    final stride_16 = outputs?[1]?.value
        as List<List<List<List<List<double>>>>>; // [1, 3, 30, 40, 16]
    final stride_32 = outputs?[2]?.value
        as List<List<List<List<List<double>>>>>; // [1, 3, 15, 20, 16]
    outputs?.forEach((element) {
      element?.release();
    });

    var relativeDetections = filterExtractDetectionsYOLOonnx(
      options: _faceOptions,
      stride32: stride_32[0],
      stride16: stride_16[0],
      stride8: stride_8[0],
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
      OrtEnv.instance.init();
      OrtEnv.instance.availableProviders().forEach((element) {
        _logger.info('onnx provider= $element');
      });

      _sessionOptions = OrtSessionOptions();
      // ..setInterOpNumThreads(1)
      // ..setIntraOpNumThreads(1)
      // ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);
      final rawAssetFile = await rootBundle.load(config.modelPath);
      final bytes = rawAssetFile.buffer.asUint8List();
      _session = OrtSession.fromBuffer(bytes, _sessionOptions!);

      _isInitialized = true;
    } catch (e, s) {
      _logger.severe('Error while initializing YOLO onnx: $e \n $s');
      throw YOLOInterpreterInitializationException();
    }
  }
}
