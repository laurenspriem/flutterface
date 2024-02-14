import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:math' as math show min, max;
import 'dart:typed_data' show Uint8List;

import 'package:flutterface/services/face_ml/face_detection/detection.dart';
import 'package:flutterface/services/face_ml/face_embedding/face_embedding_exceptions.dart';
import 'package:flutterface/services/face_ml/face_embedding/face_embedding_options.dart';
import 'package:flutterface/services/face_ml/face_embedding/mobilefacenet_model_config.dart';
import 'package:flutterface/utils/image_ml_isolate.dart';
// import 'package:flutterface/utils/image.dart';
import 'package:flutterface/utils/image_ml_util.dart';
import 'package:logging/logging.dart';
// import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

/// This class is responsible for running the MobileFaceNet model, and can be accessed through the singleton `FaceEmbedding.instance`.
class FaceEmbedding {
  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  int get getAddress => _interpreter!.address;

  final outputShapes = <List<int>>[];
  final outputTypes = <TensorType>[];

  final _logger = Logger('FaceEmbeddingService');

  final MobileFaceNetModelConfig config;
  late final FaceEmbeddingOptions embeddingOptions =
      config.faceEmbeddingOptions;
  // singleton pattern
  FaceEmbedding._privateConstructor({required this.config});

  /// Use this instance to access the FaceEmbedding service. Make sure to call `init()` before using it.
  /// e.g. `await FaceEmbedding.instance.init();`
  ///
  /// Then you can use `predict()` to get the embedding of a face, so `FaceEmbedding.instance.predict(imageData)`
  ///
  /// config options: faceEmbeddingEnte
  static final instance =
      FaceEmbedding._privateConstructor(config: faceEmbeddingEnte);
  factory FaceEmbedding() => instance;

  /// Check if the interpreter is initialized, if not initialize it with `loadModel()`
  Future<void> init() async {
    if (_interpreter == null || _isolateInterpreter == null) {
      await _loadModel();
    }
  }

  // List<List<List<num>>> getPreprocessedImage(
  //   image_lib.Image image,
  // ) {
  //   final embeddingOptions = config.faceEmbeddingOptions;

  //   // Resize image for model input (112, 112) (thought most likely it is already resized, so we check first)
  //   if (image.width != embeddingOptions.inputWidth ||
  //       image.height != embeddingOptions.inputHeight) {
  //     image = image_lib.copyResize(
  //       image,
  //       width: embeddingOptions.inputWidth,
  //       height: embeddingOptions.inputHeight,
  //       interpolation: image_lib.Interpolation
  //           .linear, // can choose `bicubic` if more accuracy is needed. But this is slow, and adds little if bilinear is already used earlier (which is the case)
  //     );
  //   }

  //   // Get image matrix representation [inputWidt, inputHeight, 3]
  //   final imageMatrix = createInputMatrixFromImage(image, normalize: true);

  //   return imageMatrix;
  // }

  // TODO: Make the predict function asynchronous with use of isolate-interpreter: https://github.com/tensorflow/flutter-tflite/issues/52
  Future<(List<double>, bool, double)> predict(
    Uint8List imageData,
    FaceDetectionRelative face,
  ) async {
    assert(_interpreter != null && _isolateInterpreter != null);

    final stopwatchDecoding = Stopwatch()..start();
    final (inputImageMatrix, alignmentResults, isBlur, blurValue) =
        await ImageMlIsolate.instance.preprocessMobileFaceNet(
      imageData,
      [face],
    );
    final input = [inputImageMatrix[0]];
    stopwatchDecoding.stop();
    devtools.log(
      'MobileFaceNet image decoding and preprocessing is finished, in ${stopwatchDecoding.elapsedMilliseconds}ms',
    );

    devtools.log('MobileFaceNet outputShapes: $outputShapes');

    final stopwatch = Stopwatch()..start();

    final output = createEmptyOutputMatrix(outputShapes[0]);

    devtools.log('MobileFaceNet interpreter.run is called');
    // Run inference
    try {
      await _isolateInterpreter!.run(input, output);
      // ignore: avoid_catches_without_on_clauses
    } catch (e) {
      devtools.log('MobileFaceNet Error while running inference: $e');
      throw MobileFaceNetInterpreterRunException();
    }
    devtools.log('MobileFaceNet interpreter.run is finished');

    // Get output tensors
    final embedding = output[0] as List<double>;

    stopwatch.stop();
    devtools.log(
      'MobileFaceNet predict() executed in ${stopwatch.elapsedMilliseconds}ms',
    );

    devtools.log(
      'MobileFaceNet results (only first few numbers): embedding ${embedding.sublist(0, 5)}',
    );
    devtools.log(
      'Mean of embedding: ${embedding.reduce((a, b) => a + b) / embedding.length}',
    );
    devtools.log(
      'Max of embedding: ${embedding.reduce(math.max)}',
    );
    devtools.log(
      'Min of embedding: ${embedding.reduce(math.min)}',
    );

    return (embedding, isBlur[0], blurValue[0]);
  }

  Future<void> _loadModel() async {
    _logger.info('loadModel is called');

    try {
      final interpreterOptions = InterpreterOptions();

      // Android Delegates
      // TODO: Make sure this works on both platforms: Android and iOS
      if (Platform.isAndroid) {
        // Use GPU Delegate (GPU). WARNING: It doesn't work on emulator
        // interpreterOptions.addDelegate(GpuDelegateV2());
        // // Use XNNPACK Delegate (CPU)
        // interpreterOptions.addDelegate(XNNPackDelegate());
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

      // Get tensor input shape [1, 112, 112, 3]
      final inputTensors = _interpreter!.getInputTensors().first;
      _logger.info('Input Tensors: $inputTensors');
      // Get tensour output shape [1, 192]
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
    } catch (e) {
      _logger.severe('Error while creating interpreter: $e');
      throw MobileFaceNetInterpreterInitializationException();
    }
  }
}
