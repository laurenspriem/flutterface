import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:math' as math show min, max;
import 'dart:typed_data' show Uint8List;

import 'package:flutterface/services/face_embedding/mobilefacenet_model_config.dart';
import 'package:flutterface/utils/image.dart';
import 'package:flutterface/utils/ml_input_output.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

class FaceEmbedding {
  FaceEmbedding._({required this.config});

  final MobileFaceNetModelConfig config;

  static Future<FaceEmbedding> create() async {
    // In the line below, we can change the model to use
    final config =
        faceEmbeddingEnte; // faceEmbeddingEnte is the only config currently
    final faceEmbeddor = FaceEmbedding._(config: config);
    await faceEmbeddor.loadModel();
    return faceEmbeddor;
  }

  final outputShapes = <List<int>>[];
  final outputTypes = <TensorType>[];

  Interpreter? interpreter;

  List<Object> get props => [];

  int get getAddress => interpreter!.address;

  Future<void> loadModel() async {
    devtools.log('MobileFaceNet loadModel is called');

    try {
      final interpreterOptions = InterpreterOptions();

      // Use XNNPACK Delegate (CPU)
      if (Platform.isAndroid) {
        interpreterOptions.addDelegate(XNNPackDelegate());
      }

      // Use GPU Delegate
      // doesn't work on emulator
      if (Platform.isAndroid) {
        interpreterOptions.addDelegate(GpuDelegateV2());
      }

      // Use Metal Delegate
      if (Platform.isIOS) {
        interpreterOptions.addDelegate(GpuDelegate());
      }

      // Load model from assets
      interpreter = interpreter ??
          await Interpreter.fromAsset(
            config.modelPath,
            options: interpreterOptions,
          );

      // Get tensor input shape [1, 112, 112, 3]
      final inputTensors = interpreter!.getInputTensors().first;
      devtools.log('MobileFaceNet Input Tensors: $inputTensors');
      // Get tensour output shape [1, 192]
      final outputTensors = interpreter!.getOutputTensors();
      final outputTensor = outputTensors.first;
      devtools.log('MobileFaceNet Output Tensors: $outputTensor');

      for (var tensor in outputTensors) {
        outputShapes.add(tensor.shape);
        outputTypes.add(tensor.type);
      }
      devtools.log('MobileFaceNet loadModel is finished');
    } catch (e) {
      devtools.log('MobileFaceNet Error while creating interpreter: $e');
    }
  }

  List<List<List<num>>> getPreprocessedImage(
    image_lib.Image image,
  ) {
    final embeddingOptions = config.faceEmbeddingOptions;

    // Resize image for model input (112, 112) (thought most likely it is already resized, so we check first)
    if (image.width != embeddingOptions.inputWidth ||
        image.height != embeddingOptions.inputHeight) {
      image = image_lib.copyResize(
        image,
        width: embeddingOptions.inputWidth,
        height: embeddingOptions.inputHeight,
        interpolation: image_lib.Interpolation
            .linear, // can choose `bicubic` if more accuracy is needed. But this is slow, and adds little if bilinear is already used earlier (which is the case)
      );
    }

    // Get image matrix representation [inputWidt, inputHeight, 3]
    final imageMatrix = createInputMatrixFromImage(image, normalize: true);

    return imageMatrix;
  }

  // TODO: Make the predict function asynchronous with use of isolate-interpreter: https://github.com/tensorflow/flutter-tflite/issues/52
  List predict(Uint8List imageData) {
    assert(interpreter != null);

    final dataConversionStopwatch = Stopwatch()..start();
    final image = convertDataToImageImage(imageData);
    dataConversionStopwatch.stop();
    devtools.log(
        'MobileFaceNet image data conversion is finished, in ${dataConversionStopwatch.elapsedMilliseconds}ms');

    devtools.log('MobileFaceNet outputShapes: $outputShapes');

    final stopwatch = Stopwatch()..start();

    final inputImageMatrix =
        getPreprocessedImage(image); // [inputWidt, inputHeight, 3]
    final input = [inputImageMatrix];

    final output = createEmptyOutputMatrix(outputShapes[0]);

    devtools.log('MobileFaceNet interpreter.run is called');
    // Run inference
    interpreter!.run(input, output);
    devtools.log('MobileFaceNet interpreter.run is finished');

    // Get output tensors
    final embedding = output[0] as List;

    stopwatch.stop();
    devtools.log(
      'MobileFaceNet predict() executed in ${stopwatch.elapsedMilliseconds}ms',
    );

    devtools.log(
        'MobileFaceNet results (only first few numbers): embedding ${embedding.sublist(0, 5)}');
    devtools.log(
      'Mean of embedding: ${embedding.cast<num>().reduce((a, b) => a + b) / embedding.length}',
    );
    devtools.log(
      'Max of embedding: ${embedding.cast<num>().reduce(math.max)}',
    );
    devtools.log(
      'Min of embedding: ${embedding.cast<num>().reduce(math.min)}',
    );

    return embedding;
  }
}
