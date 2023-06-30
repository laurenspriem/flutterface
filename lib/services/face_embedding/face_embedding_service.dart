import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:math' as math show min, max;
import 'dart:typed_data' show Uint8List;

import 'package:flutterface/services/face_embedding/mobilefacenet_model_config.dart';
import 'package:flutterface/utils/image.dart';
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

  Future<void> loadModel() async {
    devtools.log('loadModel is called');

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

      // Load model from assets
      interpreter = interpreter ??
          await Interpreter.fromAsset(
            config.modelPath,
            options: interpreterOptions,
          );

      // Get tensor input shape [1, 112, 112, 3]
      final inputTensors = interpreter!.getInputTensors().first;
      devtools.log('Input Tensors: $inputTensors');
      // Get tensour output shape [1, 192]
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

  static num normalizePixel(num pixelValue) {
    return (pixelValue / 127.5) - 1;
  }

  List<List<List<num>>> getPreprocessedImage(
    image_lib.Image image,
  ) {
    devtools.log('Preprocessing is called');
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
    final imageMatrix = List.generate(
      image.height,
      (y) => List.generate(
        image.width,
        (x) {
          final pixel = image.getPixel(x, y);
          return [
            normalizePixel(pixel.r), // Normalize the image to range [-1, 1]
            normalizePixel(pixel.g), // Normalize the image to range [-1, 1]
            normalizePixel(pixel.b), // Normalize the image to range [-1, 1]
          ];
        },
      ),
    );
    devtools.log('Preprocessing is finished');

    return imageMatrix;
  }

  // TODO: Make the predict function asynchronous with use of isolate-interpreter
  List predict(Uint8List imageData) {
    assert(interpreter != null);

    final image = convertDataToImageImage(imageData);

    devtools.log('outputShapes: $outputShapes');

    final stopwatch = Stopwatch()..start();

    final inputImageMatrix =
        getPreprocessedImage(image); // [inputWidt, inputHeight, 3]
    final input = [inputImageMatrix];

    final output = createNestedList(outputShapes[0]);
    final outputs = <int, List>{
      0: output,
    };

    devtools.log('Input of shape ${input.shape}');
    devtools
        .log('Outputs: of shape ${outputs[0]?.shape} and ${outputs[1]?.shape}');

    devtools.log('Interpreter.run is called');
    // Run inference
    interpreter!.run(input, output);
    devtools.log('Interpreter.run is finished');

    // Get output tensors
    final embedding = output[0] as List;

    stopwatch.stop();
    devtools.log(
      'MobileFaceNet predict() executed in ${stopwatch.elapsedMilliseconds}ms',
    );

    devtools.log('FaceNet results: embedding $embedding');
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
