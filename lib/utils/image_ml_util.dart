import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data' show Uint8List, ByteData;
import 'dart:ui';

import 'package:flutter/material.dart' show debugPrint;
import 'package:flutter/painting.dart' as paint show decodeImageFromList;
import 'package:flutter_isolate/flutter_isolate.dart';
import 'package:flutterface/services/face_ml/face_alignment/similarity_transform.dart';
import 'package:flutterface/services/face_ml/face_detection/detection.dart';

import 'package:logging/logging.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:synchronized/synchronized.dart';

// TODO: remove these typedefs and import them instead
typedef Num3DInputMatrix = List<List<List<num>>>;

typedef Int3DInputMatrix = List<List<List<int>>>;

typedef Double3DInputMatrix = List<List<List<double>>>;

enum ImageOperation {
  preprocessStandard,
  preprocessFaceAlign,
  preprocessMobileFaceNet,
  generateFaceThumbnail
}

Color readPixelColor(
  Image image,
  ByteData byteData,
  int x,
  int y,
) {
  if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
    throw ArgumentError('Invalid pixel coordinates.');
    // return const Color(0x00000000);
  }
  assert(byteData.lengthInBytes == 4 * image.width * image.height);

  final int byteOffset = 4 * (image.width * y + x);
  return Color(_rgbaToArgb(byteData.getUint32(byteOffset)));
}

int _rgbaToArgb(int rgbaColor) {
  final int a = rgbaColor & 0xFF;
  final int rgb = rgbaColor >> 8;
  return rgb + (a << 24);
}

/// This class is responsible for all image operations needed for ML models. It runs in a separate isolate to avoid jank.
///
/// It can be accessed through the singleton `ImageConversionIsolate.instance`. e.g. `ImageConversionIsolate.instance.convert(imageData)`
///
/// IMPORTANT: Make sure to dispose of the isolate when you're done with it with `dispose()`, e.g. `ImageConversionIsolate.instance.dispose();`
class ImageMlIsolate {
  // static const String debugName = 'ImageMlIsolate';

  final _logger = Logger('ImageMlIsolate');

  Timer? _inactivityTimer;
  final Duration _inactivityDuration = const Duration(seconds: 30);

  final _initLock = Lock();

  late FlutterIsolate _isolate;
  late ReceivePort _receivePort = ReceivePort();
  late SendPort _mainSendPort;

  bool isSpawned = false;

  // singleton pattern
  ImageMlIsolate._privateConstructor();

  /// Use this instance to access the ImageConversionIsolate service. Make sure to call `init()` before using it.
  /// e.g. `await ImageConversionIsolate.instance.init();`
  /// And kill the isolate when you're done with it with `dispose()`, e.g. `ImageConversionIsolate.instance.dispose();`
  ///
  /// Then you can use `convert()` to get the image, so `ImageConversionIsolate.instance.convert(imageData, imagePath: imagePath)`
  static final ImageMlIsolate instance = ImageMlIsolate._privateConstructor();
  factory ImageMlIsolate() => instance;

  Future<void> init() async {
    return _initLock.synchronized(() async {
      if (isSpawned) return;

      _receivePort = ReceivePort();

      try {
        _isolate = await FlutterIsolate.spawn(
          _isolateMain,
          _receivePort.sendPort,
        );
        _mainSendPort = await _receivePort.first as SendPort;
        isSpawned = true;

        _resetInactivityTimer();
      } catch (e) {
        _logger.severe('Could not spawn isolate', e);
        isSpawned = false;
      }
    });
  }

  Future<void> ensureSpawned() async {
    if (!isSpawned) {
      await init();
    }
  }

  @pragma('vm:entry-point')
  static void _isolateMain(SendPort mainSendPort) async {
    final receivePort = ReceivePort();
    mainSendPort.send(receivePort.sendPort);

    receivePort.listen((message) async {
      final functionIndex = message[0] as int;
      final function = ImageOperation.values[functionIndex];
      final args = message[1] as Map<String, dynamic>;
      final sendPort = message[2] as SendPort;

      switch (function) {
        case ImageOperation.preprocessStandard:
          final imageData = args['imageData'] as Uint8List;
          final normalize = args['normalize'] as bool;
          final requiredWidth = args['requiredWidth'] as int;
          final requiredHeight = args['requiredHeight'] as int;
          final qualityIndex = args['quality'] as int;
          final quality = FilterQuality.values[qualityIndex];
          final Num3DInputMatrix result = await _preprocessImage(
            imageData,
            normalize: normalize,
            requiredWidth: requiredWidth,
            requiredHeight: requiredHeight,
            quality: quality,
          );
          sendPort.send(result);
        case ImageOperation.preprocessFaceAlign:
          final imageData = args['imageData'] as Uint8List;
          final faceLandmarks = args['faceLandmarks'] as List<List<List<int>>>;
          final List<Uint8List> result = await _preprocessFaceAlign(
            imageData,
            faceLandmarks,
          );
          sendPort.send(List.from(result));
        case ImageOperation.preprocessMobileFaceNet:
          final imageData = args['imageData'] as Uint8List;
          final faceLandmarks = args['faceLandmarks'] as List<List<List<int>>>;
          final List<Num3DInputMatrix> result = await _preprocessMobileFaceNet(
            imageData,
            faceLandmarks,
          );
          sendPort.send(result);
        case ImageOperation.generateFaceThumbnail:
          final imageData = args['imageData'] as Uint8List;
          final faceDetection = args['faceDetection'] as FaceDetectionRelative;
          final Uint8List result =
              await _generateFaceThumbnail(imageData, faceDetection);
          sendPort.send(<dynamic>[result]);
      }
    });
  }

  /// The common method to run any operation in the isolate. It sends the [message] to [_isolateMain] and waits for the result.
  Future<dynamic> _runInIsolate(
    (ImageOperation, Map<String, dynamic>) message,
  ) async {
    await ensureSpawned();
    _resetInactivityTimer();
    final completer = Completer<dynamic>();
    final answerPort = ReceivePort();

    _mainSendPort.send([message.$1.index, message.$2, answerPort.sendPort]);

    answerPort.listen((receivedMessage) {
      completer.complete(receivedMessage);
    });

    return completer.future;
  }

  /// Resets a timer that kills the isolate after a certain amount of inactivity.
  ///
  /// Should be called after initialization (e.g. inside `init()`) and after every call to isolate (e.g. inside `_runInIsolate()`)
  void _resetInactivityTimer() {
    _inactivityTimer?.cancel();
    _inactivityTimer = Timer(_inactivityDuration, () {
      _logger.info(
        'Isolate has been inactive for $_inactivityDuration. Killing isolate.',
      );
      dispose();
    });
  }

  /// Disposes the isolate worker.
  void dispose() {
    if (!isSpawned) return;

    isSpawned = false;
    _isolate.kill();
    _receivePort.close();
    _inactivityTimer?.cancel();
  }

  /// Preprocesses [imageData] for standard ML models inside a separate isolate.
  ///
  /// Returns a [Num3DInputMatrix] image usable for ML inference.
  ///
  /// Uses [_preprocessImage] inside the isolate.
  Future<Num3DInputMatrix> preprocessImage(
    Uint8List imageData, {
    required bool normalize,
    required int requiredWidth,
    required int requiredHeight,
    FilterQuality quality = FilterQuality.medium,
  }) async {
    return await _runInIsolate(
      (
        ImageOperation.preprocessStandard,
        {
          'imageData': imageData,
          'normalize': normalize,
          'requiredWidth': requiredWidth,
          'requiredHeight': requiredHeight,
          'quality': quality.index,
        },
      ),
    );
  }

  /// Preprocesses [imageData] for face alignment inside a separate isolate, to display the aligned faces. Mostly used for debugging.
  ///
  /// Returns a list of [Uint8List] images, one for each face, in png format.
  ///
  /// Uses [_preprocessFaceAlign] inside the isolate.
  ///
  /// WARNING: For preprocessing for MobileFaceNet, use [preprocessMobileFaceNet] instead!
  Future<List<Uint8List>> preprocessFaceAlign(
    Uint8List imageData,
    List<FaceDetectionAbsolute> faces,
  ) async {
    final faceLandmarks =
        faces.map((face) => face.allKeypoints.sublist(0, 4)).toList();
    return await _runInIsolate(
      (
        ImageOperation.preprocessFaceAlign,
        {
          'imageData': imageData,
          'faceLandmarks': faceLandmarks,
        },
      ),
    ).then((value) => value.cast<Uint8List>());
  }

  /// Preprocesses [imageData] for MobileFaceNet input inside a separate isolate.
  ///
  /// Returns a list of [Num3DInputMatrix] images, one for each face.
  ///
  /// Uses [_preprocessMobileFaceNet] inside the isolate.
  Future<List<Num3DInputMatrix>> preprocessMobileFaceNet(
    Uint8List imageData,
    List<FaceDetectionAbsolute> faces,
  ) async {
    final faceLandmarks =
        faces.map((face) => face.allKeypoints.sublist(0, 4)).toList();
    return await _runInIsolate(
      (
        ImageOperation.preprocessMobileFaceNet,
        {
          'imageData': imageData,
          'faceLandmarks': faceLandmarks,
        },
      ),
    );
  }

  /// Generates a face thumbnail from [imageData] and a [faceDetection].
  ///
  /// Uses [_generateFaceThumbnail] inside the isolate.
  Future<Uint8List> generateFaceThumbnail(
    Uint8List imageData,
    FaceDetectionRelative faceDetection,
  ) async {
    return await _runInIsolate(
      (
        ImageOperation.generateFaceThumbnail,
        {
          'imageData': imageData,
          'faceDetection': faceDetection,
        },
      ),
    ).then((value) => value[0] as Uint8List);
  }
}

/// Decodes [Uint8List] image data to an ui.[Image] object.
Future<Image> _decodeImageFromData(Uint8List imageData) async {
  final Image image = await paint.decodeImageFromList(imageData);
  return image;

  // final Codec codec = await instantiateImageCodecFromBuffer(
  //   await ImmutableBuffer.fromUint8List(imageData),
  // );
  // final FrameInfo frameInfo = await codec.getNextFrame();
  // return frameInfo.image;
}

/// Decodes [Uint8List] RGBA bytes to an ui.[Image] object.
Future<Image> _decodeImageFromRgbaBytes(
  Uint8List rgbaBytes,
  int width,
  int height,
) {
  final Completer<Image> completer = Completer();
  decodeImageFromPixels(
    rgbaBytes,
    width,
    height,
    PixelFormat.rgba8888,
    (Image image) {
      completer.complete(image);
    },
  );
  return completer.future;
}

/// Returns the [ByteData] object of the image, in rawRgba format.
///
/// Throws an exception if the image could not be converted to ByteData.
Future<ByteData> _getByteData(
  Image image, {
  ImageByteFormat format = ImageByteFormat.rawRgba,
}) async {
  final ByteData? byteDataRgba = await image.toByteData(format: format);
  if (byteDataRgba == null) {
    debugPrint('Could not convert image to ByteData');
    throw Exception('Could not convert image to ByteData');
  }
  return byteDataRgba;
}

/// Encodes an [Image] object to a [Uint8List], by default in the png format.
///
/// Note that the result can be used with `Image.memory()` only if the [format] is png.
Future<Uint8List> _encodeImage(
  Image image, {
  ImageByteFormat format = ImageByteFormat.png,
}) async {
  final ByteData byteDataPng = await _getByteData(image, format: format);
  final encodedImage = byteDataPng.buffer.asUint8List();

  return encodedImage;
}

/// Resizes an [Image] object to the specified [width] and [height].
Future<Image> _resizeImage(
  Image image,
  int width,
  int height, {
  FilterQuality quality = FilterQuality.medium,
}) async {
  if (image.width == width && image.height == height) {
    return image;
  }
  final recorder = PictureRecorder();
  final canvas = Canvas(
    recorder,
    Rect.fromPoints(
      const Offset(0, 0),
      Offset(width.toDouble(), height.toDouble()),
    ),
  );

  canvas.drawImageRect(
    image,
    Rect.fromPoints(
      const Offset(0, 0),
      Offset(image.width.toDouble(), image.height.toDouble()),
    ),
    Rect.fromPoints(
      const Offset(0, 0),
      Offset(width.toDouble(), height.toDouble()),
    ),
    Paint()..filterQuality = quality,
  );

  final picture = recorder.endRecording();
  return picture.toImage(width, height);
}

/// Crops an [Image] object to the specified [width] and [height], starting at the specified [x] and [y] coordinates.
Future<Image> _cropImage(
  Image image, {
  required int x,
  required int y,
  required int width,
  required int height,
  FilterQuality quality = FilterQuality.medium,
}) async {
  if (x < 0 ||
      y < 0 ||
      (x + width) > image.width ||
      (y + height) > image.height) {
    debugPrint('Invalid crop dimensions or coordinates.');
    throw ArgumentError('Invalid crop dimensions or coordinates.');
  }

  final recorder = PictureRecorder();
  final canvas = Canvas(
    recorder,
    Rect.fromPoints(
      const Offset(0, 0),
      Offset(width.toDouble(), height.toDouble()),
    ),
  );

  canvas.drawImageRect(
    image,
    Rect.fromPoints(
      Offset(x.toDouble(), y.toDouble()),
      Offset((x + width).toDouble(), (y + height).toDouble()),
    ),
    Rect.fromPoints(
      const Offset(0, 0),
      Offset(width.toDouble(), height.toDouble()),
    ),
    Paint()..filterQuality = quality,
  );

  final picture = recorder.endRecording();
  return picture.toImage(width, height);
}

/// Preprocesses [imageData] for standard ML models
Future<Num3DInputMatrix> _preprocessImage(
  Uint8List imageData, {
  required bool normalize,
  required int requiredWidth,
  required int requiredHeight,
  FilterQuality quality = FilterQuality.medium,
}) async {
  final Image image = await _decodeImageFromData(imageData);

  if (image.width == requiredWidth && image.height == requiredHeight) {
    final ByteData imgByteData = await _getByteData(image);
    return _createInputMatrixFromImage(image, imgByteData,
        normalize: normalize);
  }

  final Image resizedImage = await _resizeImage(
    image,
    requiredWidth,
    requiredHeight,
    quality: quality,
  );

  final ByteData imgByteData = await _getByteData(resizedImage);
  final Num3DInputMatrix imageMatrix = _createInputMatrixFromImage(
    resizedImage,
    imgByteData,
    normalize: normalize,
  );

  return imageMatrix;
}

/// Preprocesses [imageData] based on [faceLandmarks] to align the faces in the images.
///
/// Returns a list of [Uint8List] images, one for each face, in png format.
Future<List<Uint8List>> _preprocessFaceAlign(
  Uint8List imageData,
  List<List<List<int>>> faceLandmarks, {
  int width = 112,
  int height = 112,
}) async {
  final alignedImages = <Uint8List>[];
  final Image image = await _decodeImageFromData(imageData);
  final ByteData imgByteData =
      await _getByteData(image, format: ImageByteFormat.rawRgba);

  for (final faceLandmark in faceLandmarks) {
    final (transformationMatrix, correctlyEstimated) =
        SimilarityTransform.instance.estimate(faceLandmark);
    if (!correctlyEstimated) {
      alignedImages.add(Uint8List(0));
      continue;
    }
    final Uint8List alignedImageRGBA = await _warpAffineToUint8List(
      image,
      imgByteData,
      transformationMatrix,
      width: width,
      height: height,
    );
    final Image alignedImage =
        await _decodeImageFromRgbaBytes(alignedImageRGBA, width, height);
    final Uint8List alignedImagePng = await _encodeImage(alignedImage);

    alignedImages.add(alignedImagePng);
  }
  return alignedImages;
}

/// Preprocesses [imageData] based on [faceLandmarks] to align the faces in the images
///
/// Returns a list of [Num3DInputMatrix] images, one for each face, ready for MobileFaceNet inference
Future<List<Num3DInputMatrix>> _preprocessMobileFaceNet(
  Uint8List imageData,
  List<List<List<int>>> faceLandmarks, {
  int width = 112,
  int height = 112,
}) async {
  final alignedImages = <Num3DInputMatrix>[];
  final Image image = await _decodeImageFromData(imageData);
  final ByteData imgByteData = await _getByteData(image);

  for (final faceLandmark in faceLandmarks) {
    final (transformationMatrix, correctlyEstimated) =
        SimilarityTransform.instance.estimate(faceLandmark);
    if (!correctlyEstimated) {
      alignedImages.add([]);
      continue;
    }
    final Num3DInputMatrix alignedImage = await _warpAffineToMatrix(
      image,
      imgByteData,
      transformationMatrix,
      width: width,
      height: height,
      normalize: true,
    );
    alignedImages.add(alignedImage);
  }
  return alignedImages;
}

/// Function to warp an image [imageData] with an affine transformation using the estimated [transformationMatrix].
///
/// Returns the warped image in the specified width and height, in [Uint8List] RGBA format.
Future<Uint8List> _warpAffineToUint8List(
  Image inputImage,
  ByteData imgByteDataRgba,
  List<List<double>> transformationMatrix, {
  required int width,
  required int height,
}) async {
  final Uint8List outputList = Uint8List(4 * width * height);

  if (width != 112 || height != 112) {
    throw Exception(
      'Width and height must be 112, other transformations are not supported yet.',
    );
  }

  final A = Matrix.fromList([
    [transformationMatrix[0][0], transformationMatrix[0][1]],
    [transformationMatrix[1][0], transformationMatrix[1][1]],
  ]);
  final aInverse = A.inverse();
  // final aInverseMinus = aInverse * -1;
  final B = Vector.fromList(
    [transformationMatrix[0][2], transformationMatrix[1][2]],
  );
  final b00 = B[0];
  final b10 = B[1];
  final a00Prime = aInverse[0][0];
  final a01Prime = aInverse[0][1];
  final a10Prime = aInverse[1][0];
  final a11Prime = aInverse[1][1];

  for (int yTrans = 0; yTrans < height; ++yTrans) {
    for (int xTrans = 0; xTrans < width; ++xTrans) {
      // Perform inverse affine transformation (original implementation, intuitive but slow)
      // final X = aInverse * (Vector.fromList([xTrans, yTrans]) - B);
      // final X = aInverseMinus * (B - [xTrans, yTrans]);
      // final xList = X.asFlattenedList;
      // num xOrigin = xList[0];
      // num yOrigin = xList[1];

      // Perform inverse affine transformation (fast implementation, less intuitive)
      num xOrigin = (xTrans - b00) * a00Prime + (yTrans - b10) * a01Prime;
      num yOrigin = (xTrans - b00) * a10Prime + (yTrans - b10) * a11Prime;

      // Clamp to image boundaries
      xOrigin = xOrigin.clamp(0, inputImage.width - 1);
      yOrigin = yOrigin.clamp(0, inputImage.height - 1);

      // Bilinear interpolation
      final int x0 = xOrigin.floor();
      final int x1 = xOrigin.ceil();
      final int y0 = yOrigin.floor();
      final int y1 = yOrigin.ceil();

      // Get the original pixels
      final Color pixel1 = readPixelColor(inputImage, imgByteDataRgba, x0, y0);
      final Color pixel2 = readPixelColor(inputImage, imgByteDataRgba, x1, y0);
      final Color pixel3 = readPixelColor(inputImage, imgByteDataRgba, x0, y1);
      final Color pixel4 = readPixelColor(inputImage, imgByteDataRgba, x1, y1);

      // Calculate the weights for each pixel
      final fx = xOrigin - x0;
      final fy = yOrigin - y0;
      final fx1 = 1.0 - fx;
      final fy1 = 1.0 - fy;

      // Calculate the weighted sum of pixels
      final int r = _bilinearInterpolation(
        pixel1.red,
        pixel2.red,
        pixel3.red,
        pixel4.red,
        fx,
        fy,
        fx1,
        fy1,
      );
      final int g = _bilinearInterpolation(
        pixel1.green,
        pixel2.green,
        pixel3.green,
        pixel4.green,
        fx,
        fy,
        fx1,
        fy1,
      );
      final int b = _bilinearInterpolation(
        pixel1.blue,
        pixel2.blue,
        pixel3.blue,
        pixel4.blue,
        fx,
        fy,
        fx1,
        fy1,
      );

      // Set the new pixel
      outputList[4 * (yTrans * width + xTrans)] = r;
      outputList[4 * (yTrans * width + xTrans) + 1] = g;
      outputList[4 * (yTrans * width + xTrans) + 2] = b;
      outputList[4 * (yTrans * width + xTrans) + 3] = 255;
    }
  }

  return outputList;
}

/// Function to warp an image [imageData] with an affine transformation using the estimated [transformationMatrix].
///
/// Returns a [Num3DInputMatrix], potentially normalized (RGB) and ready to be used as input for a ML model.
Future<Num3DInputMatrix> _warpAffineToMatrix(
  Image inputImage,
  ByteData imgByteDataRgba,
  List<List<double>> transformationMatrix, {
  required int width,
  required int height,
  bool normalize = true,
}) async {
  final List<List<List<num>>> outputMatrix = List.generate(
    height,
    (y) => List.generate(
      width,
      (_) => List.filled(3, 0.0),
    ),
  );
  final num Function(num) pixelValue =
      normalize ? _normalizePixel : (num value) => value;

  if (width != 112 || height != 112) {
    throw Exception(
      'Width and height must be 112, other transformations are not supported yet.',
    );
  }

  final A = Matrix.fromList([
    [transformationMatrix[0][0], transformationMatrix[0][1]],
    [transformationMatrix[1][0], transformationMatrix[1][1]],
  ]);
  final aInverse = A.inverse();
  // final aInverseMinus = aInverse * -1;
  final B = Vector.fromList(
    [transformationMatrix[0][2], transformationMatrix[1][2]],
  );
  final b00 = B[0];
  final b10 = B[1];
  final a00Prime = aInverse[0][0];
  final a01Prime = aInverse[0][1];
  final a10Prime = aInverse[1][0];
  final a11Prime = aInverse[1][1];

  for (int yTrans = 0; yTrans < height; ++yTrans) {
    for (int xTrans = 0; xTrans < width; ++xTrans) {
      // Perform inverse affine transformation (original implementation, intuitive but slow)
      // final X = aInverse * (Vector.fromList([xTrans, yTrans]) - B);
      // final X = aInverseMinus * (B - [xTrans, yTrans]);
      // final xList = X.asFlattenedList;
      // num xOrigin = xList[0];
      // num yOrigin = xList[1];

      // Perform inverse affine transformation (fast implementation, less intuitive)
      num xOrigin = (xTrans - b00) * a00Prime + (yTrans - b10) * a01Prime;
      num yOrigin = (xTrans - b00) * a10Prime + (yTrans - b10) * a11Prime;

      // Clamp to image boundaries
      xOrigin = xOrigin.clamp(0, inputImage.width - 1);
      yOrigin = yOrigin.clamp(0, inputImage.height - 1);

      // Bilinear interpolation
      final int x0 = xOrigin.floor();
      final int x1 = xOrigin.ceil();
      final int y0 = yOrigin.floor();
      final int y1 = yOrigin.ceil();

      // Get the original pixels
      final Color pixel1 = readPixelColor(inputImage, imgByteDataRgba, x0, y0);
      final Color pixel2 = readPixelColor(inputImage, imgByteDataRgba, x1, y0);
      final Color pixel3 = readPixelColor(inputImage, imgByteDataRgba, x0, y1);
      final Color pixel4 = readPixelColor(inputImage, imgByteDataRgba, x1, y1);

      // Calculate the weights for each pixel
      final fx = xOrigin - x0;
      final fy = yOrigin - y0;
      final fx1 = 1.0 - fx;
      final fy1 = 1.0 - fy;

      // Calculate the weighted sum of pixels
      final int r = _bilinearInterpolation(
        pixel1.red,
        pixel2.red,
        pixel3.red,
        pixel4.red,
        fx,
        fy,
        fx1,
        fy1,
      );
      final int g = _bilinearInterpolation(
        pixel1.green,
        pixel2.green,
        pixel3.green,
        pixel4.green,
        fx,
        fy,
        fx1,
        fy1,
      );
      final int b = _bilinearInterpolation(
        pixel1.blue,
        pixel2.blue,
        pixel3.blue,
        pixel4.blue,
        fx,
        fy,
        fx1,
        fy1,
      );

      // Set the new pixel
      outputMatrix[yTrans][xTrans] = [
        pixelValue(r),
        pixelValue(g),
        pixelValue(b),
      ];
    }
  }

  return outputMatrix;
}

/// Generates a face thumbnail from [imageData] and a [faceDetection].
///
/// Returns a [Uint8List] image, in png format.
Future<Uint8List> _generateFaceThumbnail(
  Uint8List imageData,
  FaceDetectionRelative faceDetection,
) async {
  final Image image = await _decodeImageFromData(imageData);

  final Image faceThumbnail = await _cropImage(
    image,
    x: (faceDetection.xMinBox * image.width).round() - 20,
    y: (faceDetection.yMinBox * image.height).round() - 30,
    width: (faceDetection.width * image.width).round() + 40,
    height: (faceDetection.height * image.height).round() + 60,
  );

  return await _encodeImage(faceThumbnail, format: ImageByteFormat.png);
}

/// Creates an input matrix from the specified image, which can be used for inference
///
/// Returns a matrix with the shape [image.height, image.width, 3], where the third dimension represents the RGB channels, as [Num3DInputMatrix].
/// In fact, this is either a [Double3DInputMatrix] or a [Int3DInputMatrix] depending on the `normalize` argument.
/// If `normalize` is true, the pixel values are normalized doubles in range [-1, 1]. Otherwise, they are integers in range [0, 255].
///
/// The `image` argument must be an [Image] object. The function returns a matrix
/// with the shape `[image.height, image.width, 3]`, where the third dimension
/// represents the RGB channels.
///
/// bool `normalize`: Normalize the image to range [-1, 1]
Num3DInputMatrix _createInputMatrixFromImage(
  Image image,
  ByteData byteDataRgba, {
  bool normalize = true,
}) {
  return List.generate(
    image.height,
    (y) => List.generate(
      image.width,
      (x) {
        final pixel = readPixelColor(image, byteDataRgba, x, y);
        return [
          normalize ? _normalizePixel(pixel.red) : pixel.red,
          normalize ? _normalizePixel(pixel.green) : pixel.green,
          normalize ? _normalizePixel(pixel.blue) : pixel.blue,
        ];
      },
    ),
  );
}

/// Creates an input matrix from the specified image, which can be used for inference
///
/// Returns a matrix with the shape `[3, image.height, image.width]`, where the first dimension represents the RGB channels, as [Num3DInputMatrix].
/// In fact, this is either a [Double3DInputMatrix] or a [Int3DInputMatrix] depending on the `normalize` argument.
/// If `normalize` is true, the pixel values are normalized doubles in range [-1, 1]. Otherwise, they are integers in range [0, 255].
///
/// The `image` argument must be an [Image] object. The function returns a matrix
/// with the shape `[3, image.height, image.width]`, where the first dimension
/// represents the RGB channels.
///
/// bool `normalize`: Normalize the image to range [-1, 1]
Num3DInputMatrix _createInputMatrixFromImageChannelsFirst(
  Image image,
  ByteData byteDataRgba, {
  bool normalize = true,
}) {
  // Create an empty 3D list.
  final Num3DInputMatrix imageMatrix = List.generate(
    3,
    (i) => List.generate(
      image.height,
      (j) => List.filled(image.width, 0),
    ),
  );

  // Determine which function to use to get the pixel value.
  final pixelValue = normalize ? _normalizePixel : (num value) => value;

  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      // Get the pixel at (x, y).
      final pixel = readPixelColor(image, byteDataRgba, x, y);

      // Assign the color channels to the respective lists.
      imageMatrix[0][y][x] = pixelValue(pixel.red);
      imageMatrix[1][y][x] = pixelValue(pixel.green);
      imageMatrix[2][y][x] = pixelValue(pixel.blue);
    }
  }
  return imageMatrix;
}

/// Function normalizes the pixel value to be in range [-1, 1].
///
/// It assumes that the pixel value is originally in range [0, 255]
double _normalizePixel(num pixelValue) {
  return (pixelValue / 127.5) - 1;
}

int _bilinearInterpolation(
  num val1,
  num val2,
  num val3,
  num val4,
  num fx,
  num fy,
  num fx1,
  num fy1,
) {
  return (val1 * fx1 * fy1 + val2 * fx * fy1 + val3 * fx1 * fy + val4 * fx * fy)
      .round();
}
