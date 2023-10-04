import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data' show Uint8List, ByteData;
import 'dart:ui'; // as ui show Image, ImageByteFormat, Color;

// import 'package:flutter/material.dart' show WidgetsFlutterBinding;
import 'package:flutter/painting.dart' as paint show decodeImageFromList;
import 'package:flutter_isolate/flutter_isolate.dart';
import 'package:flutterface/services/face_ml/face_detection/detection.dart';

import 'package:logging/logging.dart';
import 'package:synchronized/synchronized.dart';

// TODO: remove these typedefs and import them instead
typedef Num3DInputMatrix = List<List<List<num>>>;

typedef Int3DInputMatrix = List<List<List<int>>>;

typedef Double3DInputMatrix = List<List<List<double>>>;

class CouldNotConvertToImageImage implements Exception {}

enum ImageOperation {
  decode,
  encode,
  resize,
  crop,
  preprocess,
  generateFaceThumbnail
}

Color readPixelColor(
  Image image,
  ByteData byteData,
  int x,
  int y,
) {
  if (x < 0 || x >= image.width || y < 0 || y >= image.height) {
    return const Color(0x00000000);
  }
  final int byteOffset = 4 * (image.width * y + x);
  return Color(_rgbaToArgb(byteData.getUint32(byteOffset)));
}

int _rgbaToArgb(int rgbaColor) {
  final int a = rgbaColor & 0xFF;
  final int rgb = rgbaColor >> 8;
  return rgb + (a << 24);
}

/// This class is responsible for converting [Uint8List] to [image_lib.Image].
///
/// Used primarily for ML applications.
class ImageConversionIsolate {
  // static const String debugName = 'ImageMlIsolate';

  final _logger = Logger('ImageMlIsolate');

  final _initLock = Lock();

  late FlutterIsolate _isolate;
  late ReceivePort _receivePort = ReceivePort();
  late SendPort _mainSendPort;

  bool isSpawned = false;

  // singleton pattern
  ImageConversionIsolate._privateConstructor();

  /// Use this instance to access the ImageConversionIsolate service. Make sure to call `init()` before using it.
  /// e.g. `await ImageConversionIsolate.instance.init();`
  /// And kill the isolate when you're done with it with `dispose()`, e.g. `ImageConversionIsolate.instance.dispose();`
  ///
  /// Then you can use `convert()` to get the image, so `ImageConversionIsolate.instance.convert(imageData, imagePath: imagePath)`
  static final ImageConversionIsolate instance =
      ImageConversionIsolate._privateConstructor();
  factory ImageConversionIsolate() => instance;

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
      final function = message[0] as ImageOperation;
      final args = message[1] as Map<String, dynamic>;
      final sendPort = message[2] as SendPort;

      switch (function) {
        case ImageOperation.decode:
          final imageData = args['imageData'] as Uint8List;
          final includebyteDataRgba = args['includebyteDataRgba'] as bool;
          final Image image = await _decodeImage(imageData);
          if (!includebyteDataRgba) {
            sendPort.send({'image': image});
          } else {
            final byteDataRgba = await _getByteDataRgba(image);
            sendPort.send({
              'image': image,
              'byteDataRgba': byteDataRgba,
            });
          }
        case ImageOperation.encode:
          final image = args['image'] as Image;
          final Uint8List result = await _encodeImage(image);
          sendPort.send(result);
        case ImageOperation.resize:
          final image = args['image'] as Image;
          final width = args['width'] as int;
          final height = args['height'] as int;
          final quality = args['quality'] as FilterQuality;
          final Image result =
              await _resizeImage(image, width, height, quality: quality);
          sendPort.send(result);
        case ImageOperation.crop:
          final image = args['image'] as Image;
          final x = args['x'] as int;
          final y = args['y'] as int;
          final width = args['width'] as int;
          final height = args['height'] as int;
          final quality = args['quality'] as FilterQuality;
          final Image result = await _cropImage(
            image,
            x: x,
            y: y,
            width: width,
            height: height,
            quality: quality,
          );
          sendPort.send(result);
        case ImageOperation.preprocess:
          final imageData = args['imageData'] as Uint8List;
          final normalize = args['normalize'] as bool;
          final requiredWidth = args['requiredWidth'] as int;
          final requiredHeight = args['requiredHeight'] as int;
          final quality = args['quality'] as FilterQuality;
          final Num3DInputMatrix result = await _preprocessImage(
            imageData,
            normalize: normalize,
            requiredWidth: requiredWidth,
            requiredHeight: requiredHeight,
            quality: quality,
          );
          sendPort.send(result);
        case ImageOperation.generateFaceThumbnail:
          final imageData = args['imageData'] as Uint8List;
          final faceDetection = args['faceDetection'] as FaceDetectionRelative;
          final Uint8List result =
              await _generateFaceThumbnail(imageData, faceDetection);
          sendPort.send(result);
      }
    });
  }

  /// The common method to run any operation in the isolate. It sends the [message] to [_isolateMain] and waits for the result.
  Future<dynamic> _runInIsolate(
    (ImageOperation, Map<String, dynamic>) message,
  ) async {
    await ensureSpawned();
    final completer = Completer<dynamic>();
    final answerPort = ReceivePort();

    _mainSendPort.send([message.$1, message.$2, answerPort.sendPort]);

    answerPort.listen((receivedMessage) {
      completer.complete(receivedMessage);
    });

    return completer.future;
  }

  /// Converts a [Uint8List] to an ui.[Image] object inside a separate isolate.
  Future<({Image image, ByteData? byteDataRgba})> decode(
    Uint8List imageData, {
    bool includebyteDataRgba = false,
  }) async {
    final Map results = await _runInIsolate(
      (
        ImageOperation.decode,
        {
          'imageData': imageData,
          'includebyteDataRgba': includebyteDataRgba,
        },
      ),
    );
    return (
      image: results['image'] as Image,
      byteDataRgba: results['byteDataRgba'] as ByteData
    );
  }

  /// Encodes an [Image] object to a [Uint8List] inside a separate isolate
  ///
  /// Note that it is not in jpg format, but can still be used with `Image.memory()`.
  Future<Uint8List> encode(Image image) async {
    return await _runInIsolate(
      (
        ImageOperation.encode,
        {'image': image},
      ),
    );
  }

  /// Resizes an [Image] object to the specified [width] and [height] inside a separate isolate.
  Future<Image> resize(
    Image image,
    int width,
    int height, {
    FilterQuality quality = FilterQuality.medium,
  }) async {
    return await _runInIsolate(
      (
        ImageOperation.resize,
        {
          'image': image,
          'width': width,
          'height': height,
          'quality': quality,
        },
      ),
    );
  }

  /// Crops an [Image] object to the specified [width] and [height] from upper left corner ([x], [y]) inside a separate isolate.
  Future<Image> crop(
    Image image, {
    required int x,
    required int y,
    required int width,
    required int height,
    FilterQuality quality = FilterQuality.medium,
  }) async {
    return await _runInIsolate(
      (
        ImageOperation.crop,
        {
          'image': image,
          'x': x,
          'y': y,
          'width': width,
          'height': height,
          'quality': quality,
        },
      ),
    );
  }

  /// Preprocesses [imageData] for standard ML models inside a separate isolate.
  Future<Num3DInputMatrix> preprocessImage(
    Uint8List imageData, {
    required bool normalize,
    required int requiredWidth,
    required int requiredHeight,
    FilterQuality quality = FilterQuality.medium,
  }) async {
    return await _runInIsolate(
      (
        ImageOperation.preprocess,
        {
          'imageData': imageData,
          'normalize': normalize,
          'requiredWidth': requiredWidth,
          'requiredHeight': requiredHeight,
          'quality': quality,
        },
      ),
    );
  }

  /// Generates a face thumbnail from [imageData] and a [faceDetection].
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
    );
  }

  /// Disposes the isolate worker.
  void dispose() {
    if (!isSpawned) return;

    isSpawned = false;
    _isolate.kill();
    _receivePort.close();
  }
}

/// Decodes a [Uint8List] to an ui.[Image] object.
Future<Image> _decodeImage(Uint8List imageData) async {
  final Image image = await paint.decodeImageFromList(imageData);
  return image;

  // final Codec codec = await instantiateImageCodecFromBuffer(
  //   await ImmutableBuffer.fromUint8List(imageData),
  // );
  // final FrameInfo frameInfo = await codec.getNextFrame();
  // return frameInfo.image;
}

/// Returns the [ByteData] object of the image, in rawRgba format.
///
/// Throws an exception if the image could not be converted to ByteData.
Future<ByteData> _getByteDataRgba(Image image) async {
  final ByteData? byteDataRgba =
      await image.toByteData(format: ImageByteFormat.rawRgba);
  if (byteDataRgba == null) {
    throw Exception('Could not convert image to ByteData');
  }
  return byteDataRgba;
}

/// Encodes an [Image] object to a [Uint8List].
///
/// Note that the function is most efficient if the rawRgba [byteDataRgba] is provided (only if you already have it).
///
/// Also note that it is not in jpg format, but can still be used with `Image.memory()`.
Future<Uint8List> _encodeImage(Image image, {ByteData? byteDataRgba}) async {
  byteDataRgba ??= await _getByteDataRgba(image);
  final encodedImage = byteDataRgba.buffer.asUint8List();

  return encodedImage;
}

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
  final Image image = await _decodeImage(imageData);
  final ByteData imgByteData = await _getByteDataRgba(image);

  if (image.width == requiredWidth && image.height == requiredHeight) {
    return createInputMatrixFromImage(image, imgByteData, normalize: normalize);
  }

  final resizedImage = await _resizeImage(
    image,
    requiredWidth,
    requiredHeight,
    quality: quality,
  );

  final Num3DInputMatrix imageMatrix = createInputMatrixFromImage(
    resizedImage,
    imgByteData,
    normalize: normalize,
  );

  return imageMatrix;
}

/// Generates a face thumbnail from [imageData] and a [faceDetection].
Future<Uint8List> _generateFaceThumbnail(
  Uint8List imageData,
  FaceDetectionRelative faceDetection,
) async {
  final Image image = await _decodeImage(imageData);

  final Image faceThumbnail = await _cropImage(
    image,
    x: (faceDetection.xMinBox * image.width).round() - 20,
    y: (faceDetection.yMinBox * image.height).round() - 30,
    width: (faceDetection.width * image.width).round() + 40,
    height: (faceDetection.height * image.height).round() + 60,
  );

  return await _encodeImage(faceThumbnail);
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
Num3DInputMatrix createInputMatrixFromImage(
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
          normalize ? normalizePixel(pixel.red) : pixel.red,
          normalize ? normalizePixel(pixel.green) : pixel.green,
          normalize ? normalizePixel(pixel.blue) : pixel.blue,
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
Num3DInputMatrix createInputMatrixFromImageChannelsFirst(
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
  final pixelValue = normalize ? normalizePixel : (num value) => value;

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
double normalizePixel(num pixelValue) {
  return (pixelValue / 127.5) - 1;
}
