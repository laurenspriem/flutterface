// import 'dart:developer';

import 'dart:io';
import 'dart:math' show max;
import 'dart:developer' as devtools show log;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show ByteData, rootBundle;
import 'package:image/image.dart' as img_lib;

Future<Image> drawFaces(imagePath, faceDetectionResult) async {
  final stopwatch = Stopwatch()..start();
  // Get image
  img_lib.Image? drawnOnOriginalImage;
  if (imagePath.startsWith('assets/')) {
    // Load image as ByteData from asset bundle, then convert to image_lib.Image
    final ByteData imageData = await rootBundle.load(imagePath);
    drawnOnOriginalImage = img_lib.decodeImage(imageData.buffer.asUint8List());
  } else {
    // Read image bytes from file and convert to image_lib.Image
    final imageData = File(imagePath).readAsBytesSync();
    drawnOnOriginalImage = img_lib.decodeImage(imageData);
  }

  final originalImageWidth = drawnOnOriginalImage!.width;
  final originalImageHeight = drawnOnOriginalImage.height;
  final imagesize = max(originalImageWidth, originalImageHeight);
  final drawRectThickness = imagesize ~/ 100;
  final drawCircleRadius = imagesize ~/ 100;

  // Draw faces
  for (final detection in faceDetectionResult) {
    // Draw bounding box as rectangle
    drawnOnOriginalImage = img_lib.drawRect(
      drawnOnOriginalImage!,
      x1: detection.xMinBox,
      y1: detection.yMinBox,
      x2: detection.xMaxBox,
      y2: detection.yMaxBox,
      color: img_lib.ColorFloat16.rgb(0, 0, 255),
      thickness: drawRectThickness,
    );

    // Draw face landmarks as circles
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.leftEye[0],
      y: detection.leftEye[1],
      radius: drawCircleRadius,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.rightEye[0],
      y: detection.rightEye[1],
      radius: drawCircleRadius,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.nose[0],
      y: detection.nose[1],
      radius: drawCircleRadius,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.mouth[0],
      y: detection.mouth[1],
      radius: drawCircleRadius,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.leftEar[0],
      y: detection.leftEar[1],
      radius: drawCircleRadius,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.rightEar[0],
      y: detection.rightEar[1],
      radius: drawCircleRadius,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
  }
  final image = Image.memory(img_lib.encodeJpg(drawnOnOriginalImage!));
  stopwatch.stop();
  devtools.log('drawFaces() executed in ${stopwatch.elapsedMilliseconds}ms');
  return image;
}
