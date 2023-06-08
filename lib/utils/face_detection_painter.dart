import 'dart:developer' as devtools show log;
import 'dart:math' show max;

import 'package:flutter/material.dart';
import 'package:flutterface/utils/image.dart';
import 'package:image/image.dart' as image_lib;

Future<Image> drawFaces(imagePath, faceDetectionResult) async {
  final stopwatch = Stopwatch()..start();
  // Get image
  image_lib.Image drawnOnOriginalImage = await loadImageImage(imagePath);

  final originalImageWidth = drawnOnOriginalImage.width;
  final originalImageHeight = drawnOnOriginalImage.height;
  final imagesize = max(originalImageWidth, originalImageHeight);
  final drawRectThickness = imagesize ~/ 100;
  final drawCircleRadius = imagesize ~/ 100;

  // Draw faces
  for (final detection in faceDetectionResult) {
    // Draw bounding box as rectangle
    drawnOnOriginalImage = image_lib.drawRect(
      drawnOnOriginalImage,
      x1: detection.xMinBox,
      y1: detection.yMinBox,
      x2: detection.xMaxBox,
      y2: detection.yMaxBox,
      color: image_lib.ColorFloat16.rgb(0, 0, 255),
      thickness: drawRectThickness,
    );

    // Draw face landmarks as circles
    drawnOnOriginalImage = image_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.leftEye[0],
      y: detection.leftEye[1],
      radius: drawCircleRadius,
      color: image_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = image_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.rightEye[0],
      y: detection.rightEye[1],
      radius: drawCircleRadius,
      color: image_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = image_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.nose[0],
      y: detection.nose[1],
      radius: drawCircleRadius,
      color: image_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = image_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.mouth[0],
      y: detection.mouth[1],
      radius: drawCircleRadius,
      color: image_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = image_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.leftEar[0],
      y: detection.leftEar[1],
      radius: drawCircleRadius,
      color: image_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = image_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.rightEar[0],
      y: detection.rightEar[1],
      radius: drawCircleRadius,
      color: image_lib.ColorFloat16.rgb(255, 0, 0),
    );
  }
  final image = convertToFlutterImage(drawnOnOriginalImage);
  stopwatch.stop();
  devtools.log('drawFaces() executed in ${stopwatch.elapsedMilliseconds}ms');
  return image;
}
