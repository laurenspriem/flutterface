// import 'dart:developer';

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show ByteData, rootBundle;
import 'package:image/image.dart' as img_lib;

Future<Image> drawFaces(imagePath, faceDetectionResult) async {
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
    );

    // Draw face landmarks as circles
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.leftEye[0],
      y: detection.leftEye[1],
      radius: 4,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.rightEye[0],
      y: detection.rightEye[1],
      radius: 4,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.nose[0],
      y: detection.nose[1],
      radius: 4,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.mouth[0],
      y: detection.mouth[1],
      radius: 4,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.leftEar[0],
      y: detection.leftEar[1],
      radius: 4,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
    drawnOnOriginalImage = img_lib.fillCircle(
      drawnOnOriginalImage,
      x: detection.rightEar[0],
      y: detection.rightEar[1],
      radius: 4,
      color: img_lib.ColorFloat16.rgb(255, 0, 0),
    );
  }
  final image = Image.memory(img_lib.encodeJpg(drawnOnOriginalImage!));

  return image;
}
