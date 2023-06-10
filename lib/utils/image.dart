import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show ByteData, rootBundle;
import 'package:image/image.dart' as image_lib;

Future<image_lib.Image> loadImageImage(String imagePath) async {
  assert(imagePath.isNotEmpty);

  image_lib.Image? image;
  if (imagePath.startsWith('assets/')) {
    // Load image as ByteData from asset bundle, then convert to image_lib.Image
    final ByteData imageData = await rootBundle.load(imagePath);
    image = image_lib.decodeImage(imageData.buffer.asUint8List());
  } else {
    // Read image bytes from file and convert to image_lib.Image
    final imageData = File(imagePath).readAsBytesSync();
    image = image_lib.decodeImage(imageData);
  }

  if (image == null) {
    throw Exception('Image not found');
  }

  return image;
}

Image convertToFlutterImage(image_lib.Image image) {
  return Image.memory(image_lib.encodeJpg(image));
}
