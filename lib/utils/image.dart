import 'dart:typed_data' show Uint8List;

import 'package:image/image.dart' as image_lib;

image_lib.Image convertUint8ListToImagePackageImage(Uint8List imageData) {
  return image_lib.decodeImage(imageData)!;
}

Uint8List convertImagePackageImageToUint8List(image_lib.Image image) {
  return image_lib.encodeJpg(image);
}
