import 'dart:typed_data' show Uint8List;

import 'package:image/image.dart' as image_lib;

image_lib.Image convertDataToImageImage(Uint8List imageData) {
  return image_lib.decodeImage(imageData)!;
}

Uint8List convertImageImageToData(image_lib.Image image) {
  return image_lib.encodeJpg(image);
}
