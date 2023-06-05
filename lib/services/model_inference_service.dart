import 'dart:developer' as devtools show log;

import 'package:flutterface/enums/models.dart';
import 'package:flutterface/services/ai_model.dart';
// import 'package:flutterface/services/face_detection/face_detection_service.dart';
// import 'package:flutterface/services/service_locator.dart';

class ModelInferenceService {
  late AIModel model;
  late Function handler;
  Map<String, dynamic>? inferenceResults;

  Future<Map<String, dynamic>?> inference({
    required String imagePath,
  }) async {
    devtools.log('Message sent to isolate for inference');
    try {
      inferenceResults = await handler(
        imagePath: imagePath,
        detectorAddress: model.getAddress,
      );
      return inferenceResults;
    } catch (e) {
      devtools.log('Error during inference: $e');
      return null;
    }
  }

  void setModelConfig(Models modelEnum) {
    devtools.log('setModelConfig: $modelEnum');
    switch (modelEnum) {
      case Models.faceDetection:
        // model = locator<FaceDetection>();
        // handler = runFaceDetector;
        break;
      case Models.faceAlignment:
        // model = locator<FaceAlignment>();
        // handler = runFaceAlignment;
        break;
      case Models.faceEmbedding:
        // model = locator<FaceEmbedding>();
        // handler = runFaceEmbedding;
        break;
    }
  }
}
