import 'package:flutterface/services/face_detection/face_detection_service.dart';
import 'package:flutterface/services/model_inference_service.dart';
import 'package:get_it/get_it.dart';

final locator = GetIt.instance;

void setupLocator() {
  locator.registerSingleton<FaceDetection>(FaceDetection());
  // GetIt.I.registerSingleton<FaceAlignment>(FaceAlignment());

  locator.registerLazySingleton<ModelInferenceService>(
    () => ModelInferenceService(),
  );
}
