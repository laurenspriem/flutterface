import 'package:flutterface/models/model_file.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_face_detection_options.dart';

class YOLOModelConfig {
  final String modelPath;
  final FaceDetectionOptionsYOLO faceOptions;

  YOLOModelConfig({
    required this.modelPath,
    required this.faceOptions,
  });
}

final YOLOModelConfig yoloV5FaceN = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceN256x320,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.25,
    iouThreshold: 0.3,
    inputWidth: 320,
    inputHeight: 256,
  ),
);

final YOLOModelConfig yoloV5FaceS256x320 = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS256x320,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.25,
    iouThreshold: 0.3,
    inputWidth: 320,
    inputHeight: 256,
  ),
);

final YOLOModelConfig yoloV5FaceS480x640 = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS480x640,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.25,
    iouThreshold: 0.3,
    inputWidth: 640,
    inputHeight: 480,
  ),
);
