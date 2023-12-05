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

final YOLOModelConfig yoloV5FaceNtflite = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceN256x320tflite,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.25,
    iouThreshold: 0.3,
    inputWidth: 320,
    inputHeight: 256,
  ),
);

final YOLOModelConfig yoloV5FaceS256x320tflite = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS256x320tflite,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.25,
    iouThreshold: 0.3,
    inputWidth: 320,
    inputHeight: 256,
  ),
);

final YOLOModelConfig yoloV5FaceS480x640tflite = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS480x640tflite,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.25,
    iouThreshold: 0.3,
    inputWidth: 640,
    inputHeight: 480,
  ),
);

final YOLOModelConfig yoloV5FaceS480x640onnx = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS480x640onnx,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.25,
    iouThreshold: 0.3,
    inputWidth: 640,
    inputHeight: 480,
  ),
);
