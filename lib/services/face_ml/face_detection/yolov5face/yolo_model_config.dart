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

final YOLOModelConfig yoloV5FaceS480x640tflite = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS480x640tflite,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.6,
    iouThreshold: 0.4,
    inputWidth: 640,
    inputHeight: 480,
  ),
);

final YOLOModelConfig yoloV5FaceS480x640onnx = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS480x640onnx,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.6,
    iouThreshold: 0.4,
    inputWidth: 640,
    inputHeight: 480,
  ),
);

final YOLOModelConfig yoloV5FaceS640x640onnx = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS640x640onnx,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.8,
    iouThreshold: 0.4,
    inputWidth: 640,
    inputHeight: 640,
  ),
);

final YOLOModelConfig yoloV5FaceS640x640DynamicBatchonnx = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceS640x640DynamicBatchonnx,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.75,
    iouThreshold: 0.4,
    inputWidth: 640,
    inputHeight: 640,
  ),
);

final YOLOModelConfig yoloV5FaceN640x640onnx = YOLOModelConfig(
  modelPath: ModelFile.yoloV5FaceN640x640onnx,
  faceOptions: FaceDetectionOptionsYOLO(
    minScoreSigmoidThreshold: 0.6,
    iouThreshold: 0.4,
    inputWidth: 640,
    inputHeight: 640,
  ),
);
