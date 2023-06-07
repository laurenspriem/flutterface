import 'package:flutterface/constants/model_file.dart';
import 'package:flutterface/services/face_detection/anchors.dart';
import 'package:flutterface/services/face_detection/face_options.dart';

class ModelConfig {
  final String modelPath;
  final FaceOptions faceOptions;
  final AnchorOptions anchorOptions;

  ModelConfig({
    required this.modelPath,
    required this.faceOptions,
    required this.anchorOptions,
  });
}

final ModelConfig faceDetectionShortRange = ModelConfig(
  modelPath: ModelFile.faceDetectionShortRange,
  faceOptions: FaceOptions(
    numBoxes: 896,
    minScoreSigmoidThreshold: 0.60,
    iouThreshold: 0.3,
    inputWidth: 128,
    inputHeight: 128,
  ),
  anchorOptions: AnchorOptions(
    inputSizeHeight: 128,
    inputSizeWidth: 128,
    minScale: 0.1484375,
    maxScale: 0.75,
    anchorOffsetX: 0.5,
    anchorOffsetY: 0.5,
    numLayers: 4,
    featureMapHeight: [],
    featureMapWidth: [],
    strides: [8, 16, 16, 16],
    aspectRatios: [1.0],
    reduceBoxesInLowestLayer: false,
    interpolatedScaleAspectRatio: 1.0,
    fixedAnchorSize: true,
  ),
);

final ModelConfig faceDetectionFullRangeSparse = ModelConfig(
  modelPath: ModelFile.faceDetectionFullRangeSparse,
  faceOptions: FaceOptions(
    numBoxes: 2304,
    minScoreSigmoidThreshold: 0.60,
    iouThreshold: 0.3,
    inputWidth: 192,
    inputHeight: 192,
  ),
  anchorOptions: AnchorOptions(
    inputSizeHeight: 192,
    inputSizeWidth: 192,
    minScale: 0.1484375,
    maxScale: 0.75,
    anchorOffsetX: 0.5,
    anchorOffsetY: 0.5,
    numLayers: 1,
    featureMapHeight: [],
    featureMapWidth: [],
    strides: [4],
    aspectRatios: [1.0],
    reduceBoxesInLowestLayer: false,
    interpolatedScaleAspectRatio: 0.0,
    fixedAnchorSize: true,
  ),
);

final ModelConfig faceDetectionFullRangeDense = ModelConfig(
  modelPath: ModelFile.faceDetectionFullRangeDense,
  faceOptions: FaceOptions(
    numBoxes: 2304,
    minScoreSigmoidThreshold: 0.60,
    iouThreshold: 0.3,
    inputWidth: 192,
    inputHeight: 192,
  ),
  anchorOptions: AnchorOptions(
    inputSizeHeight: 192,
    inputSizeWidth: 192,
    minScale: 0.1484375,
    maxScale: 0.75,
    anchorOffsetX: 0.5,
    anchorOffsetY: 0.5,
    numLayers: 1,
    featureMapHeight: [],
    featureMapWidth: [],
    strides: [4],
    aspectRatios: [1.0],
    reduceBoxesInLowestLayer: false,
    interpolatedScaleAspectRatio: 0.0,
    fixedAnchorSize: true,
  ),
);