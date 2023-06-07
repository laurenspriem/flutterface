import 'dart:math' as math show log;

class FaceOptions {
  final int numBoxes;
  final double minScoreSigmoidThreshold;
  final double iouThreshold;
  final int inputWidth;
  final int inputHeight;
  final int numCoords;
  final int numKeypoints;
  final int keypointCoordOffset;
  final int numValuesPerKeypoint;
  final int boxCoordOffset;
  final int maxNumFaces;
  final double scoreClippingThresh;
  final double inverseSigmoidMinScoreThreshold;
  final bool applyExponentialOnBoxSize;
  final bool useSigmoidScore;
  final bool flipVertically;

  FaceOptions({
    required this.numBoxes,
    required this.minScoreSigmoidThreshold,
    required this.iouThreshold,
    required this.inputWidth,
    required this.inputHeight,
    this.numCoords = 16,
    this.numKeypoints = 6,
    this.keypointCoordOffset = 4,
    this.numValuesPerKeypoint = 2,
    this.boxCoordOffset = 0,
    this.maxNumFaces = 100,
    this.scoreClippingThresh = 100.0,
    this.applyExponentialOnBoxSize = false,
    this.useSigmoidScore = true,
    this.flipVertically = false,
  }) : inverseSigmoidMinScoreThreshold =
            math.log(minScoreSigmoidThreshold / (1 - minScoreSigmoidThreshold));
}
