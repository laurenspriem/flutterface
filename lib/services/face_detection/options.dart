import 'dart:math' as math show log;

class OptionsFace {
  final int numBoxes;
  final double minScoreSigmoidThreshold;
  final double iouThreshold;
  final int numCoords;
  final int numKeypoints;
  final int keypointCoordOffset;
  final int numValuesPerKeypoint;
  final int boxCoordOffset;
  final int maxNumFaces;
  final int inputWidth;
  final int inputHeight;
  final double scoreClippingThresh;
  final double inverseSigmoidMinScoreThreshold;
  final bool applyExponentialOnBoxSize;
  final bool useSigmoidScore;
  final bool flipVertically;

  OptionsFace({
    required this.numBoxes,
    required this.minScoreSigmoidThreshold,
    required this.iouThreshold,
    this.numCoords = 16,
    this.numKeypoints = 6,
    this.keypointCoordOffset = 4,
    this.numValuesPerKeypoint = 2,
    this.boxCoordOffset = 0,
    this.maxNumFaces = 100,
    this.inputWidth = 128,
    this.inputHeight = 128,
    this.scoreClippingThresh = 100.0,
    this.applyExponentialOnBoxSize = false,
    this.useSigmoidScore = true,
    this.flipVertically = false,
  }) : inverseSigmoidMinScoreThreshold =
            math.log(minScoreSigmoidThreshold / (1 - minScoreSigmoidThreshold));
}
