class Detection {
  final double score;
  final List<double> box;
  final double xMinBox;
  final double yMinBox;
  final double xMaxBox;
  final double yMaxBox;
  final List<List<double>> allKeypoints;
  final List<double> leftEye;
  final List<double> rightEye;
  final List<double> nose;
  final List<double> mouth;
  final List<double> leftEar;
  final List<double> rightEar;
  final bool coordinatesAreAbsolute;

  Detection({
    required this.score,
    required this.box,
    required this.allKeypoints,
    required this.coordinatesAreAbsolute,
  })  : xMinBox = box[0],
        yMinBox = box[1],
        xMaxBox = box[2],
        yMaxBox = box[3],
        leftEye = allKeypoints[0],
        rightEye = allKeypoints[1],
        nose = allKeypoints[2],
        mouth = allKeypoints[3],
        leftEar = allKeypoints[4],
        rightEar = allKeypoints[5];

  factory Detection.zero() {
    return Detection(
      score: 0,
      box: [0, 0, 0, 0],
      allKeypoints: [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
      ],
      coordinatesAreAbsolute: true,
    );
  }

  @override
  String toString() {
    return 'Detection( with absolute coordinates: $coordinatesAreAbsolute \n score: $score \n Box: xMinBox: $xMinBox, yMinBox: $yMinBox, xMaxBox: $xMaxBox, yMaxBox: $yMaxBox, \n Keypoints: leftEye: $leftEye, rightEye: $rightEye, nose: $nose, mouth: $mouth, leftEar: $leftEar, rightEar: $rightEar \n )';
  }

  get width => xMaxBox - xMinBox;
  get height => yMaxBox - yMinBox;
}
