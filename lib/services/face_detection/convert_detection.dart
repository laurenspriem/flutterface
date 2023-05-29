import 'package:flutterface/services/face_detection/anchors.dart';
import 'package:flutterface/services/face_detection/decode_box.dart';
import 'package:flutterface/services/face_detection/detection.dart';
import 'package:flutterface/services/face_detection/options.dart';

List<Detection> convertToDetections(
  List<double> rawBoxes,
  List<Anchor> anchors,
  List<double> detectionScores,
  List<int> detectionClasses,
  OptionsFace options,
) {
  final outputDetections = <Detection>[];
  for (var i = 0; i < options.numBoxes; i++) {
    if (detectionScores[i] < options.minScoreThresh) continue;
    const boxOffset = 0;
    final boxData = decodeBox(rawBoxes, i, anchors, options);

    final detection = convertToDetection(
      boxData[boxOffset + 0],
      boxData[boxOffset + 1],
      boxData[boxOffset + 2],
      boxData[boxOffset + 3],
      detectionScores[i],
      detectionClasses[i],
      options.flipVertically,
    );
    outputDetections.add(detection);
  }
  return outputDetections;
}

Detection convertToDetection(
  double boxYMin,
  double boxXMin,
  double boxYMax,
  double boxXMax,
  double score,
  int classID,
  bool flipVertically,
) {
  final yMin = flipVertically ? 1.0 - boxYMax : boxYMin;
  final width = boxXMax;
  final height = boxYMax;

  return Detection(
    score,
    classID,
    boxXMin,
    yMin,
    width,
    height,
  );
}
