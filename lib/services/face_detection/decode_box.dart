import 'dart:math' as math;

import 'package:flutterface/services/face_detection/anchors.dart';
import 'package:flutterface/services/face_detection/options.dart';
import 'package:scidart/numdart.dart';

Array decodeBox(
  List<dynamic> rawBoxes,
  int i,
  List<Anchor> anchors,
  OptionsFace options,
) {
  final boxData = Array(List<double>.generate(options.numCoords, (i) => 0.0));

  var yCenter = rawBoxes[0][i][options.boxCoordOffset];
  var xCenter = rawBoxes[0][i][options.boxCoordOffset + 1];
  var h = rawBoxes[0][i][options.boxCoordOffset + 2];
  var w = rawBoxes[0][i][options.boxCoordOffset + 3];
  if (options.reverseOutputOrder) {
    xCenter = rawBoxes[0][i][options.boxCoordOffset];
    yCenter = rawBoxes[0][i][options.boxCoordOffset + 1];
    w = rawBoxes[0][i][options.boxCoordOffset + 2];
    h = rawBoxes[0][i][options.boxCoordOffset + 3];
  }

  xCenter = xCenter / options.xScale * anchors[i].w + anchors[i].xCenter;
  yCenter = yCenter / options.yScale * anchors[i].h + anchors[i].yCenter;

  if (options.applyExponentialOnBoxSize) {
    h = math.exp(h / options.hScale) * anchors[i].h;
    w = math.exp(w / options.wScale) * anchors[i].w;
  } else {
    h = h / options.hScale * anchors[i].h;
    w = w / options.wScale * anchors[i].w;
  }

  final yMin = yCenter - h / 2.0;
  final xMin = xCenter - w / 2.0;
  final yMax = yCenter + h / 2.0;
  final xMax = xCenter + w / 2.0;

  boxData[0] = yMin;
  boxData[1] = xMin;
  boxData[2] = yMax;
  boxData[3] = xMax;

  if (options.numKeypoints > 0) {
    for (var k = 0; k < options.numKeypoints; k++) {
      final offset =
          options.keypointCoordOffset + k * options.numValuesPerKeypoint;
      var keyPointY = rawBoxes[0][i][offset];
      var keyPointX = rawBoxes[0][i][offset + 1];

      if (options.reverseOutputOrder) {
        keyPointX = rawBoxes[0][i][offset];
        keyPointY = rawBoxes[0][i][offset + 1];
      }
      boxData[4 + k * options.numValuesPerKeypoint] =
          keyPointX / options.xScale * anchors[i].w + anchors[i].xCenter;

      boxData[4 + k * options.numValuesPerKeypoint + 1] =
          keyPointY / options.yScale * anchors[i].h + anchors[i].yCenter;
    }
  }
  return boxData;
}
