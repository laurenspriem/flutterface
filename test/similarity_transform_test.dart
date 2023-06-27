import 'dart:developer' as devtools show log;

import 'package:flutter_test/flutter_test.dart';
import 'package:flutterface/services/face_alignment/similarity_transform_ml_linalg.dart';
import 'package:ml_linalg/linalg.dart';

void main() {
  final lmk = [
    <double>[103, 114],
    <double>[147, 111],
    <double>[129, 142],
    <double>[128, 160],
  ];
  final dst = [
    <double>[38.2946, 51.6963],
    <double>[73.5318, 51.5014],
    <double>[56.0252, 71.7366],
    <double>[56.1396, 92.2848],
  ];
  final tform = SimilarityTransform();
  final isNoNanInParam = tform.estimate(lmk, dst);
  final parameters = tform.params;
  final expectedParameters = Matrix.fromList([
    [0.81073804, -0.05217403, -39.88931937],
    [0.05217403, 0.81073804, -46.62302376],
    [0, 0, 1]]);
  for (var i = 0; i < parameters.rowCount; i++) {
    for (var j = 0; j < parameters.columnCount; j++) {
      final actual = parameters[i][j];
      final expected = expectedParameters[i][j];
      devtools.log('actual: $actual, expected: $expected');
      // test('Test parameter estimation of SimilarityTransform', () {
      //   expect(actual, closeTo(expected, 1e-3));
      // });
    }
  }

  devtools.log('isNoNanInParam: $isNoNanInParam');
}
