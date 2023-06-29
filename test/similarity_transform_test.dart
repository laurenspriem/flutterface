import 'dart:developer' as devtools show log;

import 'package:flutterface/services/face_alignment/similarity_transform.dart';
import 'package:ml_linalg/linalg.dart';
import 'package:test/test.dart';

// run `dart test test/similarity_transform_test.dart ` to test
void main() {
  final exampleLandmarks = [
    <int>[103, 114],
    <int>[147, 111],
    <int>[129, 142],
    <int>[128, 160],
  ];
  final expectedParameters = Matrix.fromList([
    [0.81073804, -0.05217403, -39.88931937],
    [0.05217403, 0.81073804, -46.62302376],
    [0, 0, 1]
  ]);

  final tform = SimilarityTransform();
  final isNoNanInParam = tform.estimate(exampleLandmarks);
  final parameters = tform.params;

  group('Similarity Transform Test', () {
    for (var i = 0; i < parameters.rowCount; i++) {
      for (var j = 0; j < parameters.columnCount; j++) {
        final actual = parameters[i][j];
        final expected = expectedParameters[i][j];
        devtools.log('actual: $actual, expected: $expected');
        test(
            'Test parameter estimation of SimilarityTransform at [$i, $j] in parameter matrix',
            () {
          expect(actual, closeTo(expected, 1e-4));
        });
      }
    }

    devtools.log('isNoNanInParam: $isNoNanInParam');
    test('isNoNanInParam test', () {
      expect(isNoNanInParam, isTrue);
    });

    // Let's clean the parameters and test again.
    tform.cleanParams();
    final parametersAfterClean = tform.params;
    test('cleanParams test', () {
      expect(
        parametersAfterClean,
        equals(
          Matrix.fromList([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0, 0, 1]
          ]),
        ),
      );
    });
    final secondIsNoNanInParam = tform.estimate(exampleLandmarks);
    final secondParameters = tform.params;

    for (var i = 0; i < secondParameters.rowCount; i++) {
      for (var j = 0; j < secondParameters.columnCount; j++) {
        final actual = secondParameters[i][j];
        final expected = expectedParameters[i][j];
        devtools.log('actual: $actual, expected: $expected');
        test(
            'Test parameter estimation AFTER cleaning of SimilarityTransform at [$i, $j] in parameter matrix',
            () {
          expect(actual, closeTo(expected, 1e-4));
        });
      }
    }
    devtools.log('isNoNanInParam AFTER cleaning: $secondIsNoNanInParam');
    test('isNoNanInParam test AFTER cleaning', () {
      expect(secondIsNoNanInParam, isTrue);
    });
  });
}
