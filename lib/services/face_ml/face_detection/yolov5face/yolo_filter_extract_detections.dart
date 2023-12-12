import 'dart:math' as math show exp, pow, max, min;
// import 'dart:developer' show log;

import 'package:flutterface/services/face_ml/face_detection/detection.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_face_detection_options.dart';

const _anchors = [
  [
    [4, 5],
    [8, 10],
    [13, 16],
  ], // P3/8
  [
    [23, 29],
    [43, 55],
    [73, 105],
  ], // P4/16
  [
    [146, 217],
    [231, 300],
    [335, 433],
  ], // P5/32
];
const _layers = <int>[0, 1, 2];
const _strides = <int>[8, 16, 32];

// output (stride 32,16,8): [8, 10, 16, 3], [16, 20, 16, 3], [32, 40, 16, 3]

/// Use this function only for comparing inference speed between tflite and onnx
List<FaceDetectionRelative> filterExtractDetectionsYOLOtfliteDebug({
  required FaceDetectionOptionsYOLO options,
  required List<dynamic> stride32, // Nested List of shape [8, 10, 16, 3]
  required List<dynamic> stride16, // Nested List of shape [16, 20, 16, 3]
  required List<dynamic> stride8, // Nested List of shape [32, 40, 16, 3]
}) {
  final allStrides = [stride8, stride16, stride32];
  final outputDetections = <FaceDetectionRelative>[];
  final output = <List<double>>[];
  // final rawEverythingDebug = List.generate(16, (_) => <double>[]);
  // final rawScoresDebug = <double>[];

  // Go through the raw output and process the anchors: https://github.com/deepcam-cn/yolov5-face/blob/eb23d18defe4a76cc06449a61cd51004c59d2697/torch2tensorrt/main.py#L94
  for (final i in _layers) {
    final stride = allStrides[i];
    final anchors = _anchors[i];

    int h = 0;
    int w = 0;
    for (final heightLayer in stride) {
      for (final widthLayer in heightLayer) {
        for (var j = 0; j < 3; j++) {
          // rawEverythingDebug[0].add(widthLayer[0][j]);
          // rawEverythingDebug[1].add(widthLayer[1][j]);
          // rawEverythingDebug[2].add(widthLayer[2][j]);
          // rawEverythingDebug[3].add(widthLayer[3][j]);
          // rawEverythingDebug[4].add(widthLayer[4][j]);
          // rawEverythingDebug[5].add(widthLayer[5][j]);
          // rawEverythingDebug[6].add(widthLayer[6][j]);
          // rawEverythingDebug[7].add(widthLayer[7][j]);
          // rawEverythingDebug[8].add(widthLayer[8][j]);
          // rawEverythingDebug[9].add(widthLayer[9][j]);
          // rawEverythingDebug[10].add(widthLayer[10][j]);
          // rawEverythingDebug[11].add(widthLayer[11][j]);
          // rawEverythingDebug[12].add(widthLayer[12][j]);
          // rawEverythingDebug[13].add(widthLayer[13][j]);
          // rawEverythingDebug[14].add(widthLayer[14][j]);
          // rawEverythingDebug[15].add(widthLayer[15][j]);
          // rawScoresDebug.add(widthLayer[4][j]);

          // Filter out raw detections with low scores
          if (widthLayer[4][j] < options.inverseSigmoidMinScoreThreshold) {
            continue;
          }

          // Get the raw detection
          final rawDetection = <double>[
            widthLayer[0][j],
            widthLayer[1][j],
            widthLayer[2][j],
            widthLayer[3][j],
            widthLayer[4][j],
            widthLayer[5][j],
            widthLayer[6][j],
            widthLayer[7][j],
            widthLayer[8][j],
            widthLayer[9][j],
            widthLayer[10][j],
            widthLayer[11][j],
            widthLayer[12][j],
            widthLayer[13][j],
            widthLayer[14][j],
            widthLayer[15][j],
          ];

          // Process the raw detection (still don't understand why we use sigmoid on the first 4 values (bbox) and the last value (cls) ) https://github.com/deepcam-cn/yolov5-face/issues/93#issuecomment-1064276217
          final classRange = [0, 1, 2, 3, 4, 15];
          for (final c in classRange) {
            rawDetection[c] = _sigmoid(rawDetection[c]); // sigmoid score
          }
          // bounding box
          final grid = [w, h];
          rawDetection[0] = (rawDetection[0] * 2 - 0.5 + grid[0]) * _strides[i];
          rawDetection[1] = (rawDetection[1] * 2 - 0.5 + grid[1]) * _strides[i];
          rawDetection[2] =
              math.pow((rawDetection[2] * 2), 2) * anchors[j][0] as double;
          rawDetection[3] =
              math.pow((rawDetection[3] * 2), 2) * anchors[j][1] as double;
          // keypoints
          rawDetection[5] =
              rawDetection[5] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[6] =
              rawDetection[6] * anchors[j][1] + grid[1] * _strides[i];
          rawDetection[7] =
              rawDetection[7] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[8] =
              rawDetection[8] * anchors[j][1] + grid[1] * _strides[i];
          rawDetection[9] =
              rawDetection[9] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[10] =
              rawDetection[10] * anchors[j][1] + grid[1] * _strides[i];
          rawDetection[11] =
              rawDetection[11] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[12] =
              rawDetection[12] * anchors[j][1] + grid[1] * _strides[i];
          rawDetection[13] =
              rawDetection[13] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[14] =
              rawDetection[14] * anchors[j][1] + grid[1] * _strides[i];

          // Append the processed raw detection to the output
          output.add(rawDetection);
        }
        w++;
      }
      h++;
    }
  }

  // final rawEverythingMinMax = rawEverythingDebug
  //     .map((e) => [e.reduce(math.min), e.reduce(math.max)])
  //     .toList();
  // final rawScoresMinMax = [
  //   rawScoresDebug.reduce(math.min),
  //   rawScoresDebug.reduce(math.max),
  // ];

  for (final List<double> rawDetection in output) {
    // Get absolute bounding box coordinates in format [xMin, yMin, xMax, yMax] https://github.com/deepcam-cn/yolov5-face/blob/eb23d18defe4a76cc06449a61cd51004c59d2697/utils/general.py#L216
    final xMinAbs = rawDetection[0] - rawDetection[2] / 2;
    final yMinAbs = rawDetection[1] - rawDetection[3] / 2;
    final xMaxAbs = rawDetection[0] + rawDetection[2] / 2;
    final yMaxAbs = rawDetection[1] + rawDetection[3] / 2;

    // Get the relative bounding box coordinates in format [xMin, yMin, xMax, yMax]
    final box = [
      xMinAbs / options.inputWidth,
      yMinAbs / options.inputHeight,
      xMaxAbs / options.inputWidth,
      yMaxAbs / options.inputHeight,
    ];

    // Get the keypoints coordinates in format [x, y]
    final allKeypoints = <List<double>>[
      [
        rawDetection[5] / options.inputWidth,
        rawDetection[6] / options.inputHeight,
      ],
      [
        rawDetection[7] / options.inputWidth,
        rawDetection[8] / options.inputHeight,
      ],
      [
        rawDetection[9] / options.inputWidth,
        rawDetection[10] / options.inputHeight,
      ],
      [
        rawDetection[11] / options.inputWidth,
        rawDetection[12] / options.inputHeight,
      ],
      [
        rawDetection[13] / options.inputWidth,
        rawDetection[14] / options.inputHeight,
      ],
    ];

    // Get the score
    final score =
        rawDetection[4]; // Or should it be rawDetection[4]*rawDetection[15]?

    // Create the relative detection
    final detection = FaceDetectionRelative(
      score: score,
      box: box,
      allKeypoints: allKeypoints,
    );

    // Append the relative detection to the output
    outputDetections.add(detection);
  }

  return outputDetections;
}

/// Use this function only for comparing inference speed between tflite and onnx
List<FaceDetectionRelative> filterExtractDetectionsYOLOonnxDebug({
  required FaceDetectionOptionsYOLO options,
  required List<dynamic> stride32, // Nested List of shape [3, 15, 20, 16]
  required List<dynamic> stride16, // Nested List of shape [3, 30, 40, 16]
  required List<dynamic> stride8, // Nested List of shape [3, 60, 80, 16]
}) {
  final allStrides = [stride8, stride16, stride32];
  final outputDetections = <FaceDetectionRelative>[];
  final output = <List<double>>[];
  final rawEverythingDebug = List.generate(16, (_) => <double>[]);
  final rawScoresDebug = <double>[];

  // Go through the raw output and process the anchors: https://github.com/deepcam-cn/yolov5-face/blob/eb23d18defe4a76cc06449a61cd51004c59d2697/torch2tensorrt/main.py#L94
  for (final i in _layers) {
    final stride = allStrides[i];
    final anchors = _anchors[i];

    int h = 0;
    int w = 0;
    for (var j = 0; j < 3; j++) {
      final classification = stride[j];
      for (final heightLayer in classification) {
        for (final widthLayer in heightLayer) {
          rawEverythingDebug[0].add(widthLayer[0]);
          rawEverythingDebug[1].add(widthLayer[1]);
          rawEverythingDebug[2].add(widthLayer[2]);
          rawEverythingDebug[3].add(widthLayer[3]);
          rawEverythingDebug[4].add(widthLayer[4]);
          rawEverythingDebug[5].add(widthLayer[5]);
          rawEverythingDebug[6].add(widthLayer[6]);
          rawEverythingDebug[7].add(widthLayer[7]);
          rawEverythingDebug[8].add(widthLayer[8]);
          rawEverythingDebug[9].add(widthLayer[9]);
          rawEverythingDebug[10].add(widthLayer[10]);
          rawEverythingDebug[11].add(widthLayer[11]);
          rawEverythingDebug[12].add(widthLayer[12]);
          rawEverythingDebug[13].add(widthLayer[13]);
          rawEverythingDebug[14].add(widthLayer[14]);
          rawEverythingDebug[15].add(widthLayer[15]);
          rawScoresDebug.add(widthLayer[4]);

          // Filter out raw detections with low scores
          if (widthLayer[4] < options.inverseSigmoidMinScoreThreshold) {
            continue;
          }

          // Get the raw detection
          final rawDetection = List<double>.from(widthLayer);

          // Process the raw detection (still don't understand why we use sigmoid on the first 4 values (bbox) and the last value (cls) ) https://github.com/deepcam-cn/yolov5-face/issues/93#issuecomment-1064276217
          final classRange = [0, 1, 2, 3, 4, 15];
          for (final c in classRange) {
            rawDetection[c] = _sigmoid(rawDetection[c]); // sigmoid score
          }
          // bounding box
          final grid = [w, h];
          rawDetection[0] = (rawDetection[0] * 2 - 0.5 + grid[0]) * _strides[i];
          rawDetection[1] = (rawDetection[1] * 2 - 0.5 + grid[1]) * _strides[i];
          rawDetection[2] =
              math.pow((rawDetection[2] * 2), 2) * anchors[j][0] as double;
          rawDetection[3] =
              math.pow((rawDetection[3] * 2), 2) * anchors[j][1] as double;
          // keypoints
          rawDetection[5] =
              rawDetection[5] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[6] =
              rawDetection[6] * anchors[j][1] + grid[1] * _strides[i];
          rawDetection[7] =
              rawDetection[7] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[8] =
              rawDetection[8] * anchors[j][1] + grid[1] * _strides[i];
          rawDetection[9] =
              rawDetection[9] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[10] =
              rawDetection[10] * anchors[j][1] + grid[1] * _strides[i];
          rawDetection[11] =
              rawDetection[11] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[12] =
              rawDetection[12] * anchors[j][1] + grid[1] * _strides[i];
          rawDetection[13] =
              rawDetection[13] * anchors[j][0] + grid[0] * _strides[i];
          rawDetection[14] =
              rawDetection[14] * anchors[j][1] + grid[1] * _strides[i];

          // Append the processed raw detection to the output
          output.add(rawDetection);
        }
        w++;
      }
      h++;
    }
  }

  final rawEverythingMinMax = rawEverythingDebug
      .map((e) => [e.reduce(math.min), e.reduce(math.max)])
      .toList();
  final rawScoresMinMax = [
    rawScoresDebug.reduce(math.min),
    rawScoresDebug.reduce(math.max),
  ];

  for (final List<double> rawDetection in output) {
    // Get absolute bounding box coordinates in format [xMin, yMin, xMax, yMax] https://github.com/deepcam-cn/yolov5-face/blob/eb23d18defe4a76cc06449a61cd51004c59d2697/utils/general.py#L216
    final xMinAbs = rawDetection[0] - rawDetection[2] / 2;
    final yMinAbs = rawDetection[1] - rawDetection[3] / 2;
    final xMaxAbs = rawDetection[0] + rawDetection[2] / 2;
    final yMaxAbs = rawDetection[1] + rawDetection[3] / 2;

    // Get the relative bounding box coordinates in format [xMin, yMin, xMax, yMax]
    final box = [
      xMinAbs / options.inputWidth,
      yMinAbs / options.inputHeight,
      xMaxAbs / options.inputWidth,
      yMaxAbs / options.inputHeight,
    ];

    // Get the keypoints coordinates in format [x, y]
    final allKeypoints = <List<double>>[
      [
        rawDetection[5] / options.inputWidth,
        rawDetection[6] / options.inputHeight,
      ],
      [
        rawDetection[7] / options.inputWidth,
        rawDetection[8] / options.inputHeight,
      ],
      [
        rawDetection[9] / options.inputWidth,
        rawDetection[10] / options.inputHeight,
      ],
      [
        rawDetection[11] / options.inputWidth,
        rawDetection[12] / options.inputHeight,
      ],
      [
        rawDetection[13] / options.inputWidth,
        rawDetection[14] / options.inputHeight,
      ],
    ];

    // Get the score
    final score =
        rawDetection[4]; // Or should it be rawDetection[4]*rawDetection[15]?

    // Create the relative detection
    final detection = FaceDetectionRelative(
      score: score,
      box: box,
      allKeypoints: allKeypoints,
    );

    // Append the relative detection to the output
    outputDetections.add(detection);
  }

  return outputDetections;
}

List<FaceDetectionRelative> yoloOnnxFilterExtractDetections({
  required FaceDetectionOptionsYOLO options,
  required List<List<double>> results, // // [25200, 16]
}) {
  final outputDetections = <FaceDetectionRelative>[];
  final output = <List<double>>[];

  // Go through the raw output and check the scores
  for (final result in results) {
    // Filter out raw detections with low scores
    if (result[4] < options.minScoreSigmoidThreshold) {
      continue;
    }

    // Get the raw detection
    final rawDetection = List<double>.from(result);

    // Append the processed raw detection to the output
    output.add(rawDetection);
  }

  for (final List<double> rawDetection in output) {
    // Get absolute bounding box coordinates in format [xMin, yMin, xMax, yMax] https://github.com/deepcam-cn/yolov5-face/blob/eb23d18defe4a76cc06449a61cd51004c59d2697/utils/general.py#L216
    final xMinAbs = rawDetection[0] - rawDetection[2] / 2;
    final yMinAbs = rawDetection[1] - rawDetection[3] / 2;
    final xMaxAbs = rawDetection[0] + rawDetection[2] / 2;
    final yMaxAbs = rawDetection[1] + rawDetection[3] / 2;

    // Get the relative bounding box coordinates in format [xMin, yMin, xMax, yMax]
    final box = [
      xMinAbs / options.inputWidth,
      yMinAbs / options.inputHeight,
      xMaxAbs / options.inputWidth,
      yMaxAbs / options.inputHeight,
    ];

    // Get the keypoints coordinates in format [x, y]
    final allKeypoints = <List<double>>[
      [
        rawDetection[5] / options.inputWidth,
        rawDetection[6] / options.inputHeight,
      ],
      [
        rawDetection[7] / options.inputWidth,
        rawDetection[8] / options.inputHeight,
      ],
      [
        rawDetection[9] / options.inputWidth,
        rawDetection[10] / options.inputHeight,
      ],
      [
        rawDetection[11] / options.inputWidth,
        rawDetection[12] / options.inputHeight,
      ],
      [
        rawDetection[13] / options.inputWidth,
        rawDetection[14] / options.inputHeight,
      ],
    ];

    // Get the score
    final score =
        rawDetection[4]; // Or should it be rawDetection[4]*rawDetection[15]?

    // Create the relative detection
    final detection = FaceDetectionRelative(
      score: score,
      box: box,
      allKeypoints: allKeypoints,
    );

    // Append the relative detection to the output
    outputDetections.add(detection);
  }

  return outputDetections;
}

double _sigmoid(double x) {
  return 1 / (1 + math.exp(-x));
}
