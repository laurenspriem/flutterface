import 'dart:math' show min;
import 'package:flutter/material.dart';
import 'package:flutterface/services/face_ml/face_detection/detection.dart';

class FacePainter extends CustomPainter {
  final List<FaceDetectionAbsolute> faceDetections;
  final Size imageSize;
  final Size availableSize;

  FacePainter({
    required this.faceDetections,
    required this.imageSize,
    required this.availableSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint boundingBoxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.blue;

    final scaleY = availableSize.height / imageSize.height;
    final scaleX = availableSize.width / imageSize.width;
    final scale = min(1, min(scaleX, scaleY));

    for (var face in faceDetections) {
      // Draw bounding box
      canvas.drawRect(
        Rect.fromLTRB(
          face.xMinBox.toDouble() * scale,
          face.yMinBox.toDouble() * scale,
          face.xMaxBox.toDouble() * scale,
          face.yMaxBox.toDouble() * scale,
        ),
        boundingBoxPaint,
      );

      // Draw keypoints
      final Paint keypointPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = Colors.red;

      final allKeypoints = face.allKeypoints;
      for (var keypoint in allKeypoints) {
        canvas.drawCircle(
          Offset(
            keypoint[0].toDouble() * scale,
            keypoint[1].toDouble() * scale,
          ),
          4.0,
          keypointPaint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
