import 'package:flutter/material.dart';
import 'package:flutterface/services/face_detection/detection.dart';

class FacePainter extends CustomPainter {
  final List<FaceDetectionAbsolute> faceDetections;
  final Size imageSize;

  FacePainter({required this.faceDetections, required this.imageSize});

  @override
  void paint(Canvas canvas, Size size) {
    final Paint boundingBoxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0
      ..color = Colors.blue;

    double scaleY = 1;
    double scaleX = 1;
    if (imageSize.height > 400) {
      scaleY = 400 / imageSize.height;
      scaleX = scaleY;
    }

    for (var face in faceDetections) {
      // Draw bounding box
      canvas.drawRect(
        Rect.fromLTRB(
          face.xMinBox.toDouble() * scaleX,
          face.yMinBox.toDouble() * scaleY,
          face.xMaxBox.toDouble() * scaleX,
          face.yMaxBox.toDouble() * scaleY,
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
            keypoint[0].toDouble() * scaleX,
            keypoint[1].toDouble() * scaleY,
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
