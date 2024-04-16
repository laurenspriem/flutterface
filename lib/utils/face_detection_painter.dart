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
      ..color = Colors.yellow;

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
        ..color = Colors.blue;

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

      // Draw score and direction of the face
      final TextPainter textPainter = TextPainter(
        text: TextSpan(
          text:
              // '${face.score.toStringAsFixed(3)} ${face.getFaceDirection().toDirectionString()}',
              face.getFaceDirection().toDirectionString(),
          style: const TextStyle(
            color: Colors.yellow,
            fontSize: 10.0,
            fontWeight: FontWeight.bold,
          ),
        ),
        textAlign: TextAlign.center,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      final double textHeight = textPainter.height;
      final double textX = face.xMinBox.toDouble() * scale;
      final double textY = face.yMinBox.toDouble() * scale - textHeight - 4.0;
      textPainter.paint(canvas, Offset(textX, textY));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}
