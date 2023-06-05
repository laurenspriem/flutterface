// import 'dart:developer';

import 'package:flutter/material.dart';

class FaceDetectionPainter extends CustomPainter {
  final Rect bbox;
  final double ratio;

  FaceDetectionPainter({
    required this.bbox,
    required this.ratio,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (bbox != Rect.zero) {
      final paint = Paint()
        ..color = Colors.blue
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;
      final bboxResized = Rect.fromLTRB(
        bbox.left * ratio,
        bbox.top * ratio,
        bbox.right * ratio,
        bbox.bottom * ratio,
      );
      canvas.drawRect(bboxResized, paint);
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
