import 'package:flutter/material.dart';

class ResponseSnackBar extends SnackBar {
  final String message;
  final bool isError;
  // final Duration duration;

  ResponseSnackBar({
    super.key,
    required this.message,
    required this.isError,
    super.duration = const Duration(seconds: 5),
  }) : super(
          content: Text(message),
          backgroundColor: isError ? Colors.red : Colors.green,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
          elevation: 20.0,
          behavior: SnackBarBehavior.floating,
        );
}

void showSnackbar(
  BuildContext context,
  String message, {
  int durationInSeconds = 2,
}) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Text(message),
      duration: Duration(seconds: durationInSeconds),
    ),
  );
}

void showResponseSnackbar(
  BuildContext context,
  String message, {
  bool isError = true,
  int durationInSeconds = 3,
}) {
  ScaffoldMessenger.of(context).showSnackBar(
    ResponseSnackBar(
      message: message,
      isError: isError,
      duration: Duration(seconds: durationInSeconds),
    ),
  );
}
