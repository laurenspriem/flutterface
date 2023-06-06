import 'dart:developer' as devtools show log;
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutterface/services/face_detection/detection.dart';
import 'package:flutterface/services/face_detection/face_detection_service.dart';
import 'package:flutter/services.dart' show ByteData, rootBundle;
import 'package:image/image.dart' as img_lib;
import 'package:image_picker/image_picker.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.title});

  final String title;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker picker = ImagePicker();
  String? _imagePath;
  Image? _imageOriginal;
  Image? _imageDrawn;
  int _stockImageCounter = 0;
  final List<String> _stockImagePaths = [
    'assets/images/stock_images/one_person.jpeg',
    'assets/images/stock_images/one_person2.jpeg',
    'assets/images/stock_images/one_person3.jpeg',
    'assets/images/stock_images/one_person4.jpeg',
    'assets/images/stock_images/group_of_people.jpeg',
  ];

  bool _isAnalyzed = false;
  bool _isModelLoaded = false;
  bool _predicting = false;
  late FaceDetection _faceDetection;
  List<Detection> _faceDetectionResults = [];

  void _pickImage() async {
    cleanResult();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imagePath = image.path;
        _imageOriginal = Image.file(File(_imagePath!));
      });
    } else {
      devtools.log('No image selected');
    }
  }

  void _stockImage() async {
    cleanResult();
    setState(() {
      _imageOriginal = Image.asset(_stockImagePaths[_stockImageCounter]);
      _imagePath = _stockImagePaths[_stockImageCounter];
      _stockImageCounter = (_stockImageCounter + 1) % _stockImagePaths.length;
    });
  }

  void cleanResult() {
    _isAnalyzed = false;
    _faceDetectionResults = <Detection>[];
    _imageDrawn = null;
    setState(() {});
  }

  void analyzeImage() async {
    if (_imagePath == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select an image first'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }
    assert(_imageOriginal != null);
    if (_isAnalyzed || _predicting) {
      return;
    }

    setState(() {
      _predicting = true;
    });

    devtools.log('Image is sent to the model for inference');

    // 'Image plane data length: ${_imageWidget.planes[0].bytes.length}');
    if (!_isModelLoaded) {
      _faceDetection = await FaceDetection.create();
      _isModelLoaded = true;
    }

    final processedInputImage =
        await _faceDetection.getPreprocessedImage(_imagePath!);

    _faceDetectionResults = _faceDetection.predict(processedInputImage);

    devtools.log('Inference completed');
    devtools.log('Inference results: list $_faceDetectionResults of length '
        '${_faceDetectionResults.length}');

    img_lib.Image? drawnOnOriginalImage;
    if (_imagePath!.startsWith('assets/')) {
      // Load image as ByteData from asset bundle, then convert to image_lib.Image
      final ByteData imageData = await rootBundle.load(_imagePath!);
      drawnOnOriginalImage =
          img_lib.decodeImage(imageData.buffer.asUint8List());
    } else {
      // Read image bytes from file and convert to image_lib.Image
      final imageData = File(_imagePath!).readAsBytesSync();
      drawnOnOriginalImage = img_lib.decodeImage(imageData);
    }

    for (final detection in _faceDetectionResults) {
      final xMin = detection.xMinBox.toInt();
      final yMin = detection.yMinBox.toInt();
      final xMax = detection.xMaxBox.toInt();
      final yMax = detection.yMaxBox.toInt();
      final leftEyeX = detection.leftEye[0].toInt();
      final leftEyeY = detection.leftEye[1].toInt();
      final rightEyeX = detection.rightEye[0].toInt();
      final rightEyeY = detection.rightEye[1].toInt();
      final noseX = detection.nose[0].toInt();
      final noseY = detection.nose[1].toInt();
      final mouthX = detection.mouth[0].toInt();
      final mouthY = detection.mouth[1].toInt();
      final leftEarX = detection.leftEar[0].toInt();
      final leftEarY = detection.leftEar[1].toInt();
      final rightEarX = detection.rightEar[0].toInt();
      final rightEarY = detection.rightEar[1].toInt();
      devtools.log('Result: $detection');

      // Draw bounding box as rectangle
      drawnOnOriginalImage = img_lib.drawRect(
        drawnOnOriginalImage!,
        x1: xMin,
        y1: yMin,
        x2: xMax,
        y2: yMax,
        color: img_lib.ColorFloat16.rgb(0, 0, 255),
      );

      // Draw face landmarks as circles
      drawnOnOriginalImage = img_lib.fillCircle(
        drawnOnOriginalImage,
        x: leftEyeX,
        y: leftEyeY,
        radius: 4,
        color: img_lib.ColorFloat16.rgb(255, 0, 0),
      );
      drawnOnOriginalImage = img_lib.fillCircle(
        drawnOnOriginalImage,
        x: rightEyeX,
        y: rightEyeY,
        radius: 4,
        color: img_lib.ColorFloat16.rgb(255, 0, 0),
      );
      drawnOnOriginalImage = img_lib.fillCircle(
        drawnOnOriginalImage,
        x: noseX,
        y: noseY,
        radius: 4,
        color: img_lib.ColorFloat16.rgb(255, 0, 0),
      );
      drawnOnOriginalImage = img_lib.fillCircle(
        drawnOnOriginalImage,
        x: mouthX,
        y: mouthY,
        radius: 4,
        color: img_lib.ColorFloat16.rgb(255, 0, 0),
      );
      drawnOnOriginalImage = img_lib.fillCircle(
        drawnOnOriginalImage,
        x: leftEarX,
        y: leftEarY,
        radius: 4,
        color: img_lib.ColorFloat16.rgb(255, 0, 0),
      );
      drawnOnOriginalImage = img_lib.fillCircle(
        drawnOnOriginalImage,
        x: rightEarX,
        y: rightEarY,
        radius: 4,
        color: img_lib.ColorFloat16.rgb(255, 0, 0),
      );
    }

    setState(() {
      _imageDrawn = Image.memory(img_lib.encodeJpg(drawnOnOriginalImage!));
      _predicting = false;
      _isAnalyzed = true;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            if (_imageDrawn != null && _isAnalyzed)
              SizedBox(
                height: 400,
                child: _imageDrawn,
              )
            else if (_imageOriginal !=
                null) // Show original image when not analyzed
              SizedBox(
                height: 400,
                child: _imageOriginal,
              )
            else
              const Text('No image selected'),
            const SizedBox(height: 16),
            SizedBox(
              width: 150,
              child: TextButton(
                onPressed: _pickImage,
                style: TextButton.styleFrom(
                  foregroundColor:
                      Theme.of(context).colorScheme.onPrimaryContainer,
                  backgroundColor:
                      Theme.of(context).colorScheme.primaryContainer,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: const Text('Pick image'),
              ),
            ),
            SizedBox(
              width: 150,
              child: TextButton(
                onPressed: _stockImage,
                style: TextButton.styleFrom(
                  foregroundColor:
                      Theme.of(context).colorScheme.onPrimaryContainer,
                  backgroundColor:
                      Theme.of(context).colorScheme.primaryContainer,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: const Text('Stock image'),
              ),
            ),
            SizedBox(
              width: 150,
              child: TextButton(
                onPressed: analyzeImage,
                style: TextButton.styleFrom(
                  foregroundColor:
                      Theme.of(context).colorScheme.onPrimaryContainer,
                  backgroundColor:
                      Theme.of(context).colorScheme.primaryContainer,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: const Text('Detect faces'),
              ),
            ),
            SizedBox(
              width: 150,
              child: TextButton(
                onPressed: cleanResult,
                style: TextButton.styleFrom(
                  foregroundColor:
                      Theme.of(context).colorScheme.onPrimaryContainer,
                  backgroundColor:
                      Theme.of(context).colorScheme.primaryContainer,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: const Text('Clean result'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
