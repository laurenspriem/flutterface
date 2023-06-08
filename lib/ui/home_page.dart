import 'dart:developer' as devtools show log;
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutterface/services/face_detection/detection.dart';
import 'package:flutterface/services/face_detection/face_detection_service.dart';
import 'package:flutterface/utils/face_detection_painter.dart';
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
  Size _imageSize = const Size(0, 0);
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
  List<FaceDetectionAbsolute> _faceDetectionResults = [];

  void _pickImage() async {
    cleanResult();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      final decodedImage = await decodeImageFromList(await image.readAsBytes());
      setState(() {
        _imagePath = image.path;
        _imageOriginal = Image.file(File(_imagePath!));
        _imageSize =
            Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
      });
    } else {
      devtools.log('No image selected');
    }
  }

  void _stockImage() async {
    cleanResult();
    final byteData =
        await rootBundle.load(_stockImagePaths[_stockImageCounter]);
    final decodedImage =
        await decodeImageFromList(byteData.buffer.asUint8List());
    setState(() {
      _imageOriginal = Image.asset(_stockImagePaths[_stockImageCounter]);
      _imageSize =
          Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
      _imagePath = _stockImagePaths[_stockImageCounter];
      _stockImageCounter = (_stockImageCounter + 1) % _stockImagePaths.length;
    });
  }

  void cleanResult() {
    _isAnalyzed = false;
    _faceDetectionResults = <FaceDetectionAbsolute>[];
    _imageDrawn = null;
    setState(() {});
  }

  void detectFaces() async {
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

    _faceDetectionResults = await _faceDetection.predict(_imagePath!);

    devtools.log('Inference completed');
    devtools.log('Inference results: list $_faceDetectionResults of length '
        '${_faceDetectionResults.length}');

    setState(() {
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
            SizedBox(
              height: 400,
              child: _imageOriginal != null
                  ? Stack(
                      children: [
                        _imageOriginal!,
                        if (_isAnalyzed)
                          CustomPaint(
                            painter: FacePainter(
                              faceDetections: _faceDetectionResults,
                              imageSize: _imageSize,
                            ),
                          ),
                      ],
                    )
                  : const Text('No image selected'),
            ),
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
                onPressed: _isAnalyzed ? cleanResult : detectFaces,
                style: TextButton.styleFrom(
                  foregroundColor:
                      Theme.of(context).colorScheme.onPrimaryContainer,
                  backgroundColor:
                      Theme.of(context).colorScheme.primaryContainer,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: _isAnalyzed
                    ? const Text('Clean result')
                    : const Text('Detect faces'),
              ),
            )
          ],
        ),
      ),
    );
  }
}
