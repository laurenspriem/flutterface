import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:typed_data' show Uint8List;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutterface/services/face_alignment/similarity_transform.dart';
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
  Uint8List? imageOriginalData;
  Image? _imageAligned;
  Matrix4? transformMatrix;
  Size _imageSize = const Size(0, 0);
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
  bool _isAligned = false;
  late FaceDetection _faceDetection;
  List<FaceDetectionAbsolute> _faceDetectionResults = [];

  void _pickImage() async {
    cleanResult();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      imageOriginalData = await image.readAsBytes();
      final decodedImage = await decodeImageFromList(imageOriginalData!);
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
    imageOriginalData = byteData.buffer.asUint8List();
    final decodedImage = await decodeImageFromList(imageOriginalData!);
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
    _isAligned = false;
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

  void alignFace() {
    if (imageOriginalData == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select an image first'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }
    if (!_isAnalyzed) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please detect faces first'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }
    if (_faceDetectionResults[0].score < 0.01) {
      {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('No face detected, nothing to transform/align'),
            duration: Duration(seconds: 2),
          ),
        );
        return;
      }
    }

    final firstFace = _faceDetectionResults[0];
    final firstLandmarks = firstFace.allKeypoints.sublist(0, 4);
    final tform = SimilarityTransform();
    final isNoNanInParam = tform.estimate(firstLandmarks);
    if (!isNoNanInParam) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content:
              Text('Something is going wrong in the transformation estimation'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }

    final transformMatrix = tform.params;

    final warpedFace = tform.warpAffine(
      imageData: imageOriginalData!,
      transformationMatrix: transformMatrix,
      width: 112,
      height: 112,
    );

    _imageAligned = Image.memory(warpedFace);

    setState(() {
      _isAligned = true;
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
                  ? _isAligned
                      ? _imageAligned
                      : Stack(
                          children: [
                            _imageOriginal!,
                            if (_isAnalyzed)
                              CustomPaint(
                                painter: FacePainter(
                                  faceDetections: _faceDetectionResults,
                                  imageSize: _imageSize,
                                  availableSize: Size(
                                    MediaQuery.of(context).size.width,
                                    400,
                                  ),
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
            ),
            _isAnalyzed
                ? SizedBox(
                    width: 150,
                    child: TextButton(
                      onPressed: alignFace,
                      style: TextButton.styleFrom(
                        foregroundColor:
                            Theme.of(context).colorScheme.onPrimaryContainer,
                        backgroundColor:
                            Theme.of(context).colorScheme.primaryContainer,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                      child: const Text('Align face'),
                    ),
                  )
                : const SizedBox.shrink(),
          ],
        ),
      ),
    );
  }
}
