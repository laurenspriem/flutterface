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
  Image? imageOriginal;
  Image? faceAligned;
  Uint8List? imageOriginalData;
  Size imageSize = const Size(0, 0);
  int stockImageCounter = 0;
  int faceFocusCounter = 0;
  final List<String> _stockImagePaths = [
    'assets/images/stock_images/one_person.jpeg',
    'assets/images/stock_images/one_person2.jpeg',
    'assets/images/stock_images/one_person3.jpeg',
    'assets/images/stock_images/one_person4.jpeg',
    'assets/images/stock_images/group_of_people.jpeg',
  ];

  bool isAnalyzed = false;
  bool isModelLoaded = false;
  bool isPredicting = false;
  bool isAligned = false;
  late FaceDetection faceDetection;
  List<FaceDetectionAbsolute> faceDetectionResults = [];

  void _pickImage() async {
    cleanResult();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      imageOriginalData = await image.readAsBytes();
      final decodedImage = await decodeImageFromList(imageOriginalData!);
      setState(() {
        final imagePath = image.path;
        imageOriginal = Image.file(File(imagePath));
        imageSize =
            Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
      });
    } else {
      devtools.log('No image selected');
    }
  }

  void _stockImage() async {
    cleanResult();
    final byteData =
        await rootBundle.load(_stockImagePaths[stockImageCounter]);
    imageOriginalData = byteData.buffer.asUint8List();
    final decodedImage = await decodeImageFromList(imageOriginalData!);
    setState(() {
      imageOriginal = Image.asset(_stockImagePaths[stockImageCounter]);
      imageSize =
          Size(decodedImage.width.toDouble(), decodedImage.height.toDouble());
      stockImageCounter = (stockImageCounter + 1) % _stockImagePaths.length;
    });
  }

  void cleanResult() {
    isAnalyzed = false;
    faceDetectionResults = <FaceDetectionAbsolute>[];
    isAligned = false;
    faceFocusCounter = 0;
    setState(() {});
  }

  void detectFaces() async {
    if (imageOriginalData == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please select an image first'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }
    if (isAnalyzed || isPredicting) {
      return;
    }

    setState(() {
      isPredicting = true;
    });

    devtools.log('Image is sent to the model for inference');

    // 'Image plane data length: ${_imageWidget.planes[0].bytes.length}');
    if (!isModelLoaded) {
      faceDetection = await FaceDetection.create();
      isModelLoaded = true;
    }

    faceDetectionResults = await faceDetection.predict(imageOriginalData!);

    devtools.log('Inference completed');
    devtools.log('Inference results: list $faceDetectionResults of length '
        '${faceDetectionResults.length}');

    setState(() {
      isPredicting = false;
      isAnalyzed = true;
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
    if (!isAnalyzed) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please detect faces first'),
          duration: Duration(seconds: 2),
        ),
      );
      return;
    }
    if (faceDetectionResults[0].score < 0.01) {
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
    if (faceDetectionResults.length == 1 && isAligned) {
      {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('This is the only face found in the image'),
            duration: Duration(seconds: 2),
          ),
        );
        return;
      }
    }

    final face = faceDetectionResults[faceFocusCounter];
    final faceLandmarks = face.allKeypoints.sublist(0, 4);
    final tform = SimilarityTransform();
    final isNoNanInParam = tform.estimate(faceLandmarks);
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

    setState(() {
      isAligned = true;
      faceAligned = Image.memory(warpedFace);
      faceFocusCounter = (faceFocusCounter + 1) % faceDetectionResults.length;
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
              child: imageOriginal != null
                  ? isAligned
                      ? faceAligned
                      : Stack(
                          children: [
                            imageOriginal!,
                            if (isAnalyzed)
                              CustomPaint(
                                painter: FacePainter(
                                  faceDetections: faceDetectionResults,
                                  imageSize: imageSize,
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
                onPressed: isAnalyzed ? cleanResult : detectFaces,
                style: TextButton.styleFrom(
                  foregroundColor:
                      Theme.of(context).colorScheme.onPrimaryContainer,
                  backgroundColor:
                      Theme.of(context).colorScheme.primaryContainer,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: isAnalyzed
                    ? const Text('Clean result')
                    : const Text('Detect faces'),
              ),
            ),
            isAnalyzed
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
