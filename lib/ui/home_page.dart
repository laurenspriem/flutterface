import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:typed_data' show Uint8List;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutterface/services/face_alignment/similarity_transform.dart';
import 'package:flutterface/services/face_detection/detection.dart';
import 'package:flutterface/services/face_detection/face_detection_service.dart';
import 'package:flutterface/services/face_embedding/face_embedding_service.dart';
import 'package:flutterface/utils/face_detection_painter.dart';
import 'package:flutterface/utils/snackbar_message.dart';
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
  Uint8List? faceAlignedData;
  Size imageSize = const Size(0, 0);
  late Size imageDisplaySize;
  int stockImageCounter = 0;
  int faceFocusCounter = 0;
  int embeddingStartIndex = 0;
  final List<String> _stockImagePaths = [
    'assets/images/stock_images/one_person.jpeg',
    'assets/images/stock_images/one_person2.jpeg',
    'assets/images/stock_images/one_person3.jpeg',
    'assets/images/stock_images/one_person4.jpeg',
    'assets/images/stock_images/group_of_people.jpeg',
  ];

  bool isAnalyzed = false;
  bool isBlazeFaceLoaded = false;
  bool isFaceNetLoaded = false;
  bool isPredicting = false;
  bool isAligned = false;
  bool isEmbedded = false;
  late FaceDetection faceDetection;
  late FaceEmbedding faceEmbedding;
  List<FaceDetectionAbsolute> faceDetectionResults = [];
  List faceEmbeddingResult = [];

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
    final byteData = await rootBundle.load(_stockImagePaths[stockImageCounter]);
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
    faceAlignedData = null;
    faceFocusCounter = 0;
    isEmbedded = false;
    embeddingStartIndex = 0;
    faceEmbeddingResult = [];
    setState(() {});
  }

  void detectFaces() async {
    if (imageOriginalData == null) {
      showResponseSnackbar(context, 'Please select an image first');
      return;
    }
    if (isAnalyzed || isPredicting) {
      return;
    }

    setState(() {
      isPredicting = true;
    });

    // 'Image plane data length: ${_imageWidget.planes[0].bytes.length}');
    if (!isBlazeFaceLoaded) {
      faceDetection = await FaceDetection.create();
      isBlazeFaceLoaded = true;
    }

    faceDetectionResults = faceDetection.predict(imageOriginalData!);

    setState(() {
      isPredicting = false;
      isAnalyzed = true;
    });
  }

  void alignFace() {
    if (imageOriginalData == null) {
      showResponseSnackbar(context, 'Please select an image first');
      return;
    }
    if (!isAnalyzed) {
      showResponseSnackbar(context, 'Please detect faces first');
      return;
    }
    if (faceDetectionResults[0].score < 0.01) {
      showResponseSnackbar(context, 'No face detected, nothing to align');
      return;
    }
    if (faceDetectionResults.length == 1 && isAligned) {
      showResponseSnackbar(context, 'This is the only face found in the image');
      return;
    }

    final face = faceDetectionResults[faceFocusCounter];
    final faceLandmarks = face.allKeypoints.sublist(0, 4);
    final tform = SimilarityTransform();
    final isNoNanInParam = tform.estimate(faceLandmarks);
    if (!isNoNanInParam) {
      showResponseSnackbar(
        context,
        'Something is going wrong in the transformation estimation',
      );
      return;
    }

    final transformMatrix = tform.params;

    faceAlignedData = tform.warpAffine(
      imageData: imageOriginalData!,
      transformationMatrix: transformMatrix,
      width: 112,
      height: 112,
    );

    setState(() {
      isAligned = true;
      faceEmbeddingResult = [];
      embeddingStartIndex = 0;
      isEmbedded = false;
      faceAligned = Image.memory(faceAlignedData!);
      faceFocusCounter = (faceFocusCounter + 1) % faceDetectionResults.length;
    });
  }

  void embedFace() async {
    if (isAligned == false) {
      showResponseSnackbar(context, 'Please align face first');
      return;
    }

    setState(() {
      isPredicting = true;
    });

    if (!isFaceNetLoaded) {
      faceEmbedding = await FaceEmbedding.create();
      isFaceNetLoaded = true;
    }

    faceEmbeddingResult = faceEmbedding.predict(faceAlignedData!);

    setState(() {
      isPredicting = false;
      isEmbedded = true;
    });
  }

  void nextEmbedding() {
    setState(() {
      embeddingStartIndex =
          (embeddingStartIndex + 2) % faceEmbeddingResult.length;
    });
  }

  void prevEmbedding() {
    setState(() {
      embeddingStartIndex =
          (embeddingStartIndex - 2) % faceEmbeddingResult.length;
    });
  }

  @override
  Widget build(BuildContext context) {
    imageDisplaySize = Size(
      MediaQuery.of(context).size.width * 0.8,
      MediaQuery.of(context).size.width * 0.8 * 1.5,
    );
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.surface,
        title: Text(
          widget.title,
          style: Theme.of(context).textTheme.headlineSmall,
        ),
        centerTitle: true,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: <Widget>[
            Container(
              height: imageDisplaySize.height,
              width: imageDisplaySize.width,
              color: Colors.black,
              padding: const EdgeInsets.all(8.0),
              child: Stack(
                alignment: Alignment.bottomCenter,
                children: [
                  // Image container
                  Center(
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
                                        availableSize: imageDisplaySize,
                                      ),
                                    ),
                                ],
                              )
                        : const Text(
                            'No image selected',
                            style: TextStyle(color: Colors.white),
                          ),
                  ),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      ElevatedButton.icon(
                        icon: const Icon(
                          Icons.image,
                          color: Colors.black,
                          size: 16,
                        ),
                        label: const Text(
                          'Gallery',
                          style: TextStyle(color: Colors.black, fontSize: 10),
                        ),
                        onPressed: _pickImage,
                        style: ElevatedButton.styleFrom(
                          minimumSize: const Size(50, 30),
                          backgroundColor: Colors.grey[200], // Button color
                          foregroundColor: Colors.black,
                          elevation: 1,
                        ),
                      ),
                      ElevatedButton.icon(
                        icon: const Icon(
                          Icons.collections,
                          color: Colors.black,
                          size: 16,
                        ),
                        label: const Text(
                          'Stock',
                          style: TextStyle(color: Colors.black, fontSize: 10),
                        ),
                        onPressed: _stockImage,
                        style: ElevatedButton.styleFrom(
                          minimumSize: const Size(50, 30),
                          backgroundColor: Colors.grey[200], // Button color
                          foregroundColor: Colors.black,
                          elevation: 1, // Elevation (shadow)
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: <Widget>[
                embeddingStartIndex > 0
                    ? IconButton(
                        icon: const Icon(Icons.arrow_back),
                        onPressed: prevEmbedding,
                      )
                    : const SizedBox(width: 48),
                isEmbedded
                    ? Column(
                        children: [
                          Text(
                            'Embedding: ${faceEmbeddingResult[embeddingStartIndex]}',
                          ),
                          if (embeddingStartIndex + 1 <
                              faceEmbeddingResult.length)
                            Text(
                              '${faceEmbeddingResult[embeddingStartIndex + 1]}',
                            ),
                        ],
                      )
                    : const SizedBox(height: 48),
                embeddingStartIndex + 2 < faceEmbeddingResult.length
                    ? IconButton(
                        icon: const Icon(Icons.arrow_forward),
                        onPressed: nextEmbedding,
                      )
                    : const SizedBox(width: 48),
              ],
            ),
            ElevatedButton.icon(
              icon: const Icon(Icons.people_alt_outlined),
              label: isAnalyzed
                  ? const Text('Clean result')
                  : const Text('Detect faces'),
              onPressed: isAnalyzed ? cleanResult : detectFaces,
            ),
            isAnalyzed
                ? ElevatedButton.icon(
                    icon: const Icon(Icons.face_retouching_natural),
                    label: const Text('Align faces'),
                    onPressed: alignFace,
                  )
                : const SizedBox.shrink(),
            isAligned
                ? ElevatedButton.icon(
                    icon: const Icon(Icons.numbers_outlined),
                    label: const Text('Embed face'),
                    onPressed: embedFace,
                  )
                : const SizedBox.shrink(),
          ],
        ),
      ),
    );
  }
}
