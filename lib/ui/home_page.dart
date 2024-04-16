import 'dart:async';
import 'dart:developer' as devtools show log;
import 'dart:io';
import 'dart:typed_data' show Uint8List;
// import 'dart:ui' as ui show Image;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutterface/services/face_ml/face_detection/detection.dart';
import 'package:flutterface/services/face_ml/face_ml_service.dart';
import 'package:flutterface/utils/face_detection_painter.dart';
import 'package:flutterface/utils/image_ml_util.dart';
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
  Image? faceAligned2;
  Image? faceCropped;
  Uint8List? imageOriginalData;
  Uint8List? faceAlignedData;
  Uint8List? faceAlignedData2;
  Uint8List? faceCroppedData;
  Size imageSize = const Size(0, 0);
  late Size imageDisplaySize;
  int stockImageCounter = 0;
  int faceFocusCounter = 0;
  int showingFaceCounter = 0;
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
  bool isFaceCropped = false;
  bool isEmbedded = false;
  List<FaceDetectionRelative> faceDetectionResultsRelative = [];
  List<FaceDetectionAbsolute> faceDetectionResultsAbsolute = [];
  List<double> faceEmbeddingResult = <double>[];
  double blurValue = 0;

  @override
  void initState() {
    super.initState();
    unawaited(FaceMlService.instance.init());
  }

  void _pickImage() async {
    cleanResult();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      imageOriginalData = await image.readAsBytes();
      final stopwatchImageDecoding = Stopwatch()..start();
      final decodedImage = await decodeImageFromList(imageOriginalData!);
      setState(() {
        final imagePath = image.path;

        imageOriginal = Image.file(File(imagePath));
        stopwatchImageDecoding.stop();
        devtools.log(
            'Image decoding took ${stopwatchImageDecoding.elapsedMilliseconds} ms');
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
    faceDetectionResultsAbsolute = <FaceDetectionAbsolute>[];
    faceDetectionResultsRelative = <FaceDetectionRelative>[];
    isAligned = false;
    isFaceCropped = false;
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

    faceDetectionResultsRelative =
        await FaceMlService.instance.detectFaces(imageOriginalData!);

    faceDetectionResultsAbsolute = relativeToAbsoluteDetections(
      relativeDetections: faceDetectionResultsRelative,
      imageWidth: imageSize.width.round(),
      imageHeight: imageSize.height.round(),
    );

    setState(() {
      isPredicting = false;
      isAnalyzed = true;
    });
  }

  void cropDetectedFace() async {
    if (imageOriginalData == null) {
      showResponseSnackbar(context, 'Please select an image first');
      return;
    }
    if (!isAnalyzed) {
      showResponseSnackbar(context, 'Please detect faces first');
      return;
    }
    if (faceDetectionResultsAbsolute.isEmpty) {
      showResponseSnackbar(context, 'No face detected, nothing to crop');
      return;
    }
    if (faceDetectionResultsAbsolute.length == 1 && isAligned) {
      showResponseSnackbar(context, 'This is the only face found in the image');
      return;
    }

    final face = faceDetectionResultsAbsolute[faceFocusCounter];
    try {
      final facesList = await generateFaceThumbnails(imageOriginalData!,
          faceDetections: [face]);
      faceCroppedData = facesList[0];
    } catch (e) {
      devtools.log('Alignment of face failed: $e');
      return;
    }

    setState(() {
      isFaceCropped = true;
      faceEmbeddingResult = [];
      embeddingStartIndex = 0;
      isEmbedded = false;
      faceCropped = Image.memory(faceCroppedData!);
      showingFaceCounter = faceFocusCounter;
      faceFocusCounter =
          (faceFocusCounter + 1) % faceDetectionResultsAbsolute.length;
    });
  }

  void alignFaceCustomInterpolation() async {
    if (imageOriginalData == null) {
      showResponseSnackbar(context, 'Please select an image first');
      return;
    }
    if (!isAnalyzed) {
      showResponseSnackbar(context, 'Please detect faces first');
      return;
    }
    if (faceDetectionResultsAbsolute.isEmpty) {
      showResponseSnackbar(context, 'No face detected, nothing to align');
      return;
    }
    if (faceDetectionResultsAbsolute.length == 1 && isAligned) {
      showResponseSnackbar(context, 'This is the only face found in the image');
      return;
    }

    final face = faceDetectionResultsAbsolute[faceFocusCounter];
    try {
      final bothFaces = await FaceMlService.instance
          .alignSingleFaceCustomInterpolation(imageOriginalData!, face);
      faceAlignedData = bothFaces[0];
      faceAlignedData2 = bothFaces[1];
    } catch (e) {
      devtools.log('Alignment of face failed: $e');
      return;
    }

    setState(() {
      isAligned = true;
      faceEmbeddingResult = [];
      embeddingStartIndex = 0;
      isEmbedded = false;
      faceAligned = Image.memory(faceAlignedData!);
      faceAligned2 = Image.memory(faceAlignedData2!);
      showingFaceCounter = faceFocusCounter;
      faceFocusCounter =
          (faceFocusCounter + 1) % faceDetectionResultsAbsolute.length;
    });
  }

  void alignFaceCanvasInterpolation() async {
    if (imageOriginalData == null) {
      showResponseSnackbar(context, 'Please select an image first');
      return;
    }
    if (!isAnalyzed) {
      showResponseSnackbar(context, 'Please detect faces first');
      return;
    }
    if (faceDetectionResultsAbsolute.isEmpty) {
      showResponseSnackbar(context, 'No face detected, nothing to align');
      return;
    }
    if (faceDetectionResultsAbsolute.length == 1 && isAligned) {
      showResponseSnackbar(context, 'This is the only face found in the image');
      return;
    }

    final face = faceDetectionResultsAbsolute[faceFocusCounter];
    try {
      faceAlignedData = await FaceMlService.instance
          .alignSingleFaceCanvasInterpolation(imageOriginalData!, face);
    } catch (e) {
      devtools.log('Alignment of face failed: $e');
      return;
    }

    setState(() {
      isAligned = true;
      faceEmbeddingResult = [];
      embeddingStartIndex = 0;
      isEmbedded = false;
      faceAligned = Image.memory(faceAlignedData!);
      showingFaceCounter = faceFocusCounter;
      faceFocusCounter =
          (faceFocusCounter + 1) % faceDetectionResultsAbsolute.length;
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

    final (faceEmbeddingResultLocal, isBlurLocal, blurValueLocal) =
        await FaceMlService.instance.embedSingleFace(
      imageOriginalData!,
      faceDetectionResultsRelative[showingFaceCounter],
    );
    faceEmbeddingResult = faceEmbeddingResultLocal;
    blurValue = blurValueLocal;
    devtools.log('Blur detected: $isBlurLocal, blur value: $blurValueLocal');
    // devtools.log('Embedding: $faceEmbeddingResult');

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
              child: Stack(
                alignment: Alignment.bottomCenter,
                children: [
                  // Image container
                  Center(
                    child: imageOriginal != null
                        ? isAligned
                            ? Center(
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    Column(
                                      children: [
                                        faceAligned!,
                                        const Text(
                                          'Bilinear',
                                          style: TextStyle(color: Colors.white),
                                        ),
                                      ],
                                    ),
                                    const SizedBox(width: 10),
                                    Column(
                                      children: [
                                        faceAligned2!,
                                        const Text(
                                          'Bicubic',
                                          style: TextStyle(color: Colors.white),
                                        ),
                                      ],
                                    ),
                                  ],
                                ),
                              )
                            : Stack(
                                children: [
                                  imageOriginal!,
                                  if (isAnalyzed)
                                    CustomPaint(
                                      painter: FacePainter(
                                        faceDetections:
                                            faceDetectionResultsAbsolute,
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
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Row(
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
                    : const SizedBox(height: 48),
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
                          Text('Blur: ${blurValue.round()}'),
                        ],
                      )
                    : const SizedBox(height: 48),
                embeddingStartIndex + 2 < faceEmbeddingResult.length
                    ? IconButton(
                        icon: const Icon(Icons.arrow_forward),
                        onPressed: nextEmbedding,
                      )
                    : const SizedBox(height: 48),
              ],
            ),
            ElevatedButton.icon(
              icon: isAnalyzed
                  ? const Icon(Icons.person_remove_outlined)
                  : const Icon(Icons.people_alt_outlined),
              label: isAnalyzed
                  ? const Text('Clean result')
                  : const Text('Detect faces'),
              onPressed: isAnalyzed ? cleanResult : detectFaces,
            ),
            isAnalyzed
                ? ElevatedButton.icon(
                    icon: const Icon(Icons.face_retouching_natural),
                    label: const Text('Align faces'),
                    onPressed: alignFaceCustomInterpolation,
                  )
                : const SizedBox.shrink(),
            (isAligned && !isEmbedded)
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
