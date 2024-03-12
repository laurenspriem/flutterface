import 'dart:developer' show log;
import 'dart:typed_data' show Uint8List;

import 'package:flutterface/services/face_ml/face_detection/detection.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_face_detection_exceptions.dart';
import 'package:flutterface/services/face_ml/face_detection/yolov5face/yolo_face_detection_onnx.dart';
import 'package:flutterface/services/face_ml/face_embedding/face_embedding_exceptions.dart';
import 'package:flutterface/services/face_ml/face_embedding/face_embedding_onnx.dart';
import 'package:flutterface/services/face_ml/face_embedding/face_embedding_service.dart';
import 'package:flutterface/services/face_ml/face_ml_exceptions.dart';
import 'package:flutterface/utils/image_ml_isolate.dart';
import 'package:logging/logging.dart';

class FaceMlService {
  final _logger = Logger('FaceMlService');

  // singleton pattern
  FaceMlService._privateConstructor();
  static final instance = FaceMlService._privateConstructor();
  factory FaceMlService() => instance;

  bool initialized = false;

  Future<void> init() async {
    _logger.info('init called');
    try {
      await YoloOnnxFaceDetection.instance.init();
    } catch (e, s) {
      _logger.severe('Could not initialize YOLO', e, s);
    }
    try {
      await ImageMlIsolate.instance.init();
    } catch (e, s) {
      _logger.severe('Could not initialize image ml isolate', e, s);
    }
    try {
      // await FaceEmbedding.instance.init();
      await FaceEmbeddingOnnx.instance.init();
    } catch (e, s) {
      _logger.severe('Could not initialize mobilefacenet', e, s);
    }
    initialized = true;
  }

  /// Detects faces in the given image data.
  ///
  /// `imageData`: The image data to analyze.
  ///
  /// Returns a list of face detection results.
  ///
  /// Throws [CouldNotInitializeFaceDetector], [CouldNotRunFaceDetector] or [GeneralFaceMlException] if something goes wrong.
  Future<List<FaceDetectionRelative>> detectFaces(Uint8List imageData) async {
    try {
      // Get the bounding boxes of the faces
      final List<FaceDetectionRelative> faces =
          await YoloOnnxFaceDetection.instance.predict(imageData);

      return faces;
    } on YOLOInterpreterInitializationException {
      throw CouldNotInitializeFaceDetector();
    } on YOLOInterpreterRunException {
      throw CouldNotRunFaceDetector();
      // ignore: avoid_catches_without_on_clauses
    } catch (e) {
      _logger.severe('Face detection failed: $e');
      throw GeneralFaceMlException('Face detection failed: $e');
    }
  }

  /// Aligns a single face from the given image data.
  ///
  /// `imageData`: The image data that contains the face.
  /// `face`: The face detection result for the face to align.
  ///
  /// Returns the aligned face as image data.
  ///
  /// Throws `CouldNotEstimateSimilarityTransform` or `GeneralFaceMlException` if the face alignment fails.
  Future<List<Uint8List>> alignSingleFaceCustomInterpolation(
    Uint8List imageData,
    FaceDetectionAbsolute face,
  ) async {
    try {
      // final faceLandmarks = face.allKeypoints.sublist(0, 4);
      // final similarityTransform = SimilarityTransform();
      // final isNoNanInParam = similarityTransform.estimate(faceLandmarks);
      // if (!isNoNanInParam) {
      //   throw CouldNotEstimateSimilarityTransform();
      // }

      // final List<List<double>> transformMatrix = similarityTransform.params;

      // final Uint8List faceAlignedData = await similarityTransform.warpAffine(
      //   imageData: imageData,
      //   transformationMatrix: transformMatrix,
      //   width: 112,
      //   height: 112,
      // );

      final List<Uint8List> faceAlignedData = await ImageMlIsolate.instance
          .preprocessFaceAlignCustom(imageData, [face]);

      return faceAlignedData;
      // ignore: avoid_catches_without_on_clauses
    } catch (e, s) {
      throw GeneralFaceMlException('Face alignment failed: $e \n $s');
    }
  }

  Future<Uint8List> alignSingleFaceCanvasInterpolation(
    Uint8List imageData,
    FaceDetectionAbsolute face,
  ) async {
    try {
      final List<Uint8List> faceAlignedData = await ImageMlIsolate.instance
          .preprocessFaceAlignCanvas(imageData, [face]);

      return faceAlignedData[0];
      // ignore: avoid_catches_without_on_clauses
    } catch (e, s) {
      throw GeneralFaceMlException('Face alignment failed: $e \n $s');
    }
  }

  /// Aligns multiple faces from the given image data.
  ///
  /// `imageData`: The image data that contains the faces.
  /// `faces`: The face detection results for the faces to align.
  ///
  /// Returns a list of the aligned faces as image data.
  ///
  /// Throws `CouldNotEstimateSimilarityTransform` or `GeneralFaceMlException` if the face alignment fails.
  // TODO: Make this function more efficient so that it only has to do the Image.Image conversion once
  Future<List<Uint8List>> alignFaces(
    Uint8List imageData,
    List<FaceDetectionAbsolute> faces,
  ) async {
    final alignedFaces = <Uint8List>[];
    for (int i = 0; i < faces.length; ++i) {
      final alignedFace =
          await alignSingleFaceCustomInterpolation(imageData, faces[i]);
      alignedFaces.add(alignedFace[0]);
    }

    return alignedFaces;
  }

  /// Embeds a single face from the given image data.
  ///
  /// `faceData`: The image data of the face to embed.
  ///
  /// Returns the face embedding as a list of doubles.
  ///
  /// Throws `CouldNotInitializeFaceEmbeddor`, `CouldNotRunFaceEmbeddor` or `GeneralFaceMlException` if the face embedding fails.
  Future<(List<double>, bool, double)> embedSingleFace(
    Uint8List imageData,
    FaceDetectionRelative face,
  ) async {
    try {
      // await FaceEmbedding.instance.init();
      // final (List<double> embedding, bool isBlur, double blurValue) =
      //     await FaceEmbedding.instance.predict(imageData, face);

      await FaceEmbeddingOnnx.instance.init();
      final (List<double> embedding, bool isBlur, double blurValue) =
          await FaceEmbeddingOnnx.instance.predict(imageData, face);

      // log('Embedding: $embedding');

      return (embedding, isBlur, blurValue);
    } on MobileFaceNetInterpreterInitializationException {
      throw CouldNotInitializeFaceEmbeddor();
    } on MobileFaceNetInterpreterRunException {
      throw CouldNotRunFaceEmbeddor();
      // ignore: avoid_catches_without_on_clauses
    } catch (e) {
      throw GeneralFaceMlException('Face embedding failed: $e');
    }
  }

  // TODO: implement `embedBatchFaces
  // Future<List<double>> embedBatchFaces(List<Uint8List> faceData) async {
  //   // Get (and initialize if necessary) the face detector singleton instance
  //   await FaceEmbedding.instance.init();

  //   // Get the embedding of the face
  //   final List<double> embedding = FaceEmbedding.instance.predict(faceData);

  //   return embedding;
  // }
}
