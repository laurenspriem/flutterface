import 'dart:typed_data' show Uint8List;

import 'package:flutterface/services/face_ml/face_alignment/similarity_transform.dart';
import 'package:flutterface/services/face_ml/face_detection/detection.dart';
import 'package:flutterface/services/face_ml/face_detection/face_detection_exceptions.dart';
import 'package:flutterface/services/face_ml/face_detection/face_detection_service.dart';
import 'package:flutterface/services/face_ml/face_embedding/face_embedding_exceptions.dart';
import 'package:flutterface/services/face_ml/face_embedding/face_embedding_service.dart';
import 'package:flutterface/services/face_ml/face_ml_exceptions.dart';

class FaceMlService {
  // singleton pattern
  FaceMlService._();
  static final instance = FaceMlService._();

  /// Detects faces in the given image data.
  ///
  /// `imageData`: The image data to analyze.
  ///
  /// Returns a list of face detection results.
  ///
  /// Throws `CouldNotInitializeFaceDetector`, `CouldNotRunFaceDetector` or `GeneralFaceMlException` if something goes wrong.
  Future<List<FaceDetectionAbsolute>> detectFaces(Uint8List imageData) async {
    try {
      // Get (and initialize if necessary) the face detector singleton instance
      await FaceDetection.instance.init();

      // Get the bounding boxes of the faces
      final List<FaceDetectionAbsolute> faces =
          FaceDetection.instance.predict(imageData);

      return faces;
    } on BlazeFaceInterpreterInitializationException {
      throw CouldNotInitializeFaceDetector();
    } on BlazeFaceInterpreterRunException {
      throw CouldNotRunFaceDetector();
    // ignore: avoid_catches_without_on_clauses
    } catch (e) {
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
  Uint8List alignSingleFace(Uint8List imageData, FaceDetectionAbsolute face) {
    try {
    final faceLandmarks = face.allKeypoints.sublist(0, 4);
    final similarityTransform = SimilarityTransform();
    final isNoNanInParam = similarityTransform.estimate(faceLandmarks);
    if (!isNoNanInParam) {
      throw CouldNotEstimateSimilarityTransform();
    }

    final transformMatrix = similarityTransform.params;

    final Uint8List faceAlignedData = similarityTransform.warpAffine(
      imageData: imageData,
      transformationMatrix: transformMatrix,
      width: 112,
      height: 112,
    );

    return faceAlignedData;
    // ignore: avoid_catches_without_on_clauses
    } catch (e) {
      throw GeneralFaceMlException('Face alignment failed: $e');
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
  List<Uint8List> alignFaces(
    Uint8List imageData,
    List<FaceDetectionAbsolute> faces,
  ) {
    final alignedFaces = <Uint8List>[];
    for (int i = 0; i < faces.length; ++i) {
      final alignedFace = alignSingleFace(imageData, faces[i]);
      alignedFaces.add(alignedFace);
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
  Future<List<double>> embedSingleFace(Uint8List faceData) async {
    try {
    // Get (and initialize if necessary) the face detector singleton instance
    await FaceEmbedding.instance.init();

    // Get the embedding of the face
    final List<double> embedding = FaceEmbedding.instance.predict(faceData);

    return embedding;
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
