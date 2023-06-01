import 'package:equatable/equatable.dart';
// import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

// ignore: must_be_immutable
abstract class AIModel extends Equatable {
  AIModel({this.interpreter});

  final outputShapes = <List<int>>[];
  final outputTypes = <TensorType>[];

  Interpreter? interpreter;

  @override
  List<Object> get props => [];

  int get getAddress;

  Future<void> loadModel();
  Future<List<List<List<num>>>> getPreprocessedImage(String imagePath);
  List<Map<String, dynamic>> predict(List<List<List<num>>> imagePath);
}
