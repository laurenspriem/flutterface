# FlutterFace

A basic demo app for running face recognition locally on your phone using [Flutter](https://flutter.dev), [TensorFlow Lite](https://www.tensorflow.org/lite) and [ONNX Runtime](https://onnxruntime.ai). 
This is possible with the use of the [tflite_flutter](https://pub.dev/packages/tflite_flutter) and [onnxruntime](https://pub.dev/packages/onnxruntime) plugins. 

We use [YOLOv5Face](https://arxiv.org/abs/2105.12931) for face detection and [MobileFaceNet](https://arxiv.org/abs/1804.07573) for creating embeddings.


## 🧑‍💻 Running from source

1. [Install Flutter v3.13.4](https://flutter.dev/docs/get-started/install)
2. Clone this repository with `git clone git@github.com:laurenspriem/flutterface.git`
3. Fix dependencies using `flutter pub get`
4. Attach mobile device and run `flutter run`