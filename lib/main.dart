// ignore_for_file: avoid_print, library_private_types_in_public_api

import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File? _image;
  List? _predictions;
  bool _loading = false;
  late Interpreter _interpreter;

  // Class names for Fashion MNIST
  final List<String> classNames = [
   
   
    'Bag',
    'Ankle boot',
     'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
     'Sandal',
    'T-shirt',
    'Sneaker',
  ];

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  // Load the TFLite model
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/fashion_classifier.tflite');
      print('Model loaded successfully.');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  // Pick an image from the gallery
  Future<void> pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _loading = true;
      });
      classifyImage(_image!);
    }
  }

  // Classify the selected image
  Future<void> classifyImage(File image) async {
    // Load and preprocess the image
    img.Image? originalImage = img.decodeImage(image.readAsBytesSync());

    // Convert to grayscale (Fashion MNIST uses grayscale images)
    img.Image grayscaleImage = img.grayscale(originalImage!);

    // Resize the image to 28x28 pixels
    img.Image resizedImage = img.copyResize(grayscaleImage, width: 28, height: 28);

    // Get pixel values as a list of integers (grayscale, so we only need one channel)
    List<int> pixelValues = resizedImage.getBytes();
    if (pixelValues.length > 784) {
      pixelValues = pixelValues.sublist(0, 784);
    }

    // Prepare the input tensor as a Float32List
    Float32List input = Float32List(28 * 28);

    // Normalize pixel values to range [0, 1]
    for (int i = 0; i < pixelValues.length; i++) {
      input[i] = pixelValues[i] / 255.0;
    }

    // Dynamically get the output shape from the interpreter
    var outputShape = _interpreter.getOutputTensor(0).shape;
    print('Output shape from model: $outputShape');

    // Prepare the output tensor to match the shape
    var output = List.filled(outputShape[1], 0.0).reshape([1, outputShape[1]]);

    // Run inference
    _interpreter.run(input.reshape([1, 28, 28, 1]), output);

    setState(() {
      _predictions = output[0];  // Use only the first batch's predictions

      // Print the raw output scores
      print('Raw output scores: $_predictions');

      // Find the index of the class with the highest probability
      double maxScore = (_predictions as List<double>).reduce((a, b) => a > b ? a : b);
      int predictedClassIndex = (_predictions as List<double>).indexOf(maxScore);

      // Debug: Print class scores
      for (int i = 0; i < classNames.length; i++) {
        print('Class ${classNames[i]}: ${_predictions?[i]}');
      }

      // Check confidence
      String predictedClassName;
      double confidenceThreshold = 0.5; // Adjust this as needed

      if (maxScore < confidenceThreshold) {
        predictedClassName = 'Uncertain';
      } else {
        predictedClassName = classNames[predictedClassIndex];
      }

      // Debug: Print the predicted index and class name
      print('Predicted class index: $predictedClassIndex');
      print('Predicted class name: $predictedClassName');

      // Update to show the predicted class name
      _predictions = [predictedClassName];
      _loading = false;
    });
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(
          centerTitle: true,
          title: const Text('Fashion MNIST Classifier'),
        ),
        body: Center(
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _image == null ? const Text('No image selected.') : Image.file(_image!),
                const SizedBox(height: 16),
                _loading
                    ? const CircularProgressIndicator()
                    : _predictions != null
                        ? Text(
                            "Predicted: ${_predictions![0]}",
                            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                          )
                        : Container(),
                const SizedBox(height: 20),
                ElevatedButton(
                  onPressed: pickImage,
                  child: const Text('Pick Image from Gallery'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
