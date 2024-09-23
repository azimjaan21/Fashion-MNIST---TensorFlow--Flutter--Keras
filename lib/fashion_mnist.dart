import 'package:flutter/material.dart';

class FashionMNIST extends StatelessWidget {
  const FashionMNIST({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      body: Center(
        child: Text(
          'TensorFlow & Flutter',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 20),
        ),
      ),
    );
  }
}
