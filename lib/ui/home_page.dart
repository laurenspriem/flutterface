import 'dart:developer' as devtools show log;
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key, required this.title});

  final String title;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final ImagePicker picker = ImagePicker();
  XFile? _image;
  Image? _imageWidget;

  void _pickImage() async {
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _image = image;
        _imageWidget = Image.file(File(_image!.path));
      });
    } else {
      devtools.log('No image selected');
    }
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
            (_image != null)
                ? SizedBox(
                    height: 400,
                    child: _imageWidget,
                  )
                : const Text('No image selected'),
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
                onPressed: () {},
                style: TextButton.styleFrom(
                  foregroundColor:
                      Theme.of(context).colorScheme.onPrimaryContainer,
                  backgroundColor:
                      Theme.of(context).colorScheme.primaryContainer,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                ),
                child: const Text('Detect faces'),
              ),
            )
          ],
        ),
      ),
    );
  }
}