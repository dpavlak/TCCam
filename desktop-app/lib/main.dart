import 'dart:core';
import 'package:flutter/material.dart';
import 'src/call_sample/call_sample.dart';
import 'src/route_item.dart';
import 'dart:io';

void main() async {
  String dir = (Directory.current.path +
      '\\lib\\src\\win-dshow\\virtualcam-install.bat');

  Process.run('Powershell -Command Start-Process "$dir" -Verb RunAs', [],
          runInShell: true)
      .then((ProcessResult result) {
    print(result.stderr);
  });

  runApp(new MyApp(initialRoute: '/CallSample'));
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => new _MyAppState();
  final String initialRoute;

  MyApp({required this.initialRoute});
}

class _MyAppState extends State<MyApp> {
  List<RouteItem> items = [];
  String _server = 'localhost';

  @override
  initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      routes: {
        '/': (context) => CallSample(host: _server),
      },
    );
  }
}
