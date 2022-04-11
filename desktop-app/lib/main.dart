import 'dart:core';
import 'package:flutter/material.dart';
import 'src/call_sample/call_sample.dart';
import 'src/route_item.dart';

void main() {
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
