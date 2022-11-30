import 'dart:convert';
import 'dart:core';
import 'dart:io';
import 'package:flutter/material.dart';
import 'src/call_sample/call_sample.dart';
import 'src/route_item.dart';
import 'globals.dart' as globals;
import 'package:shell/shell.dart';
import 'package:process_run/which.dart';

void main() async {
  String dir = (Directory.current.path +
      '\\lib\\src\\win-dshow\\virtualcam-install.bat');

  //String dir2 = (Directory.current.path + '\\lib\\src\\server\\teste2.py');

  /* Process.run('Powershell -Command Start-Process "$dir" -Verb RunAs', [],
      runInShell: true); */

  /* Process.runSync('Powershell.exe', ['-File', dir2]);

  Process.start('$dir2', ['go run main.go']);

  var shell = new Shell();
  shell.start('echo', arguments: ['hello world']);


  Process.run('python', [dir2], runInShell: true).then((ProcessResult result) {
    print(result.stderr);
  });

  Process.run('go run', [dir2], runInShell: true).then((ProcessResult result) {
    print(result.stderr);
  });
 */
  Process.run('Powershell -Command Start-Process "$dir" -Verb RunAs', [],
          runInShell: true)
      .then((ProcessResult result) async {
    print(result.exitCode);
    if (result.exitCode == 0) {
      globals.serverUp = true;

      final interfaces = await NetworkInterface.list(
          type: InternetAddressType.IPv4, includeLinkLocal: true);
      String serverIp = (interfaces[0].addresses.first.address).toString();

      globals.serverAdress = serverIp;
    }
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
      debugShowCheckedModeBanner: false,
      routes: {
        '/': (context) => CallSample(host: _server),
      },
    );
  }
}
