import 'dart:core';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'src/call_sample/call_sample.dart';
import 'src/route_item.dart';

void main() => runApp(new MyApp());

const MaterialColor primaryBlack = MaterialColor(
  _blackPrimaryValue,
  <int, Color>{
    50: Color(0xFF000000),
    100: Color(0xFF000000),
    200: Color(0xFF000000),
    300: Color(0xFF000000),
    400: Color(0xFF000000),
    500: Color(_blackPrimaryValue),
    600: Color(0xFF000000),
    700: Color(0xFF000000),
    800: Color(0xFF000000),
    900: Color(0xFF000000),
  },
);
const int _blackPrimaryValue = 0xFF000000;

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => new _MyAppState();
}

enum DialogDemoAction {
  cancel,
  connect,
}

class _MyAppState extends State<MyApp> {
  List<RouteItem> items = [];
  String _server = '';
  late SharedPreferences _prefs;

  @override
  initState() {
    super.initState();
    _initData();
    _initItems();
  }

  _buildRow(context, item) {
    return Container(
      child: FloatingActionButton.extended(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(5)),
        label: Row(
          children: <Widget>[
            Text('Iniciar',
                style: TextStyle(
                    fontSize: 15.0,
                    letterSpacing: 2.0,
                    color: Color.fromARGB(255, 22, 22, 22))),
            Icon(Icons.arrow_right, color: Color.fromARGB(255, 22, 22, 22)),
          ],
        ),
        backgroundColor: Color.fromARGB(227, 176, 217, 236),
        onPressed: () => item.push(context),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: primaryBlack,
      ),
      home: Scaffold(
        appBar: AppBar(
          automaticallyImplyLeading: false,
          centerTitle: true,
          elevation: 0,
          title: Text('TCCam',
              style: TextStyle(color: Color.fromARGB(227, 176, 217, 236))),
          backgroundColor: Color.fromARGB(255, 22, 22, 22),
        ),
        body: SafeArea(
          child: Column(
            children: [
              Expanded(
                flex: 2,
                child: new Container(
                  color: Color.fromARGB(255, 22, 22, 22),
                  child: new Column(
                    children: <Widget>[
                      new Expanded(
                        flex: 1,
                        child: new Container(
                          padding: EdgeInsets.only(bottom: 40.0),
                          alignment: Alignment.bottomCenter,
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: <Widget>[
                              Container(
                                width: 100,
                                child: Icon(Icons.phone_iphone_outlined,
                                    color: Color.fromARGB(227, 176, 217, 236),
                                    size: 100.0),
                              ),
                              Container(
                                width: 100,
                                child: Text('...',
                                    style: TextStyle(
                                        fontSize: 80.0,
                                        color:
                                            Color.fromARGB(227, 176, 217, 236)),
                                    textAlign: TextAlign.center),
                              ),
                              Container(
                                width: 100,
                                child: Icon(Icons.desktop_windows_outlined,
                                    color: Color.fromARGB(227, 176, 217, 236),
                                    size: 100.0),
                              ),
                            ],
                          ),
                        ),
                      ),
                      new Expanded(
                        flex: 1,
                        child: new Container(
                          padding: EdgeInsets.only(top: 80.0),
                          alignment: Alignment.topCenter,
                          child: Column(
                            children: <Widget>[
                              Container(
                                child: Text('Deseja inicar uma transmissão?',
                                    style: TextStyle(
                                        letterSpacing: 2.0,
                                        fontSize: 20.0,
                                        fontWeight: FontWeight.bold,
                                        color:
                                            Color.fromARGB(227, 176, 217, 236)),
                                    textAlign: TextAlign.center),
                              ),
                              Container(
                                padding: EdgeInsets.only(top: 30.0),
                                width: 200,
                                child: ListView.builder(
                                  shrinkWrap: true,
                                  itemCount: items.length,
                                  itemBuilder: (context, i) {
                                    return _buildRow(context, items[i]);
                                  },
                                ),
                              )
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  _initData() async {
    _prefs = await SharedPreferences.getInstance();
    setState(() {
      _server = _prefs.getString('server') ?? 'ex: 0.0.0.0';
    });
  }

  void showDemoDialog<T>(
      {required BuildContext context, required Widget child}) {
    showDialog<T>(
      context: context,
      builder: (BuildContext context) => child,
    ).then<void>((T? value) {
      if (value != null) {
        if (value == DialogDemoAction.connect) {
          _prefs.setString('server', _server);
          Navigator.push(
              context,
              MaterialPageRoute(
                  builder: (BuildContext context) =>
                      CallSample(host: _server)));
        }
      }
    });
  }

  _showAddressDialog(context) {
    showDemoDialog<DialogDemoAction>(
      context: context,
      child: AlertDialog(
        backgroundColor: Color.fromARGB(255, 176, 217, 236),
        title: const Text('Digite o endereço do servidor:',
            style: TextStyle(color: Color.fromARGB(255, 32, 32, 32))),
        content: TextFormField(
          cursorColor: Color.fromARGB(227, 176, 217, 236),
          style: TextStyle(color: Color.fromARGB(255, 32, 32, 32)),
          onChanged: (String text) {
            setState(() {
              _server = text;
            });
          },
          decoration: InputDecoration(
              focusedBorder: OutlineInputBorder(
                borderSide: const BorderSide(
                    color: Color.fromARGB(255, 32, 32, 32), width: 2.0),
              ),
              hintText: _server,
              hintStyle: TextStyle(color: Color.fromARGB(255, 32, 32, 32))),
          textAlign: TextAlign.center,
        ),
        actions: <Widget>[
          TextButton(
              child: const Text('CANCELAR',
                  style: TextStyle(color: Color.fromARGB(255, 32, 32, 32))),
              onPressed: () {
                Navigator.pop(context, DialogDemoAction.cancel);
              }),
          TextButton(
            child: const Text('CONECTAR',
                style: TextStyle(color: Color.fromARGB(255, 32, 32, 32))),
            onPressed: () {
              Navigator.pop(context, DialogDemoAction.connect);
            },
          ),
        ],
      ),
    );
  }

  _initItems() {
    items = <RouteItem>[
      RouteItem(
          title: 'P2P Call Sample',
          subtitle: 'P2P Call Sample.',
          push: (BuildContext context) {
            _showAddressDialog(context);
          }),
    ];
  }
}
