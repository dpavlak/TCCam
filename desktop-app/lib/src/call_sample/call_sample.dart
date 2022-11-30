import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'dart:core';
import 'signaling.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import '../../globals.dart' as globals;

class CallSample extends StatefulWidget {
  static String tag = 'call_sample';
  final String host;
  CallSample({required this.host});

  @override
  _CallSampleState createState() => _CallSampleState();
}

class _CallSampleState extends State<CallSample> {
  Signaling? _signaling;
  List<dynamic> _peers = [];
  String? _selfId;
  RTCVideoRenderer _remoteRenderer = RTCVideoRenderer();
  bool _inCalling = false;
  Session? _session;
  Timer? timer;

  bool _waitAccept = false;

  // ignore: unused_element
  _CallSampleState();

  @override
  initState() {
    super.initState();
    initRenderers();
    _connect();
    timer = Timer.periodic(Duration(seconds: 1), (Timer t) => updateServer());
  }

  initRenderers() async {
    await _remoteRenderer.initialize();
  }

  @override
  deactivate() {
    super.deactivate();
    timer?.cancel();
    _signaling?.close();
    _remoteRenderer.dispose();
  }

  void _connect() async {
    _signaling ??= Signaling(widget.host)..connect();
    _signaling?.onSignalingStateChange = (SignalingState state) {
      switch (state) {
        case SignalingState.ConnectionClosed:
        case SignalingState.ConnectionError:
        case SignalingState.ConnectionOpen:
          break;
      }
    };

    _signaling?.onCallStateChange = (Session session, CallState state) async {
      switch (state) {
        case CallState.CallStateNew:
          setState(() {
            _session = session;
          });
          break;
        case CallState.CallStateRinging:
          bool? accept = await _showAcceptDialog();
          if (accept!) {
            _accept();
            setState(() {
              _inCalling = true;
            });
          } else {
            _reject();
          }
          break;
        case CallState.CallStateBye:
          if (_waitAccept) {
            print('peer reject');
            _waitAccept = false;
            Navigator.of(context).pop(false);
          }
          setState(() {
            _remoteRenderer.srcObject = null;
            _inCalling = false;
            _session = null;
          });
          break;
        case CallState.CallStateInvite:
          _waitAccept = true;
          _showInvateDialog();
          break;
        case CallState.CallStateConnected:
          if (_waitAccept) {
            _waitAccept = false;
            Navigator.of(context).pop(false);
          }
          setState(() {
            _inCalling = true;
          });

          break;
        case CallState.CallStateRinging:
      }
    };

    _signaling?.onPeersUpdate = ((event) {
      setState(() {
        _selfId = event['self'];
        _peers = event['peers'];
      });
    });

    _signaling?.onAddRemoteStream = ((_, stream) {
      _remoteRenderer.srcObject = stream;
      print(
          "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
      print(MediaQuery.of(context).size.width);
      print(MediaQuery.of(context).size.height);
      print(_remoteRenderer.textureId);
      print(
          "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
    });

    _signaling?.onRemoveRemoteStream = ((_, stream) {
      _remoteRenderer.srcObject = null;
    });
  }

  void updateServer() {
    if (globals.serverUp) {
      setState(() {
        globals.serverUp = true;
      });
    }
  }

  Future<bool?> _showAcceptDialog() {
    List _peerList = _peers.toList();
    String _peerName = _peers[1]['name'].toString();
    print(_peers);
    return showDialog<bool?>(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: Color.fromARGB(255, 22, 22, 22),
          title: Text("Parece que encontramos uma conexão de $_peerName",
              style: TextStyle(color: Color.fromARGB(227, 176, 217, 236))),
          content: Text("Deseja aceitar?",
              style: TextStyle(color: Color.fromARGB(227, 176, 217, 236))),
          actions: <Widget>[
            TextButton(
              child: Text("Rejeitar",
                  style: TextStyle(color: Color.fromARGB(227, 176, 217, 236))),
              onPressed: () => Navigator.of(context).pop(false),
            ),
            TextButton(
              child: Text("Aceitar",
                  style: TextStyle(color: Color.fromARGB(227, 176, 217, 236))),
              onPressed: () {
                Navigator.of(context).pop(true);

                String dir2 = (Directory.current.path + '\\lib\\teste.py');
                Process.run('python', [dir2], runInShell: true)
                    .then((ProcessResult result) {
                  print(result.stderr);
                });
              },
            ),
          ],
        );
      },
    );
  }

  Future<bool?> _showInvateDialog() {
    return showDialog<bool?>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text("title"),
          content: Text("waiting"),
          actions: <Widget>[
            TextButton(
              child: Text("cancel"),
              onPressed: () {
                Navigator.of(context).pop(false);
              },
            ),
          ],
        );
      },
    );
  }

  _invitePeer(BuildContext context, String peerId, bool useScreen) async {
    if (_signaling != null && peerId != _selfId) {
      _signaling?.invite(peerId, 'video', useScreen);
    }
  }

  _accept() {
    if (_session != null) {
      _signaling?.accept(_session!.sid);
    }
  }

  _reject() {
    if (_session != null) {
      _signaling?.reject(_session!.sid);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _inCalling
          ? OrientationBuilder(builder: (context, orientation) {
              return Container(
                child: Stack(children: <Widget>[
                  Positioned(
                    child: Container(
                      child: RTCVideoView(_remoteRenderer, mirror: false),
                      decoration: BoxDecoration(color: Colors.black54),
                    ),
                  ),
                ]),
              );
            })
          : Scaffold(
              body: SafeArea(
                child: Column(
                  children: [
                    Expanded(
                      flex: 3,
                      child: new Container(
                        color: Color.fromARGB(255, 22, 22, 22),
                        child: new Column(
                          children: <Widget>[
                            new Expanded(
                              flex: 2,
                              child: new Container(
                                alignment: Alignment.bottomCenter,
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: <Widget>[
                                    Container(
                                      width: 100,
                                      child: Icon(
                                          Icons.desktop_windows_outlined,
                                          size: 80.0,
                                          color: Color.fromARGB(
                                              227, 176, 217, 236)),
                                    ),
                                    Container(
                                      width: 100,
                                      child: Text('...',
                                          style: TextStyle(
                                              fontSize: 80.0,
                                              color: Color.fromARGB(
                                                  227, 176, 217, 236)),
                                          textAlign: TextAlign.center),
                                    ),
                                    Container(
                                      width: 100,
                                      child: Icon(Icons.phone_iphone_outlined,
                                          size: 80.0,
                                          color: Color.fromARGB(
                                              227, 176, 217, 236)),
                                    ),
                                  ],
                                ),
                              ),
                            ),
                            new Expanded(
                              flex: 1,
                              child: new Container(
                                alignment: Alignment(0, -0.75),
                                child: Text(
                                  'Aguardando conexão do dispositivo móvel',
                                  style: TextStyle(
                                      fontSize: 25.0,
                                      color:
                                          Color.fromARGB(227, 176, 217, 236)),
                                ),
                              ),
                            ),
                            new Expanded(
                              flex: 1,
                              child: Container(
                                child: RichText(
                                  text: TextSpan(
                                    style: TextStyle(
                                        fontSize: 15.0,
                                        color:
                                            Color.fromARGB(227, 176, 217, 236),
                                        fontWeight: FontWeight.bold),
                                    children: <InlineSpan>[
                                      TextSpan(text: 'Status do Servidor: '),
                                      TextSpan(
                                        text: globals.serverUp == true
                                            ? 'Online '
                                            : 'Offline',
                                        style: new TextStyle(
                                            color: globals.serverUp == true
                                                ? Colors.green
                                                : Colors.red),
                                      ),
                                      WidgetSpan(
                                        child: Icon(
                                            globals.serverUp == true
                                                ? Icons.wifi_tethering_outlined
                                                : Icons
                                                    .wifi_tethering_off_sharp,
                                            size: 20,
                                            color: globals.serverUp == true
                                                ? Colors.green
                                                : Colors.red),
                                      ),
                                      TextSpan(
                                          text: globals.serverUp == true
                                              ? '\nEndereço: ' +
                                                  globals.serverAdress
                                              : ''),
                                    ],
                                  ),
                                  textAlign: TextAlign.center,
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
}
