import 'dart:convert';

import 'package:flutter/material.dart';
import 'dart:core';
import 'signaling.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';

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

  bool _waitAccept = false;

  // ignore: unused_element
  _CallSampleState();

  @override
  initState() {
    super.initState();
    initRenderers();
    _connect();
  }

  initRenderers() async {
    await _remoteRenderer.initialize();
  }

  @override
  deactivate() {
    super.deactivate();
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
    });

    _signaling?.onRemoveRemoteStream = ((_, stream) {
      _remoteRenderer.srcObject = null;
    });
  }

  Future<bool?> _showAcceptDialog() {
    String _peerName = _peers[1]['name'].toString();
    return showDialog<bool?>(
      context: context,
      builder: (context) {
        return AlertDialog(
          backgroundColor: Color.fromARGB(255, 22, 22, 22),
          title: Text("Parece que encontramos uma conexão de $_peerName",
              style: TextStyle(color: Colors.deepPurple)),
          content: Text("Deseja aceitar?",
              style: TextStyle(color: Colors.deepPurple)),
          actions: <Widget>[
            TextButton(
              child:
                  Text("Rejeitar", style: TextStyle(color: Colors.deepPurple)),
              onPressed: () => Navigator.of(context).pop(false),
            ),
            TextButton(
              child:
                  Text("Aceitar", style: TextStyle(color: Colors.deepPurple)),
              onPressed: () {
                Navigator.of(context).pop(true);
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
                      left: 0.0,
                      right: 0.0,
                      top: 0.0,
                      bottom: 0.0,
                      child: Container(
                        margin: EdgeInsets.fromLTRB(0.0, 0.0, 0.0, 0.0),
                        width: MediaQuery.of(context).size.width,
                        height: MediaQuery.of(context).size.height,
                        child: RTCVideoView(_remoteRenderer),
                        decoration: BoxDecoration(color: Colors.black54),
                      )),
                ]),
              );
            })
          : Scaffold(
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
                                alignment: Alignment.bottomCenter,
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: <Widget>[
                                    Container(
                                      width: 100,
                                      child: Icon(
                                          Icons.desktop_windows_outlined,
                                          size: 80.0,
                                          color: Colors.deepPurple),
                                    ),
                                    Container(
                                      width: 100,
                                      child: Text('...',
                                          style: TextStyle(
                                              fontSize: 80.0,
                                              color: Colors.deepPurple),
                                          textAlign: TextAlign.center),
                                    ),
                                    Container(
                                      width: 100,
                                      child: Icon(Icons.phone_iphone_outlined,
                                          size: 80.0, color: Colors.deepPurple),
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
                                      fontSize: 25.0, color: Colors.deepPurple),
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
