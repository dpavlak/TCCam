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
  RTCVideoRenderer _localRenderer = RTCVideoRenderer();
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
    await _localRenderer.initialize();
  }

  @override
  deactivate() {
    super.deactivate();
    _signaling?.close();
    _localRenderer.dispose();
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
            _localRenderer.srcObject = null;
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

    _signaling?.onLocalStream = ((stream) {
      _localRenderer.srcObject = stream;
    });
  }

  Future<bool?> _showAcceptDialog() {
    return showDialog<bool?>(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: Text("title"),
          content: Text("accept?"),
          actions: <Widget>[
            TextButton(
              child: Text("reject"),
              onPressed: () => Navigator.of(context).pop(false),
            ),
            TextButton(
              child: Text("accept"),
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
          content: Text("Aguardando confirmação",
              style: TextStyle(color: Color.fromARGB(227, 176, 217, 236))),
          backgroundColor: Color.fromARGB(255, 22, 22, 22),
          actions: <Widget>[
            TextButton(
              child: Text("cancelar",
                  style: TextStyle(color: Color.fromARGB(227, 176, 217, 236))),
              onPressed: () {
                Navigator.of(context).pop(false);
                _hangUp();
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

  _hangUp() {
    if (_session != null) {
      _signaling?.bye(_session!.sid);
    }
  }

  _switchCamera() {
    _signaling?.switchCamera();
  }

  _muteMic() {
    _signaling?.muteMic();
  }

  /*  _buildRow(context, peer) {
    return ListBody(children: <Widget>[
      ListTile(
        title: Text(peer['name'] + ', ID: ${peer['id']} ',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 20.0, color: Color.fromARGB(227, 176, 217, 236))),
        onTap: null,
        trailing: SizedBox(
          width: 100.0,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: <Widget>[
              IconButton(
                icon: Icon(Icons.videocam, color: Color.fromARGB(227, 176, 217, 236)),
                onPressed: () => _invitePeer(context, peer['id'], false),
                tooltip: 'Video calling',
              ),
            ],
          ),
        ),
      )
    ]);
  } */

  _buildRow(context, peer) {
    return Center(
      child: new Column(
        children: [
          new Container(
            height: 300,
            alignment: Alignment.bottomCenter,
            child: new Text(
              '''Existe um aparelho aguardando conexão! \n ''' +
                  '''Nome: ${peer['name']} \n''' +
                  '''ID: ${peer['id']} ''',
              textAlign: TextAlign.center,
              style: TextStyle(
                  fontSize: 16.0,
                  letterSpacing: 2.0,
                  height: 2,
                  color: Color.fromARGB(227, 176, 217, 236)),
            ),
          ),
          new Container(
            height: 200,
            child: new SizedBox(
              height: 100.0,
              width: 100.0,
              child: IconButton(
                padding: EdgeInsets.zero,
                icon: Icon(Icons.videocam,
                    color: Color.fromARGB(227, 176, 217, 236), size: 80.0),
                onPressed: () => _invitePeer(context, peer['id'], false),
                tooltip: 'Video call',
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    print(_peers);
    _peers.removeWhere((item) => item['id'] == _selfId);
    print(_peers);
    return Scaffold(
      appBar: _inCalling
          ? null
          : AppBar(
              leading: const BackButton(
                color: Color.fromARGB(227, 176, 217, 236),
              ),
              centerTitle: true,
              title: Text('Dispositivos disponiveis',
                  style: TextStyle(color: Color.fromARGB(227, 176, 217, 236))),
              backgroundColor: Color.fromARGB(255, 22, 22, 22),
            ),
      backgroundColor: Color.fromARGB(255, 22, 22, 22),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
      floatingActionButton: _inCalling
          ? SizedBox(
              width: 200.0,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: <Widget>[
                  FloatingActionButton(
                    child: const Icon(Icons.switch_camera),
                    backgroundColor: Color.fromARGB(227, 176, 217, 236),
                    onPressed: _switchCamera,
                  ),
                  FloatingActionButton(
                    onPressed: _hangUp,
                    tooltip: 'Hangup',
                    child: Icon(Icons.call_end),
                    backgroundColor: Colors.red,
                  ), /* 
                    FloatingActionButton(
                      child: const Icon(Icons.mic_off),
                      backgroundColor: Color.fromARGB(227, 176, 217, 236),
                      onPressed: _muteMic,
                    ) */
                ],
              ),
            )
          : null,
      body: _inCalling
          ? OrientationBuilder(
              builder: (context, orientation) {
                return Container(
                  child: Stack(
                    children: <Widget>[
                      Positioned(
                        left: 0.0,
                        right: 0.0,
                        top: 0.0,
                        bottom: 0.0,
                        child: Container(
                          margin: EdgeInsets.fromLTRB(0.0, 0.0, 0.0, 0.0),
                          width: MediaQuery.of(context).size.width,
                          height: MediaQuery.of(context).size.height,
                          child: RTCVideoView(_localRenderer),
                          decoration: BoxDecoration(color: Colors.black54),
                        ),
                      ),
                    ],
                  ),
                );
              },
            )
          : ListView.builder(
              shrinkWrap: true,
              padding: const EdgeInsets.all(0.0),
              itemCount: (_peers != null ? _peers.length : 0),
              itemBuilder: (context, i) {
                return _buildRow(context, _peers[i]);
              },
            ),
    );
  }
}
