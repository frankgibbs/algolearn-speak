"""Tests for the speak-phone daemon/client split (docs/DESIGN_PHONE_AUDIO.md section 10).

Everything uses an ephemeral TCP port (port=0) and a temp Unix socket; the live
port 8772 is never bound. "Speak servers" are real separate processes that call
speak_server.main() and then route through it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.environ["SPEAK_AUDIO_DRY_RUN"] = "1"
os.environ["SPEAK_FTCALL_BIN"] = os.path.join(os.path.abspath(HERE), "fake_ftcall")  # never the real FaceTime banner; children inherit it
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import numpy as np

import speak_phone_audio as pa
import speak_phone_daemon as pd
from test_speak_phone_audio import FakePhone, StubVAD, speech_blocks, wait_for


def _tmp_socket(case: unittest.TestCase) -> str:
    d = tempfile.mkdtemp(prefix="spk-")
    case.addCleanup(shutil.rmtree, d, True)
    return os.path.join(d, "phone.sock")


def start_daemon(case: unittest.TestCase, sock: str, **kw) -> pd.PhoneDaemon:
    d = pd.PhoneDaemon(pa.PhoneAudioServer(host="127.0.0.1", port=0, vad_factory=StubVAD, **kw), sock)
    d.start()
    case.addCleanup(d.stop)
    return d


# Run inside a child process: a speak server instance. Starts via the real main()
# (mcp.run and engine loading stubbed out), then reports its routing decision and,
# if routed to the phone, plays a cue through the daemon.
CHILD = r"""
import json, os, sys
os.environ["SPEAK_AUDIO_DRY_RUN"] = "1"
sys.path.insert(0, %(root)r)
import numpy as np
import speak_server as s
s.engines.load = lambda: None
s.mcp.run = lambda *a, **k: None
s.main()
route = s._begin_call("speak")
if route == "phone":
    s._play_pcm(np.zeros(2400, dtype=np.float32), 24000)
    s._end_call("phone")
print(json.dumps({"route": route, "daemon": s.phone.daemon_reachable()}))
"""


def run_child(sock: str) -> subprocess.Popen:
    env = {**os.environ, "SPEAK_PHONE_SOCKET": sock}
    env.pop("SPEAK_PHONE_PORT", None)
    return subprocess.Popen([sys.executable, "-c", CHILD % {"root": ROOT}], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(p: subprocess.Popen) -> dict:
    out, err = p.communicate(timeout=60)
    assert p.returncode == 0, f"child failed ({p.returncode}):\n{err[-2000:]}"
    return json.loads(out.strip().splitlines()[-1])


class TestClientApi(unittest.TestCase):
    def setUp(self):
        self.sock = _tmp_socket(self)
        self.daemon = start_daemon(self, self.sock)
        self.client = pd.PhoneClient(self.sock)

    def test_connected_tracks_phone(self):
        self.assertTrue(self.client.daemon_reachable())
        self.assertFalse(self.client.connected())
        p = FakePhone(self.daemon.phone.port)
        self.addCleanup(p.ws.close)
        self.assertTrue(wait_for(self.client.connected))
        p.close()
        self.assertTrue(wait_for(lambda: not self.client.connected()))

    def test_play_segments_reach_phone_and_block_until_played(self):
        p = FakePhone(self.daemon.phone.port)
        self.addCleanup(p.ws.close)
        self.assertTrue(wait_for(self.client.connected))
        self.client.play_segments([np.zeros(2400, np.float32), np.zeros(1200, np.float32)], pa.SPK_RATE)
        self.assertEqual(p.types().count("play_start"), 2)
        self.assertEqual(p.types().count("play_end"), 2)
        self.assertEqual(sum(len(m) for m in p.received if isinstance(m, bytes)), (2400 + 1200) * 2)

    def test_record_returns_audio_and_timeout(self):
        p = FakePhone(self.daemon.phone.port, mic_blocks=speech_blocks(5, 20, 40))
        self.addCleanup(p.ws.close)
        self.assertTrue(wait_for(self.client.connected))
        audio = self.client.record(10, 0.3, 5, cue=np.zeros(2400, np.float32))
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertGreater(len(audio), 20 * 512)
        p.close()

    def test_record_timeout_is_reported(self):
        p = FakePhone(self.daemon.phone.port, mic_blocks=speech_blocks(60, 0, 0))
        self.addCleanup(p.ws.close)
        self.assertTrue(wait_for(self.client.connected))
        self.assertIs(self.client.record(5, 0.3, 1.0), pa.TIMEOUT)

    def test_mid_call_disconnect_raises_with_original_message(self):
        p = FakePhone(self.daemon.phone.port, ack=False)
        self.addCleanup(p.ws.close)
        self.assertTrue(wait_for(self.client.connected))
        import threading
        threading.Timer(0.3, p.ws.close).start()
        with self.assertRaisesRegex(RuntimeError, "phone disconnected during speak"):
            self.client.play(np.zeros(2400, np.float32), pa.SPK_RATE)

    def test_play_without_phone_raises(self):
        with self.assertRaisesRegex(RuntimeError, "phone is not connected"):
            self.client.play(np.zeros(240, np.float32), pa.SPK_RATE)


class TestNoDaemon(unittest.TestCase):
    def test_not_running_means_no_phone_and_ops_raise(self):
        c = pd.PhoneClient(_tmp_socket(self))
        self.assertFalse(c.daemon_reachable())
        self.assertFalse(c.connected())
        with self.assertRaises(FileNotFoundError):
            c.play(np.zeros(240, np.float32), pa.SPK_RATE)


class TestDaemonLifecycle(unittest.TestCase):
    def test_tcp_port_conflict_raises_and_leaves_socket_untouched(self):
        sock = _tmp_socket(self)
        first = start_daemon(self, sock)
        second = pd.PhoneDaemon(pa.PhoneAudioServer(host="127.0.0.1", port=first.phone.port, vad_factory=StubVAD), _tmp_socket(self))
        with self.assertRaisesRegex(RuntimeError, "failed to bind"):
            second.start()
        self.assertTrue(pd.PhoneClient(sock).daemon_reachable())


class TestTwoSpeakServers(unittest.TestCase):
    """(a) two speak servers start concurrently beside the daemon; (b) both route to a connected phone; (c) no daemon -> both Mac."""

    def test_a_b_two_servers_route_to_phone(self):
        sock = _tmp_socket(self)
        daemon = start_daemon(self, sock)
        # (a) both start concurrently while the daemon (holding its TCP port) runs
        children = [run_child(sock), run_child(sock)]
        results = [finish(c) for c in children]
        self.assertEqual([r["daemon"] for r in results], [True, True])
        self.assertEqual([r["route"] for r in results], ["mac", "mac"])  # no phone yet
        # (b) with a phone connected both route to it
        p = FakePhone(daemon.phone.port)
        self.addCleanup(p.ws.close)
        self.assertTrue(wait_for(daemon.phone.connected))
        children = [run_child(sock), run_child(sock)]
        results = [finish(c) for c in children]
        self.assertEqual([r["route"] for r in results], ["phone", "phone"])
        self.assertEqual(p.types().count("play_end"), 2)  # each child's cue reached the one phone

    def test_c_no_daemon_both_use_mac(self):
        sock = _tmp_socket(self)
        results = [finish(c) for c in (run_child(sock), run_child(sock))]
        self.assertEqual(results, [{"route": "mac", "daemon": False}] * 2)

    def test_server_never_binds_phone_port(self):
        with open(os.path.join(ROOT, "speak_server.py")) as f:
            src = f.read()
        self.assertNotIn("PhoneAudioServer", src)


if __name__ == "__main__":
    unittest.main()
