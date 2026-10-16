"""Tests for the daemon's crash dumps (speak_telegram_daemon.enable_crash_dumps).

A child interpreter enables the dumps, parks a second thread in a function with a
distinctive name, and segfaults. The dump must reach stderr with that thread's stack,
and the child must still die by SIGSEGV (compose sees exit 139 and restarts the daemon).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = textwrap.dedent("""
    import faulthandler, threading
    import speak_telegram_daemon as d

    parked = threading.Event()

    def parked_in_native_work():
        parked.set()
        threading.Event().wait()

    d.enable_crash_dumps()
    threading.Thread(target=parked_in_native_work, daemon=True).start()
    parked.wait()
    faulthandler._sigsegv()
""")


class TestCrashDumps(unittest.TestCase):
    def test_sigsegv_dumps_every_thread_and_still_dies_by_the_signal(self):
        child = subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, capture_output=True, text=True,
                               timeout=120)
        self.assertEqual(child.returncode, -signal.SIGSEGV, child.stderr)
        self.assertIn("Fatal Python error: Segmentation fault", child.stderr)
        self.assertIn("parked_in_native_work", child.stderr)   # all_threads: not only the crashing one
        self.assertIn("Current thread", child.stderr)


if __name__ == "__main__":
    unittest.main()
