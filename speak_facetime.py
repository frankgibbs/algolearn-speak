"""FaceTime Audio calls from the Mac to the owner's iPhone.

docs/DESIGN_FACETIME_CALL.md has the design. This module wraps the `ftcall`
helper (ftcall/main.swift), which reads and presses the FaceTime call banner
through the Accessibility API, and implements the dial / hang-up flows and
the quiet-hours rule. It never touches audio devices; speak_server routes
audio to the call devices while the state is "connected".
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import time

# SPEAK_FTCALL_BIN exists for tests only (they point it at tests/fake_ftcall), like SPEAK_AUDIO_DRY_RUN.
FTCALL_BIN = os.environ.get("SPEAK_FTCALL_BIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "ftcall", "ftcall")
STATES = ("none", "click_to_call", "ringing", "connected", "unknown")

BANNER_TIMEOUT = 15.0      # dial -> "Click to Call" banner
RING_START_TIMEOUT = 10.0  # Call pressed -> banner shows ringing (or connected)
ANSWER_TIMEOUT = 90.0      # ringing -> connected
HANGUP_TIMEOUT = 5.0       # End/Cancel pressed -> banner gone
UNKNOWN_GRACE = 1.0        # an unreadable banner (mid-rebuild) is tolerated this long before it is an error
GONE_CONFIRM = 1.0         # a vanished banner while ringing must stay gone this long to count as not answered

ANSWERED = "answered"
ALREADY_CONNECTED = "already connected"
NOT_ANSWERED = "not answered (declined or no answer)"
HUNG_UP = "hung up"
NO_ACTIVE_CALL = "no active call"

_NUMBER_RE = re.compile(r"^\+\d{8,15}$")
_QUIET_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2})$")


def validate_number(number: str) -> None:
    if not _NUMBER_RE.match(number):
        raise RuntimeError(f"SPEAK_CALL_NUMBER={number!r} is not an E.164 number (+ followed by 8-15 digits)")


def parse_quiet_hours(window: str) -> tuple[int, int]:
    """`HH:MM-HH:MM` -> (start, end) in minutes after midnight. May wrap midnight."""
    m = _QUIET_RE.match(window)
    if not m:
        raise RuntimeError(f"SPEAK_QUIET_HOURS={window!r} is not HH:MM-HH:MM")
    sh, sm, eh, em = (int(g) for g in m.groups())
    if sh > 23 or eh > 23 or sm > 59 or em > 59:
        raise RuntimeError(f"SPEAK_QUIET_HOURS={window!r} has an out-of-range time")
    start, end = sh * 60 + sm, eh * 60 + em
    if start == end:
        raise RuntimeError(f"SPEAK_QUIET_HOURS={window!r} is an empty window")
    return start, end


def in_quiet_hours(window: tuple[int, int], now: datetime.time) -> bool:
    start, end = window
    t = now.hour * 60 + now.minute
    if start < end:
        return start <= t < end
    return t >= start or t < end  # wraps midnight


class FaceTime:
    """The FaceTime call banner, read and driven through `ftcall`."""

    def __init__(self, binary: str = FTCALL_BIN, poll_seconds: float = 0.25) -> None:
        self.binary = binary
        self.poll_seconds = poll_seconds

    def check_binary(self) -> None:
        if not os.access(self.binary, os.X_OK):
            raise RuntimeError(f"ftcall helper not found at {self.binary}; build it with ftcall/build.sh")

    def _run(self, *args: str) -> str:
        proc = subprocess.run([self.binary, *args], capture_output=True, text=True, timeout=10.0)
        if proc.returncode != 0:
            raise RuntimeError(f"ftcall {' '.join(args)} failed (exit {proc.returncode}): {proc.stderr.strip()}")
        return proc.stdout

    def state_and_text(self) -> tuple[str, str]:
        out = json.loads(self._run("state"))
        state = out["state"]
        if state not in STATES:
            raise RuntimeError(f"ftcall state returned an unknown value {state!r}")
        return state, out["text"]

    def state(self) -> str:
        return self.state_and_text()[0]

    def settled_state(self) -> tuple[str, str]:
        """The state, re-read for up to UNKNOWN_GRACE while it is "unknown" (a
        banner caught mid-rebuild). A persistent "unknown" is returned as is."""
        deadline = time.monotonic() + UNKNOWN_GRACE
        while True:
            state, text = self.state_and_text()
            if state != "unknown" or time.monotonic() >= deadline:
                return state, text
            time.sleep(self.poll_seconds)

    def press(self, button: str) -> None:
        self._run("press", button)

    def devices(self) -> dict[str, str]:
        return json.loads(self._run("devices"))

    def dial(self, number: str) -> None:
        proc = subprocess.run(["open", f"facetime-audio://{number}"], capture_output=True, text=True, timeout=10.0)
        if proc.returncode != 0:
            raise RuntimeError(f"open facetime-audio:// failed (exit {proc.returncode}): {proc.stderr.strip()}")

    def _poll(self, accept: tuple[str, ...], keep: tuple[str, ...], timeout: float) -> tuple[str, str] | None:
        """Poll until the state is in `accept` (returned) or `timeout` passes
        (None). States in `keep` keep polling; "unknown" keeps polling for up
        to UNKNOWN_GRACE in a row; anything else raises with the banner text."""
        deadline = time.monotonic() + timeout
        unknown_since: float | None = None
        while True:
            state, text = self.state_and_text()
            now = time.monotonic()
            if state in accept:
                return state, text
            if state == "unknown":
                unknown_since = unknown_since if unknown_since is not None else now
                if now - unknown_since >= UNKNOWN_GRACE:
                    raise RuntimeError(f"unexpected FaceTime banner: {text!r}")
            elif state in keep:
                unknown_since = None
            else:
                raise RuntimeError(f"unexpected FaceTime banner state {state} ({text!r})")
            if now >= deadline:
                return None
            time.sleep(self.poll_seconds)

    def _press_end_and_wait(self, button: str) -> None:
        """Press End/Cancel and wait for the banner to go. A banner that is
        already gone when the press fails (the owner hung up first) is fine."""
        try:
            self.press(button)
        except RuntimeError:
            if self.settled_state()[0] == "none":
                return
            raise
        if self._poll(("none",), ("connected", "ringing", "click_to_call"), HANGUP_TIMEOUT) is None:
            state, text = self.state_and_text()
            raise RuntimeError(f"pressed {button} but the call banner did not go away within {HANGUP_TIMEOUT:.0f}s (state {state}: {text!r})")

    def place_call(self, number: str, mic_device: str, output_device: str) -> str:
        """Dial `number` and wait for the answer. `mic_device` / `output_device`
        are substrings FaceTime's checked Microphone / Output must contain.
        Returns ANSWERED, ALREADY_CONNECTED or NOT_ANSWERED; raises on anything
        else. A failure after dialing cancels a pending "Click to Call" banner
        before raising, so a later click can never dial."""
        state, text = self.settled_state()
        if state == "connected":
            return ALREADY_CONNECTED
        if state != "none":
            raise RuntimeError(f"cannot dial: a FaceTime call banner is already up (state {state}: {text!r})")

        self.dial(number)
        try:
            return self._after_dial(mic_device, output_device)
        except BaseException as e:
            try:
                if self.state() == "click_to_call":
                    self.press("Cancel")
            except RuntimeError as cleanup:
                raise RuntimeError(f"{e}; also failed to cancel a pending 'Click to Call' banner: {cleanup}") from e
            raise

    def _after_dial(self, mic_device: str, output_device: str) -> str:
        if self._poll(("click_to_call",), ("none",), BANNER_TIMEOUT) is None:
            raise RuntimeError(f"the FaceTime 'Click to Call' banner did not appear within {BANNER_TIMEOUT:.0f}s of dialing")

        devices = self.devices()
        if mic_device not in devices["microphone"] or output_device not in devices["output"]:
            raise RuntimeError(
                f"FaceTime's audio devices are wrong for a speak call: Microphone is {devices['microphone']!r} "
                f"(must be {mic_device!r}), Output is {devices['output']!r} (must be {output_device!r}). "
                f"Set them in FaceTime's Video menu. The call was cancelled."
            )

        self.press("Call")
        # A banner-less gap right after the press is not a decline: wait for ringing first.
        if self._poll(("ringing", "connected"), ("click_to_call", "none"), RING_START_TIMEOUT) is None:
            raise RuntimeError(f"the call did not start ringing within {RING_START_TIMEOUT:.0f}s of pressing Call")

        deadline = time.monotonic() + ANSWER_TIMEOUT
        while True:
            seen = self._poll(("connected", "none"), ("ringing",), max(0.0, deadline - time.monotonic()))
            if seen is None:
                self._press_end_and_wait("End")
                return NOT_ANSWERED
            if seen[0] == "connected":
                return ANSWERED
            # Banner gone: it must stay gone for GONE_CONFIRM before it counts as not answered.
            back = self._poll(("connected", "ringing"), ("none",), GONE_CONFIRM)
            if back is None:
                return NOT_ANSWERED
            if back[0] == "connected":
                return ANSWERED

    def hang_up(self) -> str:
        state, text = self.settled_state()
        if state == "none":
            return NO_ACTIVE_CALL
        if state in ("connected", "ringing"):
            self._press_end_and_wait("End")
        elif state == "click_to_call":
            self._press_end_and_wait("Cancel")
        else:
            raise RuntimeError(f"cannot hang up: unexpected FaceTime banner (state {state}: {text!r})")
        return HUNG_UP
