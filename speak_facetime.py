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
import logging
import os
import re
import subprocess
import threading
import time

log = logging.getLogger("speak")

# SPEAK_FTCALL_BIN exists for tests only (they point it at tests/fake_ftcall), like SPEAK_AUDIO_DRY_RUN.
FTCALL_BIN = os.environ.get("SPEAK_FTCALL_BIN") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "ftcall", "ftcall")
# "loading": a banner whose labels are not filled in yet (seen for ~1 s after dialing).
# "locked": the screen is locked. Verified on this Mac: locking ends a FaceTime call,
# and no call can be placed while locked, so "locked" always means "no call".
STATES = ("locked", "none", "loading", "click_to_call", "ringing", "connected", "unknown")

BANNER_TIMEOUT = 15.0      # dial -> "Click to Call" banner
RING_START_TIMEOUT = 10.0  # Call pressed -> banner shows ringing (or connected)
ANSWER_TIMEOUT = 90.0      # ringing -> connected
HANGUP_TIMEOUT = 5.0       # End/Cancel pressed -> banner gone
UNKNOWN_GRACE = 1.0        # an unrecognised banner is tolerated this long in a row before it is an error
SETTLE_TIMEOUT = 3.0       # how long settled_state waits out "loading" / "unknown"
GONE_CONFIRM = 1.0         # a vanished banner while ringing must stay gone this long to count as not answered

NOT_CALLED_LOCKED = "not called: the Mac is locked (FaceTime can only call while it is unlocked)"
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
        """The state, re-read for up to SETTLE_TIMEOUT while it is "loading" or
        "unknown" (a banner mid-redraw). A persistent one is returned as is."""
        deadline = time.monotonic() + SETTLE_TIMEOUT
        while True:
            state, text = self.state_and_text()
            if state not in ("loading", "unknown") or time.monotonic() >= deadline:
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
        to UNKNOWN_GRACE in a row; "loading" always keeps polling (bounded by
        `timeout`); anything else raises with the banner text."""
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
            elif state in keep or state == "loading":
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
        if self._poll(("none", "locked"), ("connected", "ringing", "click_to_call"), HANGUP_TIMEOUT) is None:
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
        if state == "locked":
            return NOT_CALLED_LOCKED
        if state != "none":
            raise RuntimeError(f"cannot dial: a FaceTime call banner is already up (state {state}: {text!r})")

        self.dial(number)
        try:
            return self._after_dial(mic_device, output_device)
        except BaseException as e:
            try:
                if self.settled_state()[0] == "click_to_call":
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
        if state in ("none", "locked"):
            return NO_ACTIVE_CALL
        if state in ("connected", "ringing"):
            self._press_end_and_wait("End")
        elif state == "click_to_call":
            self._press_end_and_wait("Cancel")
        else:
            raise RuntimeError(f"cannot hang up: unexpected FaceTime banner (state {state}: {text!r})")
        return HUNG_UP


class KeepAwake:
    """While a call is up, keep the Mac from sleeping or locking: a display
    lock or the screensaver lock ends a FaceTime call (verified on this Mac).

    Holds `caffeinate -d -i -w <this pid>` (no display or idle sleep; exits on
    its own if this process dies) and declares user activity every `pulse`
    seconds (`caffeinate -u -t 1`), which resets the screensaver's idle timer.
    A watcher thread polls the call state every `poll` seconds and stops
    everything once the call is gone. docs/DESIGN_FACETIME_CALL.md section 6.
    """

    def __init__(self, facetime: FaceTime, poll: float = 5.0, pulse: float = 60.0, caffeinate: str = "caffeinate") -> None:
        self.facetime = facetime
        self.poll = poll
        self.pulse = pulse
        self.caffeinate = caffeinate
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stop: threading.Event | None = None

    def active(self) -> bool:
        with self._lock:
            return self._proc is not None

    def start(self) -> None:
        with self._lock:
            if self._proc is not None:
                return
            self._proc = subprocess.Popen([self.caffeinate, "-d", "-i", "-w", str(os.getpid())],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._stop = threading.Event()
            threading.Thread(target=self._watch, args=(self._stop,), name="keep-awake", daemon=True).start()
        self._declare_activity()
        log.info("keep-awake on for the FaceTime call")

    def stop(self) -> None:
        with self._lock:
            proc, stop = self._proc, self._stop
            self._proc = self._stop = None
        if proc is None:
            return
        stop.set()
        proc.terminate()
        proc.wait(timeout=5.0)
        log.info("keep-awake off")

    def _declare_activity(self) -> None:
        subprocess.run([self.caffeinate, "-u", "-t", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10.0)

    def _watch(self, stop: threading.Event) -> None:
        last_pulse = time.monotonic()
        while not stop.wait(self.poll):
            try:
                state = self.facetime.state()
            except Exception as e:
                if stop.is_set():
                    return  # stopped while this read was in flight
                # Unreadable state: keep the Mac awake (the safe side) and say so.
                log.warning("keep-awake: could not read the call state (%s); staying awake", e)
                continue
            if state not in ("connected", "loading", "unknown"):
                log.info("keep-awake: call state is %s", state)
                self.stop()
                return
            if time.monotonic() - last_pulse >= self.pulse:
                self._declare_activity()
                last_pulse = time.monotonic()
