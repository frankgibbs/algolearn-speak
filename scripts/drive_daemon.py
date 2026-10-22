"""Drive the speak daemon over TCP without MCP: the same request code the thin client's
tools run (speak_server's sync bodies), so a new daemon can be tried without an MCP reconnect.

Steps run in order in one process, so the session's call expectation and identity (session
id, project = the working folder's name) carry from one step to the next, as they do in a
Claude Code session. Each step prints its result or its error;
the first error stops the run.

    .venv/bin/python scripts/drive_daemon.py --daemon 192.168.86.31:8765 status
    .venv/bin/python scripts/drive_daemon.py --daemon 192.168.86.31:8765 \\
        persona call "Hi Frank, it's {persona}, testing the new speak daemon on patrick. Can you hear me clearly?" \\
        speak "This sentence came from the GPU." listen converse "Anything else?" hang_up

In a text argument, {persona} is replaced with the persona name from the latest reply that
carried this project's persona (run the persona step first); used before any such reply,
the step raises.

Steps: status | ping | debug [N] | set_debug on|off | call GREETING | call! GREETING (overrides
quiet hours; only when the owner said they are up) | speak TEXT | listen | converse TEXT | hang_up |
persona | new_persona | set_persona JSON (e.g. '{"speed": 1.1}' or '{"name": "Nova"}') |
add_to_personality NOTE
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import speak_protocol as proto  # noqa: E402
import speak_server as client  # noqa: E402

LISTEN = (120.0, 2.0, 45.0)   # the tools' defaults: max_seconds, silence_seconds, start_timeout_seconds
TAKES_ARG = {"call", "call!", "speak", "converse", "set_debug", "set_persona", "add_to_personality"}
SET_PERSONA_FIELDS = {"name", "voice", "speed", "personality"}
PERSONA_PLACEHOLDER = "{persona}"

# The persona name from the latest reply that carried this project's persona block, or None.
_persona_name: str | None = None


def fill(arg: str | None) -> str | None:
    """`arg` with PERSONA_PLACEHOLDER replaced by the name the daemon reported."""
    if arg is None or PERSONA_PLACEHOLDER not in arg:
        return arg
    if _persona_name is None:
        raise ValueError(f"{PERSONA_PLACEHOLDER} used before any reply named this project's persona; "
                         "run the persona step first")
    return arg.replace(PERSONA_PLACEHOLDER, _persona_name)


def note_persona(result) -> None:
    global _persona_name
    if isinstance(result, dict) and proto.REPLY_PERSONA in result:
        _persona_name = result[proto.REPLY_PERSONA]["name"]


def run(step: str, arg: str | None):
    if step == "status":
        return client._status_sync()
    if step == "ping":
        return client._request(proto.OP_PING, {}, client.QUICK_REPLY_TIMEOUT_SECONDS)
    if step == "debug":
        return client._call_debug_sync(int(arg) if arg else 50)
    if step == "set_debug":
        if arg not in ("on", "off"):
            raise ValueError(f"set_debug takes on or off, got {arg!r}")
        return client._set_call_debug_sync(arg == "on")
    if step in ("call", "call!"):
        return client._call_sync(arg, step == "call!", *LISTEN)
    if step == "speak":
        return client._speak_sync(arg)
    if step == "listen":
        return client._listen_sync(*LISTEN)
    if step == "converse":
        return client._converse_sync(arg, *LISTEN)
    if step == "hang_up":
        return client._hang_up_sync()
    if step == "persona":
        return client._persona_sync()
    if step == "new_persona":
        return client._new_persona_sync()
    if step == "set_persona":
        fields = json.loads(arg)
        if not isinstance(fields, dict) or not set(fields) <= SET_PERSONA_FIELDS:
            raise ValueError(f"set_persona takes a JSON object with keys from {sorted(SET_PERSONA_FIELDS)}, got {arg!r}")
        return client._set_persona_sync(fields.get("name"), fields.get("voice"), fields.get("speed"),
                                        fields.get("personality"))
    if step == "add_to_personality":
        return client._add_to_personality_sync(arg)
    raise ValueError(f"unknown step {step!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--daemon", required=True, help="host:port of the speak daemon, e.g. 192.168.86.31:8765")
    parser.add_argument("steps", nargs="+")
    args = parser.parse_args()
    client._daemon = proto.parse_address(args.daemon)
    client.init_identity(os.getcwd())   # as the MCP client does: the project is the working folder
    steps, i = [], 0
    while i < len(args.steps):
        step = args.steps[i]
        if step in TAKES_ARG:
            if i + 1 >= len(args.steps):
                parser.error(f"{step} needs an argument")
            steps.append((step, args.steps[i + 1]))
            i += 2
        else:
            arg = args.steps[i + 1] if step == "debug" and i + 1 < len(args.steps) and args.steps[i + 1].isdigit() else None
            steps.append((step, arg))
            i += 2 if arg else 1
    for step, arg in steps:
        t0 = time.monotonic()
        try:
            result = run(step, fill(arg))
            note_persona(result)
        except Exception as e:
            print(f"[{step}] {type(e).__name__}: {e}  ({time.monotonic() - t0:.1f}s, expect={client._call_generation}, persona_seen={client._persona_seen})")
            return 1
        shown = json.dumps(result, indent=2) if isinstance(result, dict) else repr(result)
        print(f"[{step}] {shown}  ({time.monotonic() - t0:.1f}s, expect={client._call_generation}, "
              f"persona_seen={client._persona_seen})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
