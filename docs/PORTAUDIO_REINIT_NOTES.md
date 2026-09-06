# PortAudio Device Re-initialization and Bluetooth Hotplug on macOS

**Memo** | Research question: In-process re-initialization as a path to pick up Bluetooth device changes after disconnection/reconnection

---

## (1) PortAudio Device List Lifecycle

**Finding: Device list is FROZEN after Pa_Initialize; refresh requires Pa_Terminate/Pa_Initialize cycle.**

Per PortAudio's official documentation on device enumeration and the DeepWiki Device Management reference:

> "The device information is considered static during the lifetime of an audio session (between Pa_Initialize() and Pa_Terminate())."
>
> "The pointers returned by Pa_GetDeviceInfo() are only guaranteed to be valid between calls to Pa_Initialize() and Pa_Terminate()."

**Refresh mechanism:** If audio hardware configuration changes (devices added/removed), the application must call `Pa_Terminate()` followed by `Pa_Initialize()` to enumerate the new device list.

**Hotplug support status on CoreAudio:** PortAudio's Wiki documents a long-standing hotplug branch with `Pa_RefreshDeviceList()` and `Pa_SetDevicesChangedCallback()`, but this remains incomplete for CoreAudio on macOS. The widely-known Evan Balster CoreAudio hotplug fork ([pa-coreaudio-hotplug](https://github.com/EvanBalster/pa-coreaudio-hotplug)) is marked **OUTDATED // DO NOT USE**. The official PortAudio main branch on macOS does **not** include hotplug device refresh—the standard documented path is a full Pa_Terminate/Pa_Initialize cycle.

**Key sources:**
- [DeepWiki: Device Management](https://deepwiki.com/PortAudio/portaudio/2.2-device-management)
- [PortAudio Wiki: HotPlug](https://github.com/PortAudio/portaudio/wiki/HotPlug)

---

## (2) Python-sounddevice Re-initialization: Private, Unsupported, Problematic

**Finding: `sd._terminate()` and `sd._initialize()` are private functions; in-process re-init is NOT documented as supported and carries known hazards.**

### API Status
The functions are listed under "Expert Mode" in the python-sounddevice documentation but carry no usage guidance, examples, or stability guarantees. The leading underscore prefix is the Python convention for private/internal APIs.

### Known Issues with In-Process Re-init

1. **Hanging on macOS** (Issue #394, Feb 2022):
   User reported that calling `sd._terminate()` and `sd._initialize()` hangs the program on macOS. No resolution was provided; the issue remains open.

2. **Device list does not refresh on its own** (Issue #516):
   Calling `sd._terminate()` + `sd._initialize()` alone does NOT refresh the device list. The PortAudio DLL must be manually closed and reopened via low-level FFI calls (`sd._ffi.dlclose()`, `sd._ffi.dlopen()`) between terminate and initialize for the device list to refresh. This workaround requires direct access to private module internals and is not a documented or supported pattern.

3. **No official maintainer endorsement**:
   The python-sounddevice maintainer has not documented or endorsed in-process re-initialization as a supported feature. Related feature requests (Issue #47, #3) asking for re-initialization capability have received no official response or path forward.

### Verdict
In-process re-initialization using private functions is a **fragile workaround**, not a supported API. It can hang, requires low-level FFI manipulation to work at all, and carries unknown interaction risks (e.g., calling while a stream exists, or from a non-main thread).

**Key sources:**
- [Issue #394: sounddevice reset hangs on macOS](https://github.com/spatialaudio/python-sounddevice/issues/394)
- [Issue #516: Device list doesn't refresh after terminate/initialize](https://github.com/spatialaudio/python-sounddevice/issues/516)
- python-sounddevice API docs (Expert Mode, no stability statement)

---

## (3) Error -9986 on macOS with Bluetooth/AirPods: Not Specifically Tied to Device Switching

**Finding: -9986 paInternalError on macOS is a documented CoreAudio issue, but literature does NOT identify Bluetooth device reconnection as a clear root cause or resolvable path.**

### What -9986 Represents
PortAudio error code -9986 is `paInternalError` in PortAudio's enum. It indicates an unspecified CoreAudio failure on macOS.

### Reported Occurrences
- Audacity #3227: Users cannot play audio through AirPods; wired headphones and built-in speakers work. Issue marked as a "dependency" problem (PortAudio limitation), no resolution provided.
- Audacity forum: "Rescan Audio Devices" (which re-enumerates but does not re-initialize PortAudio) sometimes resolves -9986 for external USB audio devices.
- python-sounddevice #454: User reports -9986 when opening an InputStream; device list includes MacBook Pro microphone and virtual devices; no clear cause or solution documented.

### Bluetooth-Specific Path
None of the primary sources (GitHub issues, Audacity forums, python-sounddevice docs) document -9986 as specifically triggered by Bluetooth device **reconnection during idle**. The issue appears tied to the device itself (AirPods in particular) being selected, not to the device switching away and back.

### No Documented Recovery Path
Audacity's workaround of "Rescan Audio Devices" (a UI action that re-enumerates without re-init) is not a complete solution. There is no published advice to call Pa_Terminate/Pa_Initialize as a fix for -9986 in the presence of Bluetooth device reconnection.

**Key sources:**
- [Audacity #3227: AirPods error](https://github.com/audacity/audacity/issues/3227)
- [Audacity forum: -9986 solutions](https://forum.audacityteam.org/t/error-code-9986-internal-portaudio-error/64536)
- [python-sounddevice #454: -9986 on input stream](https://github.com/spatialaudio/python-sounddevice/issues/454)

---

## Verdict: In-Process Re-init is NOT Documented or Safe

**Three-line summary:**

1. **PortAudio freezes its device list at Pa_Initialize and does NOT refresh it—the only documented path to pick up new devices is Pa_Terminate() followed by Pa_Initialize() (full process restart).**
2. **python-sounddevice's `_terminate()` and `_initialize()` are private, unsupported functions; in-process calls hang on macOS and do not refresh the device list without additional low-level FFI manipulation.**
3. **Bluetooth device reconnection during idle is NOT documented as a trigger for -9986; there is no published path to recover in-process, and a fresh process is the only documented-safe approach.**

### Recommended Path for algolearn-speak

- Do not attempt in-process re-initialization with `sd._terminate()/_initialize()`.
- Design the audio thread to tolerate a hard restart (new Python process) when the Bluetooth device state changes.
- Consider monitoring device availability at a higher level (e.g., via system notifications or a watchdog thread) and terminating the audio process when a disconnection is detected, allowing the supervisor to restart fresh.
- If -9986 occurs at stream-open time, log the error and request a full process restart rather than attempting re-initialization.

---

**Prepared:** 2026-09-06 | Research completed via primary sources (PortAudio docs, github.com/PortAudio, github.com/spatialaudio/python-sounddevice, Audacity forums)
