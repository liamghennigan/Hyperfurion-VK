"""Windows microphone capture through WASAPI.

Shared mode is the path that opens. It stays inside the Windows audio
engine, asks for the low-latency period, and lets WASAPI convert to the
recognizer's sample rate. Exclusive mode locks the endpoint; if that
device disappears mid-stream (a USB remount does this), the open fails
and PortAudio has crashed on a dead exclusive stream. Exclusive is
opt-in (`exclusive` or `auto`) and, when the open fails, capture
continues in shared mode. The fallback reason is reported. It is not a
guessed latency.

Linux and macOS do not use this module to open a device. They keep the
PyAudio path in `audio_capture.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

SHARED = "shared"
EXCLUSIVE = "exclusive"
AUTO = "auto"
WASAPI_MODES = (SHARED, EXCLUSIVE, AUTO)

# The cross-platform config default. On Windows it is a tenth of a second
# of onset slop the WASAPI engine does not need, so that one value opens
# at WINDOWS_CHUNK_MS. Any other configured period is kept (floored).
STOCK_CHUNK_MS = 100
WINDOWS_CHUNK_MS = 20
MIN_CHUNK_MS = 10


class WasapiUnavailable(RuntimeError):
    """No WASAPI host API to open. The caller may use PortAudio instead."""


class InputDeviceError(RuntimeError):
    """The configured input is missing or ambiguous. Do not open another device."""


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    host_api: str
    channels: int
    default_rate: int
    is_default: bool = False


@dataclass(frozen=True)
class Selection:
    device: Optional[InputDevice]
    status: str  # "ok", "none", "ambiguous", "no-inputs"
    names: tuple[str, ...] = ()


@dataclass
class OpenedCapture:
    reader: object
    info: dict
    chunk_ms: int
    frames: int


def normalize_mode(value: object) -> str:
    text = str(value or "").strip().casefold()
    if text in WASAPI_MODES:
        return text
    return SHARED


def effective_chunk_ms(configured: object) -> int:
    """Capture period, in milliseconds, for a Windows open.

    `100` is the stock cross-platform default and opens at 20 ms. Any
    other positive value is the period, and values under 10 ms are raised
    to 10 ms. A wake-word capture that asks for 80 ms stays at 80 ms.
    """
    if isinstance(configured, bool):
        return WINDOWS_CHUNK_MS
    try:
        value = int(configured)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return WINDOWS_CHUNK_MS
    if value == STOCK_CHUNK_MS:
        return WINDOWS_CHUNK_MS
    if value < MIN_CHUNK_MS:
        return MIN_CHUNK_MS
    return value


def prepare_windows_period(sample_rate: int, chunk_ms: object) -> tuple[int, int]:
    period = effective_chunk_ms(chunk_ms)
    try:
        rate = int(sample_rate)
    except (TypeError, ValueError):
        rate = 0
    frames = int(rate * period / 1000) if rate > 0 else 0
    return period, max(1, frames)


def open_attempts(mode: object) -> tuple[str, ...]:
    """Shared never tries exclusive. Exclusive and auto try exclusive, then shared."""
    if normalize_mode(mode) == SHARED:
        return (SHARED,)
    return (EXCLUSIVE, SHARED)


def _is_wasapi(host_api: str) -> bool:
    return "wasapi" in (host_api or "").casefold()


def select_input(devices: Sequence[InputDevice], device_name: str) -> Selection:
    """Pick the input capture will open.

    When a WASAPI input exists, only WASAPI devices are eligible, so a
    name that also appears under MME or DirectSound still opens the
    WASAPI endpoint. `default` prefers the flagged default inside that
    pool, then the WASAPI device whose name matches the system default,
    then the first WASAPI input.
    """
    inputs = [device for device in devices if device.channels > 0]
    if not inputs:
        return Selection(None, "no-inputs")
    wasapi = [device for device in inputs if _is_wasapi(device.host_api)]
    pool = wasapi or inputs
    wanted = (device_name or "").strip()
    if not wanted or wanted.casefold() == "default":
        chosen = _default_device(pool, inputs)
        if chosen is None:
            return Selection(None, "none")
        return Selection(chosen, "ok")
    matches = _match_name(pool, wanted)
    if len(matches) == 1:
        return Selection(matches[0], "ok")
    # A name that hits more than one input stays ambiguous, even when one
    # of them is the system default. Picking the default would hide the
    # collision from doctor and from the open.
    if len(matches) > 1:
        return Selection(None, "ambiguous", tuple(device.name for device in matches))
    return Selection(None, "none")


def _default_device(pool: Sequence[InputDevice], inputs: Sequence[InputDevice]) -> Optional[InputDevice]:
    flagged = [device for device in pool if device.is_default]
    if flagged:
        return flagged[0]
    system = next((device for device in inputs if device.is_default), None)
    if system is not None and pool is not inputs:
        same = [device for device in pool if device.name.casefold() == system.name.casefold()]
        if len(same) == 1:
            return same[0]
        loose = _match_name(pool, system.name)
        if len(loose) == 1:
            return loose[0]
    return pool[0] if pool else None


def _match_name(pool: Sequence[InputDevice], wanted: str) -> list[InputDevice]:
    needle = wanted.casefold()
    exact = [device for device in pool if device.name.casefold() == needle]
    if exact:
        return exact
    return [device for device in pool if needle in device.name.casefold()]


def device_error(selection: Selection, device_name: str) -> InputDeviceError:
    if selection.status == "ambiguous":
        shown = ", ".join(selection.names[:4])
        return InputDeviceError(
            f'Input device "{device_name}" matches {len(selection.names)} devices ({shown})'
        )
    if (device_name or "").strip().casefold() in ("", "default"):
        return InputDeviceError(
            "No microphone found — connect one or set a default"
            " input device in your sound settings"
        )
    return InputDeviceError(f"Input device not found: {device_name}")


def chunk_ms_for_stream(capture: object, configured: object) -> float:
    """The period a captured chunk actually spans.

    The silence gate counts this, so a Windows open at 20 ms must not be
    timed as the configured 100 ms. A source with no `chunk_ms` (the
    phone mic) keeps the configured value.
    """
    actual = getattr(capture, "chunk_ms", None)
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual > 0:
        return float(actual)
    if isinstance(configured, bool):
        return float(STOCK_CHUNK_MS)
    try:
        value = float(configured)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(STOCK_CHUNK_MS)
    if value > 0:
        return value
    return float(STOCK_CHUNK_MS)


def stream_latency_ms(stream: object) -> Optional[float]:
    """PortAudio's latency for an open input stream, in milliseconds.

    `None` when the stream does not report one. Zero and NaN are not
    reported: they are not a measurement. The buffer period is separate
    and is not added in.

    sounddevice exposes `latency` (seconds, or an input/output pair).
    PyAudio exposes `get_input_latency()`. Either one is used. A missing
    method is an unreported latency, not a zero.
    """
    value = _reported_latency(stream)
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        value = value[0]
    try:
        ms = float(value) * 1000.0
    except (TypeError, ValueError):
        return None
    if ms != ms or ms <= 0.0:  # NaN or nothing reported
        return None
    return round(ms, 1)


def format_capture(info: object) -> str:
    """One line for `voice-keyboard stats` and doctor. Empty when there is nothing to say."""
    if not isinstance(info, dict) or not info:
        return ""
    host = str(info.get("host_api") or "")
    mode = str(info.get("mode") or "")
    bits: list[str] = []
    label = " ".join(part for part in (host, mode) if part)
    if label:
        bits.append(label)
    chunk = info.get("chunk_ms")
    if isinstance(chunk, (int, float)) and not isinstance(chunk, bool):
        bits.append(f"{int(chunk)} ms buffer")
    latency = info.get("stream_latency_ms")
    if isinstance(latency, (int, float)) and not isinstance(latency, bool):
        bits.append(f"stream latency {float(latency):.1f} ms")
    text = ", ".join(bits)
    device = str(info.get("device") or "")
    if device:
        text = f"{text} — {device}" if text else device
    fallback = str(info.get("fallback") or "").strip()
    if fallback:
        text = f"{text}; {fallback}" if text else fallback
    return text


def planned_capture_line(config: object) -> str:
    """What a Windows open will do, from config alone. No stream latency."""
    audio = {}
    if isinstance(config, dict):
        raw = config.get("audio") or {}
        if isinstance(raw, dict):
            audio = raw
    configured = audio.get("chunk_ms", STOCK_CHUNK_MS)
    try:
        configured_ms = int(configured)
    except (TypeError, ValueError):
        configured_ms = STOCK_CHUNK_MS
    if isinstance(configured, bool):
        configured_ms = STOCK_CHUNK_MS
    period = effective_chunk_ms(configured_ms)
    mode = normalize_mode(audio.get("wasapi", SHARED))
    device = str(audio.get("device_name") or "default").strip() or "default"
    line = f"WASAPI {mode}, {period} ms buffer, device {device}"
    if period != configured_ms:
        line += f" (chunk_ms {configured_ms} opens at {period} ms)"
    return line


def _reported_latency(stream: object):
    try:
        value = getattr(stream, "latency", None)
    except Exception:
        value = None
    if value is not None:
        return value
    getter = getattr(stream, "get_input_latency", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:
        return None


def _short_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if not text:
        text = exc.__class__.__name__
    if len(text) > 180:
        text = text[:177] + "..."
    return text


class WasapiReader:
    """The read/stop/close surface `AudioCapture` already uses for PyAudio."""

    def __init__(self, stream: object, frames: int):
        self._stream = stream
        self._frames = frames

    def read(self, _n: int, exception_on_overflow: bool = False) -> bytes:
        del exception_on_overflow  # WASAPI reports overflow as a flag, not an exception
        result = self._stream.read(self._frames)  # type: ignore[attr-defined]
        overflowed = False
        if isinstance(result, tuple):
            data = result[0]
            if len(result) > 1:
                overflowed = bool(result[1])
        else:
            data = result
        if overflowed:
            logger.debug("WASAPI input overflow")
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            raw = memoryview(data).tobytes()
        width = self._frames * 2  # int16 mono
        if len(raw) > width:
            raw = raw[:width]
        return raw

    def stop_stream(self) -> None:
        stop = getattr(self._stream, "stop", None)
        if stop is not None:
            stop()

    def close(self) -> None:
        close = getattr(self._stream, "close", None)
        if close is not None:
            close()


class SoundDeviceBackend:
    """The real WASAPI open. Tests pass a fake module instead of sounddevice."""

    def __init__(self, sd: object = None):
        if sd is None:
            try:
                import sounddevice as sd  # type: ignore[no-redef]
            except Exception as exc:
                raise WasapiUnavailable(f"sounddevice is not available ({exc})") from exc
        self._sd = sd

    def list_inputs(self) -> list[InputDevice]:
        try:
            return self._list_inputs()
        except WasapiUnavailable:
            raise
        except Exception as exc:
            raise WasapiUnavailable(f"could not query WASAPI devices ({exc})") from exc

    def _list_inputs(self) -> list[InputDevice]:
        sd = self._sd
        hostapis = sd.query_hostapis()
        if isinstance(hostapis, dict):
            hostapis = [hostapis]
        names: list[str] = []
        wasapi_default: Optional[int] = None
        for api in hostapis:
            name = str(api.get("name", ""))
            names.append(name)
            if _is_wasapi(name) and wasapi_default is None:
                wasapi_default = _as_int(api.get("default_input_device"), -1)
        system_default = _default_input_index(sd)
        listed = sd.query_devices()
        if isinstance(listed, dict):
            listed = [listed]
        devices: list[InputDevice] = []
        for position, info in enumerate(listed):
            mapping = _as_mapping(info)
            channels = _as_int(mapping.get("max_input_channels", mapping.get("maxInputChannels")), 0)
            if channels <= 0:
                continue
            host_index = _as_int(mapping.get("hostapi", mapping.get("hostApi")), -1)
            host = names[host_index] if 0 <= host_index < len(names) else ""
            index = _as_int(mapping.get("index"), position)
            is_default = False
            if _is_wasapi(host) and wasapi_default is not None and wasapi_default >= 0:
                is_default = index == wasapi_default
            elif not _is_wasapi(host):
                is_default = index == system_default
            rate = _as_int(mapping.get("default_samplerate", mapping.get("defaultSampleRate")), 0)
            devices.append(InputDevice(
                index=index,
                name=str(mapping.get("name") or ""),
                host_api=host,
                channels=channels,
                default_rate=rate,
                is_default=is_default,
            ))
        return devices

    def open_stream(self, *, device_index: int, sample_rate: int, frames: int, exclusive: bool):
        sd = self._sd
        # Shared mode has to convert a 48 kHz endpoint down to the
        # recognizer rate. auto_convert does that. It has no effect in
        # exclusive mode, which must already match the hardware format
        # and so usually rejects 16 kHz — that rejection is the fallback.
        settings = sd.WasapiSettings(exclusive=bool(exclusive), auto_convert=not exclusive)
        stream = sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=frames,
            device=device_index,
            channels=1,
            dtype="int16",
            latency="low",
            extra_settings=settings,
        )
        try:
            stream.start()
        except Exception:
            try:
                stream.close()
            except Exception:
                logger.debug("closing a WASAPI stream that failed to start", exc_info=True)
            raise
        return stream


def open_capture(
    *,
    device_name: str,
    sample_rate: int,
    chunk_ms: int,
    mode: str,
    backend: Optional[SoundDeviceBackend] = None,
) -> OpenedCapture:
    """Open a WASAPI input. Exclusive falls back to shared. Never invents a latency."""
    backend = backend or SoundDeviceBackend()
    devices = backend.list_inputs()
    if not any(_is_wasapi(device.host_api) for device in devices):
        raise WasapiUnavailable("no WASAPI input host API")
    selection = select_input(devices, device_name)
    if selection.device is None:
        raise device_error(selection, device_name)
    device = selection.device
    period, frames = prepare_windows_period(sample_rate, chunk_ms)
    errors: list[tuple[str, BaseException]] = []
    stream = None
    opened_mode = SHARED
    for attempt in open_attempts(mode):
        try:
            stream = backend.open_stream(
                device_index=device.index,
                sample_rate=sample_rate,
                frames=frames,
                exclusive=(attempt == EXCLUSIVE),
            )
        except Exception as exc:
            logger.warning("WASAPI %s open failed: %s", attempt, exc)
            errors.append((attempt, exc))
            continue
        opened_mode = attempt
        break
    if stream is None:
        last = errors[-1][1] if errors else RuntimeError("WASAPI open failed")
        raise RuntimeError(f"WASAPI open failed: {_short_error(last)}") from last
    info = {
        "backend": "wasapi",
        "host_api": device.host_api or "Windows WASAPI",
        "mode": opened_mode,
        "device": device.name,
        "sample_rate": int(sample_rate),
        "chunk_ms": period,
        "frames": frames,
    }
    fallback = _fallback_text(mode, opened_mode, errors)
    if fallback:
        info["fallback"] = fallback
    latency = stream_latency_ms(stream)
    if latency is not None:
        info["stream_latency_ms"] = latency
    logger.info(
        "WASAPI capture: mode=%s device=%s rate=%s chunk_ms=%s frames=%s stream_latency_ms=%s",
        opened_mode,
        device.name,
        sample_rate,
        period,
        frames,
        latency if latency is not None else "unreported",
    )
    return OpenedCapture(
        reader=WasapiReader(stream, frames),
        info=info,
        chunk_ms=period,
        frames=frames,
    )


def _fallback_text(requested: str, opened: str, errors: Sequence[tuple[str, BaseException]]) -> str:
    if opened != SHARED or normalize_mode(requested) == SHARED:
        return ""
    exclusive = [exc for attempt, exc in errors if attempt == EXCLUSIVE]
    if not exclusive:
        return ""
    return f"exclusive open failed ({_short_error(exclusive[0])}); using shared"


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_mapping(info: object) -> dict:
    if isinstance(info, dict):
        return info
    keys = getattr(info, "keys", None)
    if callable(keys):
        return {key: info[key] for key in keys()}  # type: ignore[index]
    return dict(info)  # type: ignore[arg-type]


def _default_input_index(sd: object) -> int:
    try:
        device = sd.default.device  # type: ignore[attr-defined]
    except Exception:
        return -1
    try:
        if isinstance(device, int):
            return device
        return int(device[0])
    except (TypeError, ValueError, IndexError, KeyError):
        return -1
