"""Windows WASAPI capture decisions, without a Windows device."""

import sys
from types import SimpleNamespace

import pytest

from voice_keyboard.windows.wasapi import (
    EXCLUSIVE,
    SHARED,
    InputDevice,
    InputDeviceError,
    SoundDeviceBackend,
    WasapiUnavailable,
    chunk_ms_for_stream,
    effective_chunk_ms,
    format_capture,
    open_attempts,
    open_capture,
    planned_capture_line,
    select_input,
    stream_latency_ms,
)


def _device(index, name, host, *, default=False, channels=1, rate=48000):
    return InputDevice(index, name, host, channels, rate, default)


def _inputs():
    return [
        _device(0, "USB Mic", "MME", default=True),
        _device(1, "Speakers", "Windows WASAPI", channels=0),
        _device(3, "USB Mic", "Windows WASAPI", default=True),
        _device(4, "Headset", "Windows DirectSound"),
    ]


class TestPeriod:
    def test_stock_100ms_opens_at_20(self) -> None:
        assert effective_chunk_ms(100) == 20

    def test_an_explicit_period_is_kept(self) -> None:
        assert effective_chunk_ms(80) == 80
        assert effective_chunk_ms(20) == 20
        assert effective_chunk_ms(200) == 200

    def test_tiny_periods_floor_at_10(self) -> None:
        assert effective_chunk_ms(5) == 10
        assert effective_chunk_ms(True) == 20
        assert effective_chunk_ms("nope") == 20

    def test_silence_timing_uses_the_opened_period(self) -> None:
        capture = SimpleNamespace(chunk_ms=20)
        assert chunk_ms_for_stream(capture, 100) == 20.0
        assert chunk_ms_for_stream(SimpleNamespace(), 80) == 80.0
        assert chunk_ms_for_stream(None, 0) == 100.0


class TestDeviceChoice:
    def test_default_prefers_the_wasapi_endpoint(self) -> None:
        chosen = select_input(_inputs(), "default")
        assert chosen.status == "ok"
        assert chosen.device is not None
        assert chosen.device.index == 3
        assert chosen.device.host_api == "Windows WASAPI"

    def test_a_shared_name_does_not_open_mme(self) -> None:
        chosen = select_input(_inputs(), "USB")
        assert chosen.device is not None and chosen.device.index == 3

    def test_two_wasapi_matches_are_ambiguous(self) -> None:
        devices = [
            _device(1, "USB Mic A", "Windows WASAPI"),
            _device(2, "USB Mic B", "Windows WASAPI"),
        ]
        chosen = select_input(devices, "usb")
        assert chosen.status == "ambiguous"
        assert chosen.names == ("USB Mic A", "USB Mic B")

    def test_a_substring_that_hits_the_default_and_another_stays_ambiguous(self) -> None:
        devices = [
            _device(0, "USB Mic", "Windows WASAPI"),
            _device(1, "Built-in Mic", "Windows WASAPI", default=True),
        ]
        chosen = select_input(devices, "mic")
        assert chosen.status == "ambiguous"
        assert chosen.device is None
        assert chosen.names == ("USB Mic", "Built-in Mic")

    def test_a_missing_name_does_not_fall_through_to_mme(self) -> None:
        chosen = select_input(_inputs(), "Headset")
        assert chosen.status == "none"

    def test_without_wasapi_the_system_default_is_used(self) -> None:
        devices = [_device(0, "USB Mic", "MME", default=True)]
        chosen = select_input(devices, "default")
        assert chosen.device is not None and chosen.device.index == 0


class TestOpenPolicy:
    def test_shared_never_tries_exclusive(self) -> None:
        assert open_attempts("shared") == ("shared",)
        assert open_attempts("nope") == ("shared",)
        assert open_attempts("exclusive") == ("exclusive", "shared")
        assert open_attempts("auto") == ("exclusive", "shared")

    def test_exclusive_failure_falls_back_and_reports_the_real_error(self) -> None:
        backend = _Backend(fail_exclusive=True, latency=0.0125)
        opened = open_capture(
            device_name="USB",
            sample_rate=16000,
            chunk_ms=100,
            mode=EXCLUSIVE,
            backend=backend,
        )
        assert backend.opened == [(3, 16000, 320, True), (3, 16000, 320, False)]
        assert opened.chunk_ms == 20
        assert opened.frames == 320
        assert opened.info["mode"] == SHARED
        assert opened.info["backend"] == "wasapi"
        assert opened.info["device"] == "USB Mic"
        assert opened.info["stream_latency_ms"] == 12.5
        assert "Device in use" in opened.info["fallback"]
        assert "using shared" in opened.info["fallback"]
        assert opened.reader.read(320) == b"\x01\x00" * 320

    def test_shared_does_not_record_a_latency_the_stream_did_not_report(self) -> None:
        backend = _Backend(latency=0)
        opened = open_capture(
            device_name="default",
            sample_rate=16000,
            chunk_ms=100,
            mode=SHARED,
            backend=backend,
        )
        assert backend.opened == [(3, 16000, 320, False)]
        assert opened.info["mode"] == SHARED
        assert "stream_latency_ms" not in opened.info
        assert "fallback" not in opened.info
        assert "stream latency" not in format_capture(opened.info)

    def test_a_missing_or_ambiguous_name_is_not_opened(self) -> None:
        backend = _Backend()
        with pytest.raises(InputDeviceError, match="not found"):
            open_capture(
                device_name="no such mic",
                sample_rate=16000,
                chunk_ms=100,
                mode=SHARED,
                backend=backend,
            )
        assert backend.opened == []
        with pytest.raises(InputDeviceError, match="matches 2"):
            open_capture(
                device_name="usb",
                sample_rate=16000,
                chunk_ms=20,
                mode=SHARED,
                backend=_Backend(devices=[
                    _device(1, "USB Mic A", "Windows WASAPI"),
                    _device(2, "USB Mic B", "Windows WASAPI"),
                ]),
            )

    def test_no_wasapi_host_api_is_unavailable(self) -> None:
        backend = _Backend(devices=[_device(0, "USB Mic", "MME", default=True)])
        with pytest.raises(WasapiUnavailable):
            open_capture(
                device_name="default",
                sample_rate=16000,
                chunk_ms=100,
                mode=SHARED,
                backend=backend,
            )
        assert backend.opened == []


class TestSoundDeviceSettings:
    def test_shared_mode_converts_to_the_recognizer_rate(self) -> None:
        seen = {}

        class Settings:
            def __init__(self, exclusive=False, auto_convert=False, explicit_sample_format=False):
                seen["settings"] = (exclusive, auto_convert, explicit_sample_format)

        class Raw:
            def __init__(self, **kwargs):
                seen["kwargs"] = kwargs
                self.latency = 0.01

            def start(self):
                seen["started"] = True

            def close(self):
                seen["closed"] = True

        sd = SimpleNamespace(
            WasapiSettings=Settings,
            RawInputStream=Raw,
            query_hostapis=lambda: (
                {"name": "MME", "default_input_device": 0},
                {"name": "Windows WASAPI", "default_input_device": 2},
            ),
            query_devices=lambda: [
                {"name": "USB Mic", "hostapi": 0, "max_input_channels": 1, "default_samplerate": 48000.0, "index": 0},
                {"name": "Speakers", "hostapi": 1, "max_input_channels": 0, "default_samplerate": 48000.0, "index": 1},
                {"name": "USB Mic", "hostapi": 1, "max_input_channels": 2, "default_samplerate": 48000.0, "index": 2},
            ],
            default=SimpleNamespace(device=(0, 1)),
        )
        backend = SoundDeviceBackend(sd)
        devices = backend.list_inputs()
        assert [(d.index, d.host_api, d.is_default) for d in devices] == [
            (0, "MME", True),
            (2, "Windows WASAPI", True),
        ]
        backend.open_stream(device_index=2, sample_rate=16000, frames=320, exclusive=False)
        assert seen["settings"] == (False, True, False)
        assert seen["kwargs"]["samplerate"] == 16000
        assert seen["kwargs"]["blocksize"] == 320
        assert seen["kwargs"]["dtype"] == "int16"
        assert seen["kwargs"]["latency"] == "low"
        assert seen["kwargs"]["channels"] == 1
        assert seen["kwargs"]["device"] == 2
        assert seen["started"] is True
        assert stream_latency_ms(SimpleNamespace(latency=0.01)) == 10.0

        backend.open_stream(device_index=2, sample_rate=16000, frames=320, exclusive=True)
        assert seen["settings"] == (True, False, False)
        assert stream_latency_ms(SimpleNamespace(get_input_latency=lambda: 0.011)) == 11.0
        assert stream_latency_ms(SimpleNamespace(get_input_latency=lambda: 0)) is None

    def test_a_stream_that_fails_to_start_is_closed(self) -> None:
        seen = {}

        class Raw:
            def __init__(self, **kwargs):
                del kwargs
                self.latency = None

            def start(self):
                raise OSError("endpoint gone")

            def close(self):
                seen["closed"] = True

        sd = SimpleNamespace(
            WasapiSettings=lambda **kwargs: kwargs,
            RawInputStream=Raw,
        )
        backend = SoundDeviceBackend(sd)
        with pytest.raises(OSError, match="endpoint gone"):
            backend.open_stream(device_index=1, sample_rate=16000, frames=320, exclusive=True)
        assert seen["closed"] is True


class TestReporting:
    def test_format_uses_only_numbers_it_was_given(self) -> None:
        assert format_capture({}) == ""
        assert format_capture({"mode": "shared", "chunk_ms": 20, "device": "USB Mic"}) == (
            "shared, 20 ms buffer — USB Mic"
        )
        text = format_capture({
            "host_api": "Windows WASAPI",
            "mode": "shared",
            "chunk_ms": 20,
            "device": "USB Mic",
            "stream_latency_ms": 13.5,
            "fallback": "exclusive open failed (device busy); using shared",
        })
        assert "13.5 ms" in text
        assert "device busy" in text

    def test_the_plan_from_config_has_no_stream_latency(self) -> None:
        line = planned_capture_line({"audio": {"chunk_ms": 100, "wasapi": "shared", "device_name": "USB Mic"}})
        assert line == "WASAPI shared, 20 ms buffer, device USB Mic (chunk_ms 100 opens at 20 ms)"
        assert "latency" not in line
        kept = planned_capture_line({"audio": {"chunk_ms": 80, "wasapi": "auto", "device_name": "default"}})
        assert kept == "WASAPI auto, 80 ms buffer, device default"


class _Stream:
    def __init__(self, latency):
        self.latency = latency

    def read(self, frames):
        return b"\x01\x00" * frames, False

    def stop(self):
        return None

    def close(self):
        return None


class _Backend:
    def __init__(self, *, fail_exclusive=False, latency=0.0125, devices=None):
        self.fail_exclusive = fail_exclusive
        self.latency = latency
        self.devices = list(_inputs() if devices is None else devices)
        self.opened = []

    def list_inputs(self):
        return [device for device in self.devices if device.channels > 0]

    def open_stream(self, *, device_index, sample_rate, frames, exclusive):
        self.opened.append((device_index, sample_rate, frames, exclusive))
        if exclusive and self.fail_exclusive:
            raise OSError("Device in use")
        return _Stream(self.latency)


def test_linux_doctor_does_not_grow_a_capture_row(monkeypatch) -> None:
    from voice_keyboard import doctor

    monkeypatch.setattr(sys, "platform", "linux")
    assert doctor.check_windows_capture({"audio": {"chunk_ms": 100}}) is None
