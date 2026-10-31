"""The microphone reader keeps pulling while the consumer is busy."""

import sys
import time
from unittest import mock

import pytest

from voice_keyboard.audio_capture import AudioCapture
from voice_keyboard.windows.wasapi import InputDeviceError, OpenedCapture, WasapiUnavailable


class _Stream:
    def __init__(self) -> None:
        self.reads = 0
        self.closed = False

    def read(self, n: int, exception_on_overflow: bool = False) -> bytes:
        self.reads += 1
        if self.reads == 1:
            return b"\x01\x02\x03\x04"
        # Later reads block briefly, the way a real device does, so stop()
        # can wait for read() to return before closing.
        time.sleep(0.05)
        if self.closed:
            raise OSError("closed")
        return b"\x00" * 4

    def stop_stream(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True


class _PyAudio:
    def __init__(self) -> None:
        self.stream = _Stream()
        self.terminated = False

    def get_default_input_device_info(self) -> dict:
        return {"index": 1}

    def open(self, **kwargs) -> _Stream:
        return self.stream

    def terminate(self) -> None:
        self.terminated = True


def test_reader_pulls_before_the_consumer_asks() -> None:
    pa = _PyAudio()
    with mock.patch("voice_keyboard.audio_capture.pyaudio.PyAudio", return_value=pa):
        capture = AudioCapture()
        capture.start()
        try:
            deadline = time.time() + 1
            while pa.stream.reads < 1 and time.time() < deadline:
                time.sleep(0.01)
            assert pa.stream.reads >= 1
            assert capture.read_chunk() == b"\x01\x02\x03\x04"
        finally:
            capture.stop()
    assert pa.terminated
    assert pa.stream.closed


def test_read_before_start_raises() -> None:
    capture = AudioCapture()
    with pytest.raises(RuntimeError, match="not started"):
        capture.read_chunk()


def test_stop_is_safe_twice() -> None:
    pa = _PyAudio()
    with mock.patch("voice_keyboard.audio_capture.pyaudio.PyAudio", return_value=pa):
        capture = AudioCapture()
        capture.start()
        capture.stop()
        capture.stop()
    assert not capture.running


def test_linux_keeps_the_pyaudio_period(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    pa = _PyAudio()
    opened = {}

    def open_stream(**kwargs):
        opened.update(kwargs)
        return pa.stream

    pa.open = open_stream
    with mock.patch("voice_keyboard.audio_capture.pyaudio.PyAudio", return_value=pa):
        capture = AudioCapture(sample_rate=16000, chunk_ms=100)
        capture.start()
        try:
            assert capture.chunk_ms == 100
            assert opened["frames_per_buffer"] == 1600
            assert opened["rate"] == 16000
            assert capture.info["backend"] == "pyaudio"
            assert "stream_latency_ms" not in capture.info
        finally:
            capture.stop()


def test_windows_start_uses_wasapi(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    reader = _Stream()
    info = {
        "backend": "wasapi",
        "host_api": "Windows WASAPI",
        "mode": "shared",
        "device": "USB Mic",
        "sample_rate": 16000,
        "chunk_ms": 20,
        "frames": 320,
        "stream_latency_ms": 11.0,
    }
    calls = {}

    def fake_open(**kwargs):
        calls.update(kwargs)
        return OpenedCapture(reader=reader, info=dict(info), chunk_ms=20, frames=320)

    monkeypatch.setattr("voice_keyboard.windows.wasapi.open_capture", fake_open)

    def _no_pyaudio():
        raise AssertionError("PyAudio should not open")

    monkeypatch.setattr("voice_keyboard.audio_capture.pyaudio.PyAudio", _no_pyaudio)
    capture = AudioCapture(chunk_ms=100, device_name="USB", wasapi="shared")
    capture.start()
    try:
        assert calls["mode"] == "shared"
        assert calls["device_name"] == "USB"
        assert capture.chunk_ms == 20
        assert capture.chunk_size == 320
        assert capture.info["stream_latency_ms"] == 11.0
        assert capture.read_chunk() == b"\x01\x02\x03\x04"
    finally:
        capture.stop()
    assert reader.closed


def test_windows_falls_back_to_pyaudio_when_wasapi_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def unavailable(**kwargs):
        del kwargs
        raise WasapiUnavailable("no WASAPI input host API")

    monkeypatch.setattr("voice_keyboard.windows.wasapi.open_capture", unavailable)
    pa = _WindowsPyAudio()
    with mock.patch("voice_keyboard.audio_capture.pyaudio.PyAudio", return_value=pa):
        capture = AudioCapture(chunk_ms=100, device_name="default", wasapi="exclusive")
        capture.start()
        try:
            assert capture.chunk_ms == 20
            assert pa.kwargs["frames_per_buffer"] == 320
            assert pa.kwargs["input_device_index"] == 1  # the WASAPI endpoint, not MME
            assert capture.info["backend"] == "pyaudio"
            assert "no WASAPI input host API" in capture.info["fallback"]
            assert "stream_latency_ms" not in capture.info
        finally:
            capture.stop()


def test_a_missing_windows_device_is_not_replaced(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    def missing(**kwargs):
        del kwargs
        raise InputDeviceError("Input device not found: nope")

    monkeypatch.setattr("voice_keyboard.windows.wasapi.open_capture", missing)

    def _no_pyaudio():
        raise AssertionError("PyAudio should not open")

    monkeypatch.setattr("voice_keyboard.audio_capture.pyaudio.PyAudio", _no_pyaudio)
    capture = AudioCapture(device_name="nope")
    with pytest.raises(InputDeviceError, match="not found"):
        capture.start()


class _WindowsPyAudio:
    """MME and WASAPI entries for the same microphone."""

    def __init__(self) -> None:
        self.stream = _Stream()
        self.kwargs = {}
        self.devices = [
            {"index": 0, "name": "USB Mic", "maxInputChannels": 1, "defaultSampleRate": 48000, "hostApi": 0},
            {"index": 1, "name": "USB Mic", "maxInputChannels": 1, "defaultSampleRate": 48000, "hostApi": 1},
        ]
        self.hosts = {0: {"name": "MME"}, 1: {"name": "Windows WASAPI"}}

    def get_default_input_device_info(self) -> dict:
        return self.devices[0]

    def get_device_count(self) -> int:
        return len(self.devices)

    def get_device_info_by_index(self, index: int) -> dict:
        return self.devices[index]

    def get_host_api_info_by_index(self, index: int) -> dict:
        return self.hosts[index]

    def open(self, **kwargs):
        self.kwargs = kwargs
        return self.stream

    def terminate(self) -> None:
        return None
