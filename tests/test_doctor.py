from voice_keyboard import doctor


def test_render_marks_and_fixes() -> None:
    text = doctor.render([
        doctor.Finding(doctor.OK, "config", "/x/config.toml"),
        doctor.Finding(doctor.FAIL, "typing", "/dev/uinput is missing", "sudo modprobe uinput"),
        doctor.Finding(doctor.WARN, "clipboard", "no xclip", "install xclip"),
    ])
    assert "✓ config" in text and "✗ typing" in text and "! clipboard" in text
    assert "→ sudo modprobe uinput" in text
    assert "1 problem(s) to fix, 1 warning(s)." in text


def test_all_good_says_so() -> None:
    assert "Everything checks out." in doctor.render([doctor.Finding(doctor.OK, "config", "x")])


def test_missing_config_is_a_failure_with_the_fix(tmp_path) -> None:
    finding = doctor.check_config(tmp_path / "config.toml")
    assert finding.status == doctor.FAIL and "voice-keyboard setup" in finding.fix


def test_a_broken_check_never_hides_the_others(tmp_path, monkeypatch) -> None:
    def boom():
        raise RuntimeError("kaput")

    monkeypatch.setattr(doctor, "check_typing", boom)
    findings = doctor.run(tmp_path / "config.toml", None)
    labels = [f.label for f in findings]
    assert "clipboard" in labels and "focus" in labels
    assert any("kaput" in f.detail for f in findings)


def test_windows_microphone_prefers_wasapi(monkeypatch) -> None:
    import sys

    import voice_keyboard.client as client

    monkeypatch.setattr(sys, "platform", "win32")
    devices = [
        {"name": "USB Mic", "channels": 1, "rate": 48000, "default": True, "host_api": "MME", "index": 0},
        {"name": "USB Mic", "channels": 1, "rate": 48000, "default": False, "host_api": "Windows WASAPI", "index": 4},
    ]
    monkeypatch.setattr(client, "_list_input_devices", lambda: devices)
    assert doctor.check_audio({"audio": {"device_name": "usb"}}).status == doctor.OK
    assert doctor.check_audio({"audio": {"device_name": "usb"}}).detail == "USB Mic"
    ambiguous = [
        {"name": "USB Mic A", "channels": 1, "rate": 48000, "default": False, "host_api": "Windows WASAPI"},
        {"name": "USB Mic B", "channels": 1, "rate": 48000, "default": False, "host_api": "Windows WASAPI"},
    ]
    monkeypatch.setattr(client, "_list_input_devices", lambda: ambiguous)
    assert doctor.check_audio({"audio": {"device_name": "mic"}}).status == doctor.FAIL


def test_windows_capture_shows_the_opened_stream_or_the_plan(monkeypatch) -> None:
    import sys

    monkeypatch.setattr(sys, "platform", "win32")
    live = doctor.check_windows_capture(
        {"audio": {"chunk_ms": 100, "wasapi": "shared", "device_name": "default"}},
        {"capture": {
            "backend": "wasapi",
            "host_api": "Windows WASAPI",
            "mode": "shared",
            "chunk_ms": 20,
            "device": "USB Mic",
            "stream_latency_ms": 13.5,
        }},
    )
    assert live is not None and live.status == doctor.OK
    assert "13.5" in live.detail and "20 ms" in live.detail
    planned = doctor.check_windows_capture(
        {"audio": {"chunk_ms": 100, "wasapi": "shared", "device_name": "USB Mic"}},
        {},
    )
    assert planned is not None and planned.status == doctor.OK
    assert "20 ms" in planned.detail
    assert "latency" not in planned.detail
    fallback = doctor.check_windows_capture(
        {"audio": {}},
        {"capture": {
            "backend": "wasapi",
            "mode": "shared",
            "chunk_ms": 20,
            "fallback": "exclusive open failed (busy); using shared",
        }},
    )
    assert fallback is not None and fallback.status == doctor.WARN
    assert "busy" in fallback.detail


def test_device_name_must_match_one_device(monkeypatch) -> None:
    import voice_keyboard.client as client

    devices = [
        {"name": "USB Mic", "channels": 1, "rate": 48000, "default": False},
        {"name": "Built-in Mic", "channels": 2, "rate": 44100, "default": True},
    ]
    monkeypatch.setattr(client, "_list_input_devices", lambda: devices)
    assert doctor.check_audio({"audio": {"device_name": "usb"}}).status == doctor.OK
    assert doctor.check_audio({"audio": {"device_name": "mic"}}).status == doctor.FAIL  # two match
    assert doctor.check_audio({"audio": {}}).detail == "Built-in Mic"


def test_llm_features_without_llm_fail(monkeypatch) -> None:
    import voice_keyboard.llm as llm

    monkeypatch.setattr(llm, "llm_ready", lambda config: False)
    finding = doctor.check_llm({"polish": {"map": {"slack": "casual"}}, "flow": {}})
    assert finding.status == doctor.FAIL and "[polish.map]" in finding.detail
    # "VK, …" rewrites and pause review use [llm] by default.
    assert doctor.check_llm({"flow": {}}).status == doctor.WARN
    assert doctor.check_llm({"flow": {"wake_word": "", "pause_review": "rules"}}).status == doctor.OK
    monkeypatch.setattr(llm, "llm_ready", lambda config: True)
    assert doctor.check_llm({"flow": {"corrections": "llm", "wake_word": "", "pause_review": "rules"}}).detail == (
        "ready for self-corrections"
    )


def test_a_cloud_provider_without_a_key_is_a_failure() -> None:
    finding = doctor.check_speech({"stt": {"provider": "deepgram"}, "providers": {"deepgram": {"api_key": ""}}})
    assert finding.status == doctor.FAIL and "[providers.deepgram] api_key" in finding.fix
    hyper = doctor.check_speech({"stt": {"provider": "hyperfurion"}})
    assert hyper.status == doctor.FAIL and "base_url" in hyper.fix
    keyed = doctor.check_speech({"stt": {"provider": "deepgram"}, "providers": {"deepgram": {"api_key": "k"}}})
    assert keyed.status == doctor.OK


def test_a_local_server_needs_no_key() -> None:
    config = {"stt": {"provider": "openai"}, "providers": {"openai": {"base_url": "http://localhost:8000/v1"}}}
    finding = doctor.check_speech(config)
    assert finding.status == doctor.OK and "local" in finding.detail
