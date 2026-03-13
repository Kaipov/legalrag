from arlc import telemetry as telemetry_mod


def test_telemetry_timer_clamps_positive_sub_ms_durations(monkeypatch) -> None:
    timestamps = iter([100.0, 100.0004, 100.0007])
    monkeypatch.setattr(telemetry_mod.time, "perf_counter", lambda: next(timestamps))

    timer = telemetry_mod.TelemetryTimer()
    timer.mark_token()
    timing = timer.finish()

    assert timing.ttft_ms == 1
    assert timing.tpot_ms == 0
    assert timing.total_time_ms == 1
