"""CPU-affinity portability tests for the latency benchmark."""

from benchmarks import benchmark_cpu_latency as benchmark


def test_invalid_requested_cpu_falls_back_to_allowed_cpu(monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda _pid: {2, 4})
    monkeypatch.setattr(
        benchmark.os, "sched_setaffinity", lambda pid, cpus: calls.append((pid, cpus))
    )

    assert benchmark._pin_cpu(5) == 2
    assert calls == [(0, {2})]
    assert "CPU 5 is not available" in capsys.readouterr().out


def test_affinity_oserror_is_nonfatal(monkeypatch, capsys) -> None:
    monkeypatch.setattr(benchmark.os, "sched_getaffinity", lambda _pid: {0})

    def fail_to_pin(_pid, _cpus):
        raise OSError("not supported")

    monkeypatch.setattr(benchmark.os, "sched_setaffinity", fail_to_pin)

    assert benchmark._pin_cpu(0) is None
    assert "continuing unpinned" in capsys.readouterr().out
