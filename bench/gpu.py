"""Фоновый опрос nvidia-smi: VRAM, загрузка и мощность GPU во время прогона."""
from __future__ import annotations

import subprocess
import threading
import time

QUERY = "memory.used,utilization.gpu,power.draw,temperature.gpu"


def _num(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


class GpuMonitor:
    def __init__(self, period: float = 1.0):
        self.period, self.samples, self._stop = period, [], threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
                mem, util, power, temp = (_num(x.strip()) for x in out.split(","))
                self.samples.append((time.time(), mem, util, power, temp))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)

    def window(self, t_from: float, t_to: float) -> dict:
        s = [x for x in self.samples if t_from <= x[0] <= t_to]
        if not s:
            return {}
        avg = lambda i: round(sum(x[i] for x in s) / len(s), 1)  # noqa: E731
        return {"vram_mb_max": max(x[1] for x in s), "gpu_util_avg": avg(2),
                "power_w_avg": avg(3), "power_w_max": max(x[3] for x in s), "temp_c_max": max(x[4] for x in s)}
