from __future__ import annotations

import threading
import time
from typing import List, Optional


class EnergyMeter:
    def __init__(self, device_index: int = 0, sample_hz: float = 10.0):
        self.device_index = device_index
        self.interval = 1.0 / sample_hz
        self.samples: List[float] = []
        self.timestamps: List[float] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.available = False
        self._handle = None

        try:
            import pynvml

            self.pynvml = pynvml
            self.pynvml.nvmlInit()
            self._handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.available = True
        except Exception:
            self.pynvml = None

    def _watts(self) -> float:
        return self.pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._watts())
                self.timestamps.append(time.time())
            except Exception:
                pass
            time.sleep(self.interval)

    def idle_baseline(self, seconds: float = 5.0) -> Optional[float]:
        if not self.available:
            return None
        readings = []
        end = time.time() + seconds
        while time.time() < end:
            readings.append(self._watts())
            time.sleep(self.interval)
        return sum(readings) / len(readings) if readings else None

    def start(self) -> None:
        if not self.available:
            return
        self.samples, self.timestamps = [], []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if not self.available or self._thread is None:
            return {"available": False}
        self._stop.set()
        self._thread.join(timeout=2.0)

        if len(self.samples) < 2:
            return {"available": True, "joules": 0.0, "samples": len(self.samples)}

        joules = 0.0
        for i in range(1, len(self.samples)):
            dt = self.timestamps[i] - self.timestamps[i - 1]
            joules += 0.5 * (self.samples[i] + self.samples[i - 1]) * dt

        duration = self.timestamps[-1] - self.timestamps[0]
        return {
            "available": True,
            "joules": round(joules, 2),
            "duration_s": round(duration, 3),
            "mean_watts": round(joules / duration, 2) if duration else 0.0,
            "peak_watts": round(max(self.samples), 2),
            "samples": len(self.samples),
        }

    def shutdown(self) -> None:
        if self.available and self.pynvml is not None:
            try:
                self.pynvml.nvmlShutdown()
            except Exception:
                pass
