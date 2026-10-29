# SPDX-License-Identifier: Apache-2.0
"""The three array libraries the harness compares, behind one small interface.

Each backend moves a NumPy array to its device, brings a result back, waits
for its device to finish, and offers the clocks that can time one call.
The GPU libraries are imported lazily, so a machine without them (or without
a GPU) can still run the NumPy backend.
"""

import importlib
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

# A NumPy array, a CuPy array or a PyTorch tensor. The GPU libraries are
# imported at runtime, so their types are not visible to the type checker.
Array = Any
FloatArray = npt.NDArray[np.float32]
Call = Callable[[], Array]
Timer = Callable[[Call], float]


class BackendUnavailableError(Exception):
    """Raised when a backend's library or its GPU cannot be used here."""


class Backend(Protocol):
    """What the harness needs from an array library."""

    name: str

    def asarray(self, host: FloatArray) -> Array: ...

    def to_numpy(self, x: Array) -> npt.NDArray[Any]: ...

    def synchronize(self) -> None: ...

    def timers(self) -> dict[str, Timer]: ...


class NumpyBackend:
    """NumPy on the CPU: the baseline, timed with a CPU clock."""

    name = "numpy"

    def asarray(self, host: FloatArray) -> Array:
        return host

    def to_numpy(self, x: Array) -> npt.NDArray[Any]:
        return np.asarray(x)

    def synchronize(self) -> None:
        # NumPy returns only after the work is done: there is no queue to wait on.
        pass

    def timers(self) -> dict[str, Timer]:
        return {"cpu_timer": _cpu_timer_ms}


class _GpuBackend:
    """The timing methods shared by the CUDA backends.

    Subclasses provide the library-specific parts: moving data, waiting for
    the GPU, and recording CUDA events.
    """

    name = "gpu"

    def synchronize(self) -> None:
        raise NotImplementedError

    def _events_ms(self, fn: Call) -> float:
        raise NotImplementedError

    def timers(self) -> dict[str, Timer]:
        return {
            "cuda_events": self._events_ms,
            "synced_cpu_timer": self._synced_cpu_timer_ms,
            "unsynced_cpu_timer": self._unsynced_cpu_timer_ms,
        }

    def _synced_cpu_timer_ms(self, fn: Call) -> float:
        # Empty the GPU's queue before starting the clock, so earlier work is
        # not counted, and wait for this call before stopping it.
        self.synchronize()
        start = time.perf_counter()
        fn()
        self.synchronize()
        return (time.perf_counter() - start) * 1000

    def _unsynced_cpu_timer_ms(self, fn: Call) -> float:
        # Deliberately wrong, kept as the "what not to do" row: the clock stops
        # when Python gets control back, which is right after the launch, not
        # after the GPU finishes. The final synchronize is outside the timed
        # region; it only keeps the next repetition from inheriting this work.
        self.synchronize()
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000
        self.synchronize()
        return elapsed


class CupyBackend(_GpuBackend):
    """CuPy on the GPU."""

    name = "cupy"

    def __init__(self) -> None:
        try:
            self._cp: Any = importlib.import_module("cupy")
            device_count = self._cp.cuda.runtime.getDeviceCount()
        except ImportError as error:
            raise BackendUnavailableError(f"cupy not installed: {error}") from error
        except Exception as error:
            # CuPy raises its own runtime errors when there is no driver or GPU;
            # any of them means the same thing here: this backend cannot run.
            raise BackendUnavailableError(
                f"cupy cannot reach a GPU: {error}"
            ) from error
        if device_count < 1:
            raise BackendUnavailableError("cupy sees no CUDA device")

    def asarray(self, host: FloatArray) -> Array:
        return self._cp.asarray(host)

    def to_numpy(self, x: Array) -> npt.NDArray[Any]:
        return np.asarray(self._cp.asnumpy(x))

    def synchronize(self) -> None:
        self._cp.cuda.Device().synchronize()

    def _events_ms(self, fn: Call) -> float:
        # The events go into the GPU's queue around the call; the GPU itself
        # records the time when it reaches each one.
        start = self._cp.cuda.Event()
        end = self._cp.cuda.Event()
        start.record()
        fn()
        end.record()
        end.synchronize()
        return float(self._cp.cuda.get_elapsed_time(start, end))


class TorchBackend(_GpuBackend):
    """PyTorch on the GPU."""

    name = "torch"

    def __init__(self) -> None:
        try:
            self._torch: Any = importlib.import_module("torch")
        except ImportError as error:
            raise BackendUnavailableError(f"torch not installed: {error}") from error
        except OSError as error:
            # A broken install fails here, while loading its bundled CUDA
            # libraries, before any GPU is touched.
            raise BackendUnavailableError(f"torch cannot load: {error}") from error
        if not self._torch.cuda.is_available():
            raise BackendUnavailableError("torch sees no CUDA device")
        self._device = self._torch.device("cuda")
        # PyTorch creates a CUDA event lazily, on its first record(). Created
        # per call, the end event would be created inside the timed region,
        # and its creation time counted whenever the GPU finishes first. One
        # pair, created here and recorded once, is reused by every timed call.
        self._start = self._torch.cuda.Event(enable_timing=True)
        self._end = self._torch.cuda.Event(enable_timing=True)
        self._start.record()
        self._end.record()
        self._end.synchronize()

    def asarray(self, host: FloatArray) -> Array:
        return self._torch.from_numpy(host).to(self._device)

    def to_numpy(self, x: Array) -> npt.NDArray[Any]:
        return np.asarray(x.cpu().numpy())

    def synchronize(self) -> None:
        self._torch.cuda.synchronize()

    def _events_ms(self, fn: Call) -> float:
        # Same idea as CuPy: markers in the GPU's queue, timed by the GPU.
        self._start.record()
        fn()
        self._end.record()
        self._end.synchronize()
        return float(self._start.elapsed_time(self._end))

    def float32_matmul_precision(self) -> str:
        """PyTorch's matmul precision setting, which can change float32 results."""
        return str(self._torch.get_float32_matmul_precision())


def _cpu_timer_ms(fn: Call) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


BACKENDS: dict[str, Callable[[], Backend]] = {
    "numpy": NumpyBackend,
    "cupy": CupyBackend,
    "torch": TorchBackend,
}
