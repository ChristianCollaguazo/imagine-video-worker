"""H3 runtime settings and Linux cgroup counters, without importing PyTorch."""
from __future__ import annotations

import math
import os
from pathlib import Path


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _cpu_cgroups(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Find the process CPU cgroup and its visible ancestors, leaf first."""
    for line in _read(membership).splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, path = fields
        names = controllers.split(",")
        if controllers and "cpu" not in names:
            continue
        bases = [root] if not controllers else [root / pair for pair in (
            controllers, "cpu,cpuacct", "cpuacct,cpu", "cpu",
        )]
        for base in bases:
            if not (base / "cpu.stat").is_file():
                continue
            leaf = base / path.lstrip("/")
            if leaf.is_dir():
                return [leaf, *[parent for parent in leaf.parents if parent == base or base in parent.parents]]
            return [base]
    return [root]


def _quota(path):
    try:
        v2 = _read(path / "cpu.max")
        if v2:
            amount, period = v2.split()
            return None if amount == "max" else int(amount) / int(period)
        amount, period = int(_read(path / "cpu.cfs_quota_us")), int(_read(path / "cpu.cfs_period_us"))
        return amount / period if amount > 0 and period > 0 else None
    except (ValueError, ZeroDivisionError):
        return None


def cpu_limits(cgroups=None):
    if cgroups is None:
        cgroups = _cpu_cgroups()
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quotas = [quota for path in cgroups if (quota := _quota(path)) is not None and quota > 0]
    quota = min(quotas, default=None)
    return {"affinity_cpus": affinity, "quota_cpus": quota,
            "effective_cpus": min(affinity, quota) if quota is not None else affinity}


def cpu_throttling(cgroups):
    """Keep leaf and limiting ancestors distinct to avoid double-counting."""
    counters = {}
    for index, path in enumerate(cgroups):
        if index and _quota(path) is None:
            continue
        values = {}
        for line in _read(path / "cpu.stat").splitlines():
            key, _, value = line.partition(" ")
            if key in {"nr_periods", "nr_throttled", "throttled_usec", "throttled_time"} and value.isdecimal():
                values[key] = int(value)
        if "throttled_time" in values:
            values["throttled_usec"] = values.pop("throttled_time") // 1000
        if values:
            counters["leaf" if index == 0 else f"ancestor_{index}"] = values
    return counters


def counter_delta(before, after):
    return {scope: {key: value - before[scope][key]
                    for key, value in current.items() if key in before[scope] and value >= before[scope][key]}
            for scope, current in after.items() if scope in before}


def _integer_or_auto(value, name, automatic, *, allow_auto=True):
    if value == "auto" and allow_auto:
        return automatic
    if not value.isascii() or not value.isdecimal() or len(value) > 4:
        suffix = " or auto" if allow_auto else ""
        raise ValueError(f"{name} must be a nonnegative integer{suffix}")
    return int(value)


def vae_output_mode(configured=None):
    value = os.environ.get("IMAGINE_H3_VAE_OUTPUT", "cpu") if configured is None else configured
    if value not in {"cpu", "gpu-buffer"}:
        raise ValueError("IMAGINE_H3_VAE_OUTPUT must be cpu or gpu-buffer")
    return value


def runtime_settings():
    effective = max(1, math.floor(cpu_limits()["effective_cpus"]))
    intra = _integer_or_auto(os.environ.get("IMAGINE_H3_INTRAOP_THREADS", "0"),
                             "IMAGINE_H3_INTRAOP_THREADS", effective)
    inter = _integer_or_auto(os.environ.get("IMAGINE_H3_INTEROP_THREADS", "0"),
                             "IMAGINE_H3_INTEROP_THREADS", min(4, effective))
    trace = _integer_or_auto(os.environ.get("IMAGINE_H3_CUDA_TRACE_PROMPT", "0"),
                              "IMAGINE_H3_CUDA_TRACE_PROMPT", 0, allow_auto=False)
    if trace and os.environ.get("IMAGINE_DEBUG_METRICS") != "1":
        raise ValueError("IMAGINE_H3_CUDA_TRACE_PROMPT requires IMAGINE_DEBUG_METRICS=1")
    profile = os.environ.get("IMAGINE_H3_VRAM_PROFILE", "auto")
    if profile not in {"auto", "high"}:
        raise ValueError("IMAGINE_H3_VRAM_PROFILE must be auto or high")
    reserve = os.environ.get("IMAGINE_H3_RESERVE_VRAM_GIB", "")
    if reserve:
        try:
            gib = float(reserve)
        except ValueError:
            gib = -1
        if not 0 <= gib < 1024:
            raise ValueError("IMAGINE_H3_RESERVE_VRAM_GIB must be a finite number from 0 to less than 1024")
        reserve = str(gib)
    elif profile == "high":
        reserve = "12"
    return intra, inter, trace, reserve or "-", vae_output_mode()


def _main():
    try:
        print(*runtime_settings())
    except ValueError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    _main()
