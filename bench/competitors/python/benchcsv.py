"""Schema-v2 rows from Python, byte-compatible with dev.cajeta.bench.

The header, column order, field quoting, the min/median/mean/p95 rules and the
smoke tagging are the harness's own (cajeta-bench src/main/cajeta/dev/cajeta/
bench/report/Result.cajeta, Csv.cajeta, Harness.cajeta), so a Python row and a
Cajeta row land in one results.csv that tools/verify-fidelity.py reads as one.

Environment, as the harness reads it:
  CAJETA_BENCH_SMOKE=1     warmup 1, trials 3, every row's notes tagged `smoke`
  CAJETA_BENCH_MACHINE     the machine column (default: hostname, lowercased)
  CAJETA_BENCH_RUN_ID      the run_id column
  CAJETA_BENCH_NO_GPU=1    the GPU is not ours: device rows are skipped and
                           nothing touches it (no torch.cuda call at all)
"""
import os
import platform
import resource
import socket
import sys
import time

HEADER = (
    "schema_version,run_id,timestamp_unix,benchmark,area,dataset_name,dataset_bytes,"
    "dataset_sha256,input_size,language,language_version,library,library_version,flags,"
    "warmup,trials,min_ns,median_ns,mean_ns,p95_ns,throughput,throughput_unit,peak_rss_kb,"
    "cgroup_peak_kb,alloc_bytes,alloc_count,working_set_kb,status,check_ok,notes,variant,"
    "backend,machine,os,driver,model,platform"
)
COLUMNS = HEADER.split(",")
GPU_HELD = "GPU held by another session (CAJETA_BENCH_NO_GPU=1)"


def flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def smoke():
    return flag("CAJETA_BENCH_SMOKE")


def no_gpu():
    return flag("CAJETA_BENCH_NO_GPU")


def machine():
    m = os.environ.get("CAJETA_BENCH_MACHINE", "").strip()
    if m:
        return m
    h = socket.gethostname().strip()
    return h.lower() if h else "unknown"


def _key_value(text, key):
    for line in text.splitlines():
        if line.startswith(key + "="):
            return line[len(key) + 1:].strip().strip('"')
    return ""


def detect_os():
    """`NAME VERSION_ID` from os-release plus ` (WSL2)`: OsDetect.fromLinux."""
    if sys.platform.startswith("win"):
        return f"Windows ({platform.version()})"
    rel = ""
    for p in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            rel = open(p).read()
            break
        except OSError:
            pass
    try:
        pv = open("/proc/version").read().lower()
    except OSError:
        pv = ""
    name, ver = _key_value(rel, "NAME"), _key_value(rel, "VERSION_ID")
    base = (name + " " + ver if ver else name) if name else (_key_value(rel, "PRETTY_NAME") or "Linux")
    if "wsl2" in pv or "microsoft-standard" in pv:
        return base + " (WSL2)"
    if "microsoft" in pv:
        return base + " (WSL1)"
    return base


def warmup_trials(warmup, trials):
    return (1, 3) if smoke() else (warmup, max(1, trials))


def measure(fn, warmup, trials):
    """Warmup then timed trials of fn(); returns the sorted ns samples."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(trials):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(max(0, time.perf_counter_ns() - t0))
    samples.sort()
    return samples


def stats(samples):
    n = len(samples)
    p95 = min((n * 95) // 100, n - 1)
    return samples[0], samples[n // 2], sum(samples) // n, samples[p95]


def peak_rss_kb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def notes_for(*parts):
    out = [p for p in parts if p]
    if smoke():
        out.append("smoke")
    return "; ".join(out)


def row(**kw):
    """A row with the harness's defaults (Result()), overridden by kw."""
    r = {
        "schema_version": "2", "run_id": os.environ.get("CAJETA_BENCH_RUN_ID", "local-run"),
        "timestamp_unix": str(int(time.time())), "benchmark": "", "area": "", "dataset_name": "",
        "dataset_bytes": "-1", "dataset_sha256": "", "input_size": "-1", "language": "python",
        "language_version": platform.python_version(), "library": "", "library_version": "",
        "flags": "", "warmup": "0", "trials": "0", "min_ns": "-1", "median_ns": "-1",
        "mean_ns": "-1", "p95_ns": "-1", "throughput": "", "throughput_unit": "",
        "peak_rss_kb": "-1", "cgroup_peak_kb": "-1", "alloc_bytes": "-1", "alloc_count": "-1",
        "working_set_kb": "-1", "status": "ok", "check_ok": "true", "notes": "", "variant": "",
        "backend": "cpu", "machine": machine(), "os": detect_os(), "driver": "", "model": "",
        "platform": "",
    }
    for k, v in kw.items():
        if k not in r:
            raise KeyError(k)
        r[k] = v if isinstance(v, str) else ("true" if v is True else "false" if v is False else str(v))
    return r


def measured_row(samples, warmup, input_size=-1, unit="", check_ok=True, note="", **kw):
    mn, md, me, p95 = stats(samples)
    thr = ""
    if unit and input_size > 0 and md > 0:
        thr = f"{input_size / (md / 1e9):.6g}"
    return row(warmup=warmup, trials=len(samples), min_ns=mn, median_ns=md, mean_ns=me,
               p95_ns=p95, input_size=input_size, throughput=thr, throughput_unit=unit,
               peak_rss_kb=peak_rss_kb(), status="ok" if check_ok else "invalid",
               check_ok=bool(check_ok), notes=notes_for(note), **kw)


def skipped_row(reason, **kw):
    return row(status="skipped", check_ok=False, notes=notes_for(reason), **kw)


def failed_row(message, **kw):
    return row(status="failed", check_ok=False, notes=notes_for(message), **kw)


def field(s):
    """Csv.field: quote only on , \" CR LF; double embedded quotes."""
    s = "" if s is None else str(s)
    if any(c in s for c in ',"\r\n'):
        return '"' + s.replace('"', '""') + '"'
    return s


def write(path, rows):
    with open(path, "w", newline="") as f:
        f.write(HEADER + "\n")
        for r in rows:
            f.write(",".join(field(r[c]) for c in COLUMNS) + "\n")
