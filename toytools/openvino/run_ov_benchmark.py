import datetime
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class BenchmarkResult:
    stdout: str = ''
    stderr: str = ''
    avg_latency: float = 0.
    throughput: float = 0.
    cmd: str = ''
    openvino_version: str = ''
    status_ok: bool = True


def run_ov_benchmark(cmd):
    from openvino.runtime import get_version
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        p.wait()
        stdout = p.stdout.read().decode()
        stderr = p.stderr.read().decode()
        stdout = stdout.strip()
        avg_line = filter(None, stdout.split('\n')[-4].split())
        throughput_line = filter(None, stdout.split('\n')[-1].split())
        avg_line = list(avg_line)
        throughput_line = list(throughput_line)
        avg_latency = -1
        throughput = -1
        status_ok = False
        if 'Average:' in avg_line and 'Throughput:' in throughput_line:
            avg_latency = float(list(avg_line)[-2])
            throughput = float(list(throughput_line)[-2])
            status_ok = True
        return BenchmarkResult(
            stdout=stdout,
            stderr=stderr,
            avg_latency=avg_latency,
            throughput=throughput,
            cmd=cmd,
            openvino_version=get_version(),
            status_ok=status_ok,
        )
