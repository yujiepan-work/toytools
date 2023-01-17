# pylint: disable=no-member,missing-function-docstring,too-few-public-methods
# pylint: disable=invalid-name,redefined-outer-name,missing-class-docstring,bare-except,import-errors


import asyncio
import datetime
import heapq
import json
import os
import subprocess
import time
from pathlib import Path
from typing import List, Union
from copy import deepcopy

try:
    from termcolor import colored
except:

    def colored(string, *args, **kwargs):
        OKGREEN = "\033[92m"
        ENDC = "\033[0m"
        return OKGREEN + str(string) + ENDC


__all__ = ["avail_cuda_list", "Launcher", "Job"]


def now_time():
    return datetime.datetime.now().strftime("%Y%m%d-%H:%M:%S")


def readable_seconds(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{ seconds // 60 % 60:02d}:{seconds % 60:02d}"


def avail_cuda_list(memory_requirement: int):
    with subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A5 GPU | grep Free", shell=True, stdout=subprocess.PIPE
    ) as p:
        free_mem = [(-int(x.split()[2]), i) for i, x in enumerate(p.stdout.readlines())]

    heapq.heapify(free_mem)

    def _get_one(free_mem):
        free, idx = free_mem[0]
        if free + memory_requirement > -20:
            return -1
        heapq.heapreplace(free_mem, (free + memory_requirement, idx))
        return idx

    result = []
    i = _get_one(free_mem)
    while i >= 0:
        result.append(i)
        i = _get_one(free_mem)
    return result


class Job:
    def __init__(self, cmd, cwd, io_folder, env=None) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.io_folder = io_folder
        self.env = env

    @property
    def cmd_str(self):
        if isinstance(self.cmd, str):
            cmd_l = self.cmd.split()
        else:
            cmd_l = self.cmd
        return " ".join(map(str, cmd_l))

    @property
    def cmd_list(self):
        return self.cmd_str.split()


class Launcher:
    def __init__(self, jobs: List[Job], cuda_list: List[Union[str, int]], env: Union[dict, None] = None) -> None:
        self.jobs = jobs
        self.cuda_list = cuda_list
        self.job_status = {i: "pending" for i in range(len(jobs))}
        self.lock = asyncio.Lock()
        self.env = env

    def launch(self):
        start = time.time()
        asyncio.run(self._async_launch(self.jobs, self.cuda_list))
        end = time.time()
        print("Finished in", readable_seconds(end - start))
        print(self.job_status)

    async def _async_launch(self, jobs: List[Job], cuda_list):
        queue = asyncio.Queue()
        for cuda in cuda_list:
            queue.put_nowait(cuda)
        tasks = []
        for job_id, job in enumerate(jobs):
            print(f"Job [{job_id}/{len(jobs)}]:", job.cmd_str)
            tasks.append(self._async_run_job(queue, job, job_id))
        await asyncio.gather(*tasks)

    async def _async_run_job(self, queue: asyncio.Queue, job: Job, job_id: int):
        cuda = await queue.get()
        # Got a cuda. Launch!
        async with self.lock:
            self.job_status[job_id] = "running"
        job_info = f"{now_time()} Starting job #{job_id} with cuda={cuda}: {job.cmd_str}"
        print(colored(job_info, "green"))
        env = deepcopy(job.env) or deepcopy(self.env) or deepcopy(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(cuda)
        if job.io_folder is not None:
            io_folder = Path(job.io_folder)
            io_folder.mkdir(exist_ok=True, parents=True)
            with open(io_folder / "env.json", "w", encoding="utf-8") as f_env:
                json.dump(env, f_env, indent=4)
            with open(io_folder / "cmd.json", "w", encoding="utf-8") as f_cmd:
                json.dump(
                    {
                        "str": job.cmd_str,
                        "cmd_list": job.cmd_list,
                    },
                    f_cmd,
                    indent=4,
                )
            with open(io_folder / "stdout.log", "w", encoding="utf-8") as f_out, open(
                io_folder / "stderr.log", "w", encoding="utf-8"
            ) as f_err:
                proc = await asyncio.subprocess.create_subprocess_shell(
                    job.cmd_str, stdout=f_out, stderr=f_err, env=env, cwd=job.cwd
                )
        else:
            proc = await asyncio.subprocess.create_subprocess_shell(job.cmd_str, env=env, cwd=job.cwd)
        await proc.wait()
        if proc.returncode == 0:
            print(now_time(), f"Finished job #{job_id}: {job.cmd_str}")
            status = "finished"
        else:
            print(now_time(), f"FAILED job #{job_id}: {job.cmd_str}")
            status = "failed"
        async with self.lock:
            self.job_status[job_id] = status
        await queue.put(cuda)


if __name__ == "__main__":
    job_list = []
    for i in range(1, 6):
        job_list.append(Job(cmd=["echo", i], cwd=".", io_folder="."))
    Launcher(job_list, [0, 1, 2]).launch()
