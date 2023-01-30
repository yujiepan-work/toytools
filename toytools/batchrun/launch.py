import asyncio
import heapq
import json
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("toytools.batchrun")
logger.setLevel(logging.INFO)
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"


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


class ResourceManager:
    def __init__(self, items: List[Any]) -> None:
        self._resources = asyncio.Queue()
        for item in items:
            self._resources.put_nowait(item)

    @asynccontextmanager
    async def allocate(self, quantity: int = 1):
        items = []
        for _ in range(quantity):
            items.append(await self._resources.get())
        yield items
        for item in items:
            await self._resources.put(item)


@dataclass
class Task:
    def __init__(
        self,
        cmd: Union[str, List[Any]],
        cwd: Union[str, Path],
        io_folder: Union[str, Path],
        identifier: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cuda_quantity: int = 1,
        prepare_fn: Optional[Callable] = None,
        prepare_fn_args: Optional[tuple] = None,
    ) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.io_folder = io_folder
        self.env = env or os.environ.copy()
        self.cuda_quantity = cuda_quantity
        self.prepare_fn = prepare_fn or Mock()
        self.prepare_fn_args = prepare_fn_args or tuple()
        self.identifier = str(identifier) or self.cmd_str()

    def cmd_str(self):
        cmd_l = self.cmd.split() if isinstance(self.cmd, str) else self.cmd
        return " ".join(map(str, cmd_l))

    def cmd_list(self):
        return self.cmd_str().split()


def pre_launch_worker(io_folder):
    Path(io_folder).mkdir()


class Launcher:
    def __init__(self, cuda_list: List[int]) -> None:
        self.cuda_list = cuda_list

    def run(self, tasks: List[Task]):
        asyncio.run(self._run(tasks))

    async def _run(self, tasks):
        rm = ResourceManager(self.cuda_list)
        jobs = [self._launch_task(task, i, rm) for i, task in enumerate(tasks)]
        await asyncio.gather(*jobs)

    async def _launch_task(self, task: Task, task_id: int, resource_manager: ResourceManager) -> str:
        async with resource_manager.allocate(quantity=task.cuda_quantity) as cuda_list:
            cuda = ",".join(map(str, cuda_list))
            logger.info("Running Task-%d CUDA=%s: %s", task_id, cuda, task.identifier)
            env = deepcopy(task.env)
            env[CUDA_VISIBLE_DEVICES] = str(cuda)
            task.prepare_fn(*task.prepare_fn_args)

            io_folder = Path(task.io_folder)
            io_folder.mkdir(exist_ok=True, parents=True)
            with open(io_folder / "task_description.json", "w", encoding="utf-8") as f_task_desc:
                json.dump(
                    {
                        "cmd_str": task.cmd_str(),
                        "cwd": Path(task.cwd).absolute().as_posix(),
                        "cmd_list": task.cmd_list(),
                        "env": dict(sorted(env.items())),
                    },
                    f_task_desc,
                    indent=4,
                )
            proc = await self._run_single_process(task.cmd_str(), task.io_folder, task.cwd, env)
            status = "SUCCESS" if proc.returncode == 0 else "FAIL"
            logger.warning("%s Task-%d PID=%d CUDA=%s: %s", status, task_id, proc.pid, cuda, task.identifier)
            return status

    async def _run_single_process(self, cmd: str, io_folder: Union[Path, str], cwd: str, env: Dict[str, str]):
        with open(io_folder / "stdout.log", "w", encoding="utf-8") as f_out, open(
            io_folder / "stderr.log", "w", encoding="utf-8"
        ) as f_err:
            proc = await asyncio.create_subprocess_shell(cmd, stdout=f_out, stderr=f_err, cwd=cwd, env=env)
            await proc.wait()
            return proc


if __name__ == "__main__":
    launcher = Launcher([1, 2, 3])
    tasks = []

    def create_folder(folder):
        Path(folder).mkdir(exist_ok=True)

    for i in range(5):
        task = Task(
            cmd=["python", "-c", f'"print({i})"'],
            cwd=".",
            io_folder=f"/tmp/{i}",
            prepare_fn=create_folder,
            prepare_fn_args=(f"/tmp/{i}",),
        )
        tasks.append(task)
    launcher.run(tasks)
