import argparse
import subprocess
from pathlib import Path


def _log_package_status(name, package_folder, logging_folder):
    p = subprocess.run(["git", "status"], capture_output=True, cwd=package_folder, check=True)
    result_list = [p.stdout.decode(), p.stderr.decode(), "=" * 20]
    p = subprocess.run(["git", "log", "-n", "3"], capture_output=True, cwd=package_folder, check=True)
    result_list += [p.stdout.decode(), p.stderr.decode()]
    with open(logging_folder / f"{name}.git-status", "w") as file:
        file.write("\n".join(result_list))
    with open(logging_folder / f"{name}.git-patch", "w") as file:
        subprocess.run(["git", "diff"], cwd=package_folder, stdout=file, check=True)


def log_python_env_status(logging_folder):
    folder = Path(logging_folder, "python_env_status")
    folder.mkdir(exist_ok=True, parents=True)
    p = subprocess.run(["pip", "list"], capture_output=True)
    with open(folder / "pip.txt", "w", encoding="utf8") as f:
        subprocess.run(["pip", "list"], stdout=f, check=True)
    with open(folder / "conda.txt", "w", encoding="utf8") as f:
        subprocess.run(["conda", "list"], stdout=f, check=True)
    for row in p.stdout.decode().split("\n")[2:]:
        cols = row.split()
        if len(cols) == 3:  # means locally installed package
            status = _log_package_status(cols[0], cols[2], folder)


parser = argparse.ArgumentParser()
parser.add_argument("--status_folder", "-F", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    folder = Path(args.status_folder)
    log_python_env_status(folder)
