#!/usr/bin/env python3
"""Environment doctor and dependency bootstrap for the video extractor."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from tempfile import mkstemp

ROOT = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except (AttributeError, OSError):
    pass
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ROOT / "requirements.txt"
REPORT_PATH = ROOT / "environment-report.json"
LOG_DIR = ROOT / "logs"
MIN_PYTHON = (3, 10)
DOTENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def parse_dotenv_value(value: str) -> str:
    value = value.strip()
    if value[:1] in {"'", '"'}:
        quote = value[0]
        closing = value.find(quote, 1)
        if closing >= 0:
            return value[1:closing]
        return value[1:]
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        content = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read environment file {path}: {exc}") from exc
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if DOTENV_KEY_RE.match(key):
            values[key] = parse_dotenv_value(value)
    return values


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv(no_venv: bool) -> None:
    if in_venv() or no_venv:
        return

    py = venv_python()
    version_code = (
        "import sys; "
        "raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    )
    code, _ = run_capture([str(py), "-c", version_code], timeout=20) if py.exists() else (1, "")
    if code != 0 and VENV_DIR.exists():
        backup = ROOT / f".venv.invalid-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
        print(f"[bootstrap] Existing virtual environment is unusable; moving it to: {backup}")
        try:
            VENV_DIR.rename(backup)
        except OSError as exc:
            raise RuntimeError(
                f"Could not move the broken virtual environment. Close running Python processes and remove {VENV_DIR}."
            ) from exc

    if code != 0:
        print(f"[bootstrap] Creating virtual environment: {VENV_DIR}")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        code, output = run_capture([str(py), "-c", version_code], timeout=20)
        if code != 0:
            raise RuntimeError(f"The new virtual environment is unusable: {output}")

    env = os.environ.copy()
    env["VCE_BOOTSTRAPPED"] = "1"
    env["PYTHONUTF8"] = "1"
    args = [str(py), str(Path(__file__).resolve()), *sys.argv[1:]]
    print("[bootstrap] Re-entering the project virtual environment")
    result = subprocess.run(args, env=env, check=False)
    raise SystemExit(result.returncode)


def run_capture(command: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode, output.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def command_version(name: str) -> str | None:
    if not shutil.which(name):
        return None
    code, output = run_capture([name, "--version"], timeout=15)
    return output.splitlines()[0] if code == 0 and output else "installed"


def version_parts(value: str) -> tuple[int, ...]:
    parts = [int(part) for part in re.findall(r"\d+", value)]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def requirement_problems(path: Path) -> list[str]:
    problems: list[str] = []
    if not path.is_file():
        return [f"missing requirements file: {path}"]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.match(
            r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(>=|==)?\s*([0-9][A-Za-z0-9.*+!_-]*)?$",
            line,
        )
        if not match:
            problems.append(f"unsupported requirement: {line}")
            continue
        distribution, operator, required = match.groups()
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            problems.append(f"{distribution} is not installed")
            continue
        if operator and required:
            installed_parts = version_parts(installed)
            required_parts = version_parts(required)
            if operator == ">=" and installed_parts < required_parts:
                problems.append(f"{distribution} {installed} is older than {required}")
            elif operator == "==" and installed_parts != required_parts:
                problems.append(f"{distribution} {installed} does not equal {required}")
    return problems




def internet_ok() -> bool:
    targets = [("pypi.org", 443), ("huggingface.co", 443)]
    for host, port in targets:
        try:
            with socket.create_connection((host, port), timeout=5):
                return True
        except OSError:
            continue
    return False


def gpu_info() -> dict[str, object]:
    info: dict[str, object] = {
        "nvidia_smi": bool(shutil.which("nvidia-smi")),
        "gpus": [],
    }
    if not info["nvidia_smi"]:
        return info
    code, output = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        timeout=20,
    )
    if code == 0:
        info["gpus"] = [line.strip() for line in output.splitlines() if line.strip()]
    return info


def disk_info() -> dict[str, object]:
    try:
        usage = shutil.disk_usage(ROOT)
        return {
            "free_gb": round(usage.free / (1024**3), 2),
            "total_gb": round(usage.total / (1024**3), 2),
        }
    except OSError as exc:
        return {"error": str(exc)}


def base_environment_report() -> dict[str, object]:
    packages: dict[str, str | None] = {}
    for package in ("yt_dlp", "faster_whisper", "imageio_ffmpeg", "ctranslate2"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        packages[package] = spec.origin if spec else None

    ffmpeg = shutil.which("ffmpeg")
    bundled_ffmpeg = None
    try:
        import imageio_ffmpeg  # type: ignore

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "virtual_environment": in_venv(),
        },
        "requirements": {
            "minimum_python": ".".join(map(str, MIN_PYTHON)),
            "python_ok": sys.version_info >= MIN_PYTHON,
        },
        "tools": {
            "git": command_version("git"),
            "ffmpeg_system": ffmpeg,
            "ffmpeg_bundled": bundled_ffmpeg,
        },
        "packages": packages,
        "gpu": gpu_info(),
        "disk": disk_info(),
        "internet": internet_ok(),
        "commands": {
            "winget": bool(shutil.which("winget")),
            "brew": bool(shutil.which("brew")),
            "apt": bool(shutil.which("apt-get")),
            "dnf": bool(shutil.which("dnf")),
            "pacman": bool(shutil.which("pacman")),
        },
    }


def print_report(report: dict[str, object]) -> None:
    platform_info = report["platform"]
    hardware = report["gpu"]
    disk = report["disk"]
    tools = report["tools"]
    packages = report["packages"]

    print("\nEnvironment summary")
    print("-------------------")
    print(
        f"OS: {platform_info['system']} {platform_info['release']} "
        f"({platform_info['machine']})"
    )
    print(f"Python: {platform_info['python']} ({platform_info['python_executable']})")
    print(f"Virtual environment: {platform_info['virtual_environment']}")
    print(f"Internet: {'yes' if report['internet'] else 'no or unreachable'}")
    print(
        f"Disk free: {disk.get('free_gb', 'unknown')} GB "
        f"of {disk.get('total_gb', 'unknown')} GB"
    )
    gpu_lines = hardware.get("gpus", [])
    print(f"GPU: {'; '.join(gpu_lines) if gpu_lines else 'not detected'}")
    print(f"FFmpeg: {tools['ffmpeg_system'] or tools['ffmpeg_bundled'] or 'missing'}")
    print(
        "Packages: "
        + ", ".join(
            f"{name}={'ok' if path else 'missing'}"
            for name, path in packages.items()
        )
    )


def rerun_current_process() -> None:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    args = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    result = subprocess.run(args, env=env, check=False)
    raise SystemExit(result.returncode)


def run_logged(
    command: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[bootstrap] Running: {' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n$ {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            process.stdout.close()


def pip_install(
    packages: list[str],
    log_path: Path,
    optional: bool = False,
) -> bool:
    if not packages:
        return True
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--timeout",
        "60",
        *packages,
    ]
    code = run_logged(command, log_path)
    if code == 0:
        return True
    message = f"[bootstrap] Package installation failed: {' '.join(packages)}"
    if optional:
        print(message + " (optional; continuing)")
        return False
    print(message)
    print(f"[bootstrap] See log: {log_path}")
    raise RuntimeError("A required dependency could not be installed")


def cuda_library_dirs() -> list[Path]:
    candidates: list[Path] = []
    try:
        import site

        roots = [Path(p) for p in site.getsitepackages()]
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
    except Exception:
        roots = []

    for root in roots:
        nvidia = root / "nvidia"
        if not nvidia.exists():
            continue
        try:
            children = list(nvidia.iterdir())
        except OSError:
            continue
        for child in children:
            for name in ("bin", "lib", "lib64"):
                path = child / name
                if path.exists() and path not in candidates:
                    candidates.append(path)
    return candidates


def configure_library_path(env: dict[str, str]) -> None:
    paths = [str(path) for path in cuda_library_dirs()]
    if not paths:
        return
    if os.name == "nt":
        env["PATH"] = os.pathsep.join(paths + [env.get("PATH", "")])
    else:
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(paths + ([old] if old else []))


def cuda_usable(env: dict[str, str]) -> bool:
    code = (
        "import ctranslate2; "
        "n=ctranslate2.get_cuda_device_count(); "
        "types=ctranslate2.get_supported_compute_types('cuda') if n else set(); "
        "print(f'devices={n} compute_types={sorted(types)}')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[bootstrap] CUDA check failed: {exc}")
        return False
    output = (result.stdout or "") + (result.stderr or "")
    print(f"[bootstrap] CUDA check: {output.strip()}")
    device_match = re.search(r"devices=(\d+)", output)
    has_compute_types = bool(re.search(r"compute_types=\[[^\]]+\]", output))
    return (
        result.returncode == 0
        and device_match is not None
        and int(device_match.group(1)) > 0
        and has_compute_types
    )


def install_dependencies(no_cuda: bool, skip_install: bool) -> None:
    log_path = LOG_DIR / "bootstrap.log"
    problems = requirement_problems(REQUIREMENTS)
    if skip_install:
        print("[bootstrap] Dependency installation skipped")
        if problems:
            print("[bootstrap] Missing or invalid requirements:")
            for problem in problems:
                print(f"  - {problem}")
        return
    elif not problems:
        print("[bootstrap] Required Python packages are already installed")
    else:
        print("[bootstrap] Installing missing Python requirements:")
        for problem in problems:
            print(f"  - {problem}")
        print("[bootstrap] Updating pip tooling")
        pip_install(["--upgrade", "pip", "setuptools", "wheel"], log_path, optional=True)
        pip_install(["-r", str(REQUIREMENTS)], log_path, optional=False)
        remaining = requirement_problems(REQUIREMENTS)
        if remaining:
            raise RuntimeError("; ".join(remaining))

    env = os.environ.copy()
    configure_library_path(env)
    if no_cuda:
        print("[bootstrap] CUDA installation disabled by configuration")
        return

    hardware = gpu_info()
    if not hardware["gpus"]:
        print("[bootstrap] No NVIDIA GPU detected; CPU transcription will be used")
        return

    print("[bootstrap] NVIDIA GPU detected; checking CUDA runtime libraries")
    if cuda_usable(env):
        print("[bootstrap] CUDA runtime is already usable")
        return

    print(
        "[bootstrap] Installing CUDA 12 wheels for cuBLAS and cuDNN. "
        "This can take several minutes."
    )
    pip_install(
        ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"],
        log_path,
        optional=True,
    )
    env = os.environ.copy()
    configure_library_path(env)
    if not cuda_usable(env):
        print(
            "[bootstrap] CUDA remains unavailable. "
            "The extractor will automatically fall back to CPU."
        )


def finalize_report(report: dict[str, object], env: dict[str, str]) -> None:
    report["packages"] = base_environment_report()["packages"]
    report["gpu"]["ctranslate2_cuda_usable"] = cuda_usable(env)
    report["cuda_library_dirs"] = [str(path) for path in cuda_library_dirs()]
    atomic_write_text(
        REPORT_PATH,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"[bootstrap] Environment report written to: {REPORT_PATH}")


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Check the environment, install dependencies, and run the extractor."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only print diagnostics; do not install packages or run extraction.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip installation and use the current environment.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Do not attempt to install CUDA 12 runtime wheels.",
    )
    parser.add_argument(
        "--no-venv",
        action="store_true",
        help="Use the current interpreter instead of creating .venv.",
    )
    return parser.parse_known_args(argv)


def main() -> int:
    if sys.version_info < MIN_PYTHON:
        print(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; "
            f"current version is {platform.python_version()}."
        )
        return 2

    os.chdir(ROOT)
    args, forwarded = parse_args(sys.argv[1:])
    try:
        dotenv = read_dotenv(ROOT / ".env")
    except ValueError as exc:
        print(f"[bootstrap] {exc}")
        return 2
    try:
        auto_install_cuda = parse_bool(
            os.environ.get("AUTO_INSTALL_CUDA", "").strip()
            or dotenv.get("AUTO_INSTALL_CUDA"),
            True,
        )
    except ValueError as exc:
        print(f"[bootstrap] Invalid AUTO_INSTALL_CUDA setting: {exc}")
        return 2
    try:
        ensure_venv(args.no_venv)
    except Exception as exc:
        print(f"[bootstrap] Could not create or enter the virtual environment: {exc}")
        print("[bootstrap] On Linux, install python3-venv and retry, or use --no-venv.")
        return 2

    if args.check_only:
        report = base_environment_report()
        report["settings"] = {"auto_install_cuda": auto_install_cuda}
        print_report(report)
        try:
            atomic_write_text(
                REPORT_PATH,
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            )
            print(f"Report: {REPORT_PATH}")
        except OSError as exc:
            print(f"[bootstrap] Warning: could not write report: {exc}")
        return 0

    report = base_environment_report()
    report["settings"] = {"auto_install_cuda": auto_install_cuda}
    print_report(report)
    try:
        install_dependencies(args.no_cuda or not auto_install_cuda, args.skip_install)
    except Exception as exc:
        print(f"[bootstrap] Dependency setup failed: {exc}")
        return 3

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    configure_library_path(env)
    try:
        finalize_report(report, env)
    except Exception as exc:
        print(f"[bootstrap] Warning: could not write environment report: {exc}")

    command = [sys.executable, str(ROOT / "extract_video.py"), *forwarded]
    print("\n[bootstrap] Starting video extraction")
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    except OSError as exc:
        print(f"[bootstrap] Could not start the extractor: {exc}")
        return 4
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
