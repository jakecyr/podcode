#!/usr/bin/env python3
"""Deploy and manage open-weight LLMs on Runpod Pods.

This program intentionally shells out to Runpod's supported `runpodctl` CLI.
It does not send secrets to any service other than Runpod, and it avoids
depending on an undocumented GraphQL schema.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / ".runpod-manager-state.json"
VLLM_IMAGE = "vllm/vllm-openai:latest"


@dataclass(frozen=True)
class Model:
    key: str
    huggingface_id: str
    parameters: str
    min_vram_gb: int
    preferred_gpu: str
    gpu_count: int = 1
    quantization: str | None = None
    max_model_len: int | None = None
    note: str = ""
    recommended_volume_gb: int = 60


# VRAM is a deployment floor, not a training requirement. Values assume inference
# with vLLM and the listed quantization; context length and concurrency add memory.
MODELS: dict[str, Model] = {
    "gpt-oss-20b": Model("gpt-oss-20b", "openai/gpt-oss-20b", "20B", 16, "NVIDIA GeForce RTX 4090", quantization="mxfp4", note="24 GB is a comfortable single-GPU choice."),
    "gpt-oss-120b": Model("gpt-oss-120b", "openai/gpt-oss-120b", "120B", 80, "NVIDIA H100 80GB HBM3", quantization="mxfp4", max_model_len=16384, recommended_volume_gb=100, note="Use an 80 GB GPU; reduce context/concurrency if needed."),
    "deepseek-coder-v2-lite": Model("deepseek-coder-v2-lite", "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "16B", 24, "NVIDIA GeForce RTX 4090", note="A 24 GB GPU suits standard code inference."),
    "deepseek-coder-v2": Model("deepseek-coder-v2", "deepseek-ai/DeepSeek-Coder-V2-Instruct", "236B", 160, "NVIDIA H100 80GB HBM3", gpu_count=2, max_model_len=8192, recommended_volume_gb=200, note="Two 80 GB GPUs are a practical minimum for quantized serving."),
    "qwen2.5-coder-7b": Model("qwen2.5-coder-7b", "Qwen/Qwen2.5-Coder-7B-Instruct", "7B", 16, "NVIDIA GeForce RTX 3090", note="An economical 24 GB option."),
    "qwen2.5-coder-14b": Model("qwen2.5-coder-14b", "Qwen/Qwen2.5-Coder-14B-Instruct", "14B", 24, "NVIDIA GeForce RTX 4090", note="24 GB supports typical coding workloads."),
    "qwen2.5-coder-32b": Model("qwen2.5-coder-32b", "Qwen/Qwen2.5-Coder-32B-Instruct", "32B", 48, "NVIDIA A40", note="48 GB leaves useful room for KV cache."),
    "qwen3-coder-next": Model("qwen3-coder-next", "RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16", "80B MoE / 3B active", 48, "NVIDIA RTX A6000", max_model_len=32768, note="Default coding-agent choice; verified vLLM-compatible 4-bit checkpoint."),
    "qwen3.6-27b": Model("qwen3.6-27b", "Qwen/Qwen3.6-27B", "27B", 48, "NVIDIA RTX A6000", max_model_len=32768, note="Lower-cost general coding alternative; 48 GB is recommended."),
}


def load_dotenv() -> None:
    """Load .env from the current directory, then the source directory."""
    paths = [Path.cwd() / ".env", ROOT / ".env"]
    for path in paths:
        if not path.exists():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def require_ctl() -> str:
    binary = shutil.which("runpodctl")
    if not binary:
        binary = find_or_install_runpodctl()
    if not os.getenv("RUNPOD_API_KEY"):
        raise SystemExit("RUNPOD_API_KEY is missing. Copy .env.example to .env and set it, then run `runpodctl config --apiKey $RUNPOD_API_KEY` once.")
    return binary


def find_or_install_runpodctl() -> str:
    """Use a common local installation even if the shell has not added it to PATH."""
    for candidate in (
        Path("~/.local/bin/runpodctl").expanduser(),
        Path("~/bin/runpodctl").expanduser(),
        Path("/opt/homebrew/bin/runpodctl"),
        Path("/usr/local/bin/runpodctl"),
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return install_runpodctl()


def install_runpodctl() -> str:
    """Install Runpod's published CLI into the user's local bin directory."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    if system not in {"darwin", "linux"} or not arch:
        raise SystemExit(
            "runpodctl was not found and automatic installation is unsupported on "
            f"{platform.system()} {platform.machine()}. Install it from "
            "https://docs.runpod.io/runpodctl."
        )

    destination_dir = Path(os.getenv("RUNPODCTL_BIN_DIR", "~/.local/bin")).expanduser()
    destination = destination_dir / "runpodctl"
    url = f"https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-{system}-{arch}"
    print(f"runpodctl was not found; installing it to {destination}...", file=sys.stderr)
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            if response.status != 200:
                raise urllib.error.HTTPError(url, response.status, "download failed", response.headers, None)
            with tempfile.NamedTemporaryFile(dir=destination_dir, delete=False) as tmp:
                shutil.copyfileobj(response, tmp)
                temporary = Path(tmp.name)
        temporary.chmod(0o755)
        temporary.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        raise SystemExit(
            "Could not install runpodctl automatically. Install it from "
            f"https://docs.runpod.io/runpodctl ({error})"
        ) from error

    return str(destination)


def ctl(arguments: Iterable[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    binary = require_ctl()
    env = os.environ.copy()
    # Recent runpodctl versions honor RUNPOD_API_KEY; config is still recommended.
    return subprocess.run([binary, *arguments], text=True, env=env, check=check, capture_output=capture)


def ctl_output(arguments: Iterable[str]) -> str:
    binary = require_ctl()
    result = subprocess.run([binary, *arguments], text=True, env=os.environ.copy(), capture_output=True)
    if result.returncode:
        raise SystemExit(result.stderr.strip() or "Runpod could not retrieve live GPU pricing.")
    return result.stdout


def require_timer_support(stop_after: str | None, terminate_after: str | None) -> None:
    """Avoid accepting a cost-control option the installed CLI cannot submit."""
    if not (stop_after or terminate_after):
        return
    help_result = subprocess.run(
        [require_ctl(), "pod", "create", "--help"], text=True, capture_output=True, check=False
    )
    available = help_result.stdout + help_result.stderr
    unsupported = []
    if stop_after and "--stop-after" not in available:
        unsupported.append("--stop-after")
    if terminate_after and "--terminate-after" not in available:
        unsupported.append("--terminate-after")
    if unsupported:
        raise SystemExit(
            f"The installed runpodctl does not support {', '.join(unsupported)} on `pod create`. "
            "No Pod was created. Re-run without that option and delete the Pod explicitly with "
            "`podcode destroy POD_ID` when finished."
        )


def duration_seconds(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError("use a duration such as 30s, 10m, or 1h")
    return float(match.group(1)) * {"s": 1, "m": 60, "h": 3600}[match.group(2)]


def vllm_loading_stage(pod_id: str) -> str:
    """Summarize the most recent useful vLLM startup state without printing logs."""
    result = subprocess.run(
        [require_ctl(), "pod", "logs", pod_id, "--tail", "50", "--source", "container"],
        text=True, capture_output=True, check=False, env=os.environ.copy(),
    )
    lines: list[str] = []
    for raw in result.stdout.splitlines():
        try:
            lines.append(str(json.loads(raw).get("line", "")))
        except json.JSONDecodeError:
            continue
    recent = "\n".join(lines).lower()
    latest_error = max((index for index, line in enumerate(lines) if any(word in line.lower() for word in ("out of memory", "traceback", "runtimeerror", "exception"))), default=-1)
    latest_load = max((index for index, line in enumerate(lines) if "loading model from scratch" in line.lower() or "loading model weights" in line.lower()), default=-1)
    if latest_error > latest_load:
        return "vLLM reported a startup error; inspect `podcode logs " + pod_id + "`"
    if "api server" in recent and "started" in recent:
        return "vLLM API server is starting"
    if "loading model from scratch" in recent or "loading model weights" in recent:
        return "GPU is loading model weights"
    if "resolved architecture" in recent or "initializing a v1 llm engine" in recent:
        return "vLLM is initializing the GPU engine"
    return "Pod is starting"


def wait_for_vllm(pod_id: str, timeout_seconds: float) -> None:
    """Show deploy progress until the public vLLM health endpoint responds."""
    url = f"https://{pod_id}-8000.proxy.runpod.net/health"
    started = time.monotonic()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    index = 0
    stage = "Pod is starting"
    print("\nWaiting for vLLM to load model weights", end="", flush=True)
    while time.monotonic() - started < timeout_seconds:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                if response.status == 200:
                    elapsed = int(time.monotonic() - started)
                    print(f"\r✓ vLLM is ready after {elapsed // 60}m {elapsed % 60:02d}s.{' ' * 20}")
                    return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            pass
        elapsed = int(time.monotonic() - started)
        if index % 5 == 0:
            stage = vllm_loading_stage(pod_id)
        print(f"\r{frames[index % len(frames)]} {stage} ({elapsed // 60}m {elapsed % 60:02d}s)", end="", flush=True)
        index += 1
        time.sleep(5)
    print()
    raise SystemExit(
        f"vLLM was not ready after {int(timeout_seconds // 60)} minute(s). The Pod may still be loading; "
        f"inspect it with `runpodctl pod logs {pod_id} --follow` or delete it with `podcode destroy {pod_id}`."
    )


def show_quote(model: Model, gpu: str, count: int, cloud: str, hours: float) -> None:
    """Print live Runpod pricing relevant to this exact deployment selection."""
    listing = ctl_output(["gpu", "list"])
    matching = [line.strip() for line in listing.splitlines() if gpu.lower() in line.lower()]
    print(f"\nCost preflight — {model.key}: {count}x {gpu}, {cloud} cloud")
    if not matching:
        print("Runpod returned no exact matching GPU row. Choose an available GPU with `podcode gpus` before continuing.")
        return
    for line in matching:
        print("  Live Runpod row:", line)
    prices = [float(value) for value in re.findall(r"(?:\$|USD\s*)(\d+(?:\.\d+)?)", " ".join(matching), flags=re.I)]
    if prices:
        # runpodctl commonly prints Secure then Community price; show candidates
        # rather than silently claiming an ambiguous table column is authoritative.
        per_gpu = prices[0] if cloud == "SECURE" or len(prices) == 1 else prices[-1]
        hourly = per_gpu * count
        print(f"  Estimated GPU cost: ${hourly:.2f}/hour; about ${hourly * hours:.2f} for {hours:g} hour(s).")
    else:
        print("  Price format could not be parsed; the live row above is authoritative.")


def confirm(word: str, message: str) -> None:
    print(message)
    try:
        answer = input(f"Type {word} to continue: ").strip()
    except EOFError:
        answer = ""
    if answer != word:
        raise SystemExit("Cancelled. No Runpod resources were created or deleted.")


def state() -> dict:
    if not STATE_PATH.exists():
        return {"deployments": {}}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {"deployments": {}}


def save_state(data: dict) -> None:
    STATE_PATH.write_text(json.dumps(data, indent=2) + "\n")


def choose_model(key: str) -> Model:
    if key not in MODELS:
        raise SystemExit(f"Unknown model '{key}'. Run `podcode models` to see supported presets.")
    return MODELS[key]


def cmd_models(_: argparse.Namespace) -> None:
    print("MODEL                     PARAMS  MIN VRAM  RECOMMENDED GPU                 NOTES")
    for model in MODELS.values():
        gpu = f"{model.gpu_count}x {model.preferred_gpu}" if model.gpu_count > 1 else model.preferred_gpu
        print(f"{model.key:<25} {model.parameters:<7} {model.min_vram_gb:>4} GB  {gpu:<31} {model.note}")


def cmd_gpus(_: argparse.Namespace) -> None:
    """Show real-time GPU availability and hourly price directly from Runpod."""
    ctl(["gpu", "list"])


def cmd_deploy(args: argparse.Namespace) -> None:
    model = choose_model(args.model)
    gpu = args.gpu or model.preferred_gpu
    count = args.gpu_count or model.gpu_count
    cloud = args.cloud_type or os.getenv("RUNPOD_CLOUD_TYPE", "SECURE")
    volume_gb = args.volume_gb or int(os.getenv("RUNPOD_VOLUME_GB", str(model.recommended_volume_gb)))
    container_gb = args.container_disk_gb or int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "30"))
    mount = os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace")
    cache = "/root/.cache/huggingface" if args.ephemeral else os.getenv("RUNPOD_MODEL_CACHE", "/workspace/huggingface")
    name = args.name or f"llm-{model.key}"
    require_timer_support(args.stop_after, args.terminate_after)
    if not getattr(args, "confirmed", False):
        show_quote(model, gpu, count, cloud, args.estimate_hours)
        confirm("DEPLOY", f"This will create a billable Runpod Pod named '{name}' and download {model.huggingface_id}.")
    vllm_key = os.getenv("RUNPOD_VLLM_API_KEY")
    if not vllm_key:
        raise SystemExit("RUNPOD_VLLM_API_KEY is required to protect the public vLLM endpoint. Set it in .env.")

    # vllm/vllm-openai has a vLLM entrypoint, so these are its arguments rather
    # than a nested `vllm serve` command.
    serve = ["--model", model.huggingface_id, "--host", "0.0.0.0", "--port", "8000", "--download-dir", cache]
    if model.quantization:
        serve += ["--quantization", model.quantization]
    if model.max_model_len:
        serve += ["--max-model-len", str(model.max_model_len)]
    if count > 1:
        serve += ["--tensor-parallel-size", str(count)]
    serve += ["--api-key", vllm_key]
    docker_args = " ".join(serve)
    env_vars = {"HF_HOME": cache, "HUGGINGFACE_HUB_CACHE": cache}
    if os.getenv("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]

    print(textwrap.dedent(f"""\
        Deploying {model.huggingface_id}
          GPU: {count}x {gpu} (model floor: {model.min_vram_gb} GB total)
          Storage: {"ephemeral container disk" if args.ephemeral else f"{volume_gb} GB persistent volume mounted at {mount}"}
          Endpoint after startup: https://<pod-id>-8000.proxy.runpod.net/v1
        Check exact live availability and hourly price first with: podcode gpus
    """))
    command = ["pod", "create", "--name", name, "--image", VLLM_IMAGE, "--gpu-id", gpu,
               "--gpu-count", str(count), "--cloud-type", cloud, "--container-disk-in-gb", str(container_gb),
               "--volume-mount-path", mount, "--ports", "8000/http,22/tcp",
               "--env", json.dumps(env_vars), "--docker-args", docker_args]
    if args.network_volume_id:
        command += ["--network-volume-id", args.network_volume_id]
    elif not args.ephemeral:
        command += ["--volume-in-gb", str(volume_gb)]
    if args.stop_after:
        command += ["--stop-after", args.stop_after]
    if args.terminate_after:
        command += ["--terminate-after", args.terminate_after]
    result = ctl(command, capture=True, check=False)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or "runpodctl could not create the Pod."
        raise SystemExit(message)
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    print("\nPod submitted. Run `podcode status` to get its ID and endpoint. Model weights download on first boot and remain on the Pod volume.")
    # runpodctl output varies by version; accept its explicit id fields only.
    match = re.search(r'(?im)(?:pod\s*(?:id)?|"id")\s*[:=]\s*["\']?([a-z0-9]{6,})', result.stdout)
    pod_id = match.group(1) if match else None
    if pod_id:
        wait_for_vllm(pod_id, args.wait_timeout)
    return pod_id


def cmd_status(_: argparse.Namespace) -> None:
    ctl(["pod", "list", "--all"])
    print("\nTip: `runpodctl pod get POD_ID` shows SSH and proxy connection details.")


def cmd_wait(args: argparse.Namespace) -> None:
    """Poll a Pod's vLLM health endpoint and concise GPU-loading stage."""
    wait_for_vllm(args.pod_id, args.timeout)


def cmd_logs(args: argparse.Namespace) -> None:
    """Pass through container logs for detailed startup diagnosis."""
    ctl(["pod", "logs", args.pod_id, "--follow"])


def cmd_control(args: argparse.Namespace) -> None:
    ctl(["pod", args.action, args.pod_id])


def cmd_destroy(args: argparse.Namespace) -> None:
    print(f"Deleting Pod {args.pod_id}. This is permanent; Pod-attached volume data will not be recoverable.")
    ctl(["pod", "delete", args.pod_id])
    if args.delete_network_volume:
        print(f"Deleting network volume {args.delete_network_volume}.")
        ctl(["network-volume", "delete", args.delete_network_volume])


def cmd_swap(args: argparse.Namespace) -> None:
    """Replace a Pod with one serving another model.

    A running vLLM server cannot safely hot-reload a model or change GPUs, so a
    replacement Pod is the reliable swap mechanism. A network volume retains
    the model cache across the replacement.
    """
    model = choose_model(args.model)
    gpu = args.gpu or model.preferred_gpu
    count = args.gpu_count or model.gpu_count
    cloud = args.cloud_type or os.getenv("RUNPOD_CLOUD_TYPE", "SECURE")
    show_quote(model, gpu, count, cloud, args.estimate_hours)
    confirm("REPLACE", f"This permanently deletes Pod {args.pod_id} and creates a billable replacement serving {model.huggingface_id}.")
    print(f"Replacing Pod {args.pod_id} with {args.model}.")
    if not args.network_volume_id:
        print("Warning: no --network-volume-id was supplied; the old Pod-attached model cache will be deleted.")
    ctl(["pod", "delete", args.pod_id])
    args.confirmed = True
    cmd_deploy(args)


def cmd_opencode(args: argparse.Namespace) -> None:
    """Write a project-local OpenCode custom-provider config without overwriting one."""
    model = choose_model(args.model)
    target = Path(args.output).resolve()
    if target.exists() and not args.force:
        raise SystemExit(f"{target} already exists; use --force to replace it, or choose --output.")
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"runpod/{model.huggingface_id}",
        "provider": {"runpod": {"npm": "@ai-sdk/openai-compatible", "name": "Runpod vLLM", "options": {
            "baseURL": f"https://{args.pod_id}-8000.proxy.runpod.net/v1", "apiKey": "{env:RUNPOD_VLLM_API_KEY}"},
            "models": {model.huggingface_id: {"name": model.key, "limit": {
                "context": model.max_model_len or 32768, "output": 8192}}}}},
    }
    target.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote {target}\nStart OpenCode in this directory; default model: runpod/{model.huggingface_id}")


def cmd_up(args: argparse.Namespace) -> None:
    """Single-command path: deploy the model, then create local OpenCode config."""
    pod_id = cmd_deploy(args)
    if not pod_id:
        print("\nPod was created, but this runpodctl version did not return a recognizable Pod ID.")
        print(f"Run `podcode status`, then: podcode opencode-config POD_ID {args.model}")
        return
    config_args = argparse.Namespace(pod_id=pod_id, model=args.model, output=args.opencode_output, force=args.force_opencode_config)
    cmd_opencode(config_args)


def cmd_volume(args: argparse.Namespace) -> None:
    # Pass through exactly so this stays compatible with new runpodctl volume flags.
    ctl(["network-volume", *args.runpodctl_args])


def cmd_usage(_: argparse.Namespace) -> None:
    print("Current Pods (their live hourly rate is shown by `podcode gpus`):")
    ctl(["pod", "list", "--all"])
    print(textwrap.dedent("""
        Runpod charges based on the actual selected GPU, cloud tier, storage, and
        running time. Use `podcode gpus` before deployment for the current
        hourly rate, and Runpod Console → Billing for the authoritative accrued
        usage, invoices, credits, and payment history. This command deliberately
        does not invent a cost from a stale static price table.
    """))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="podcode", description="Deploy and operate open-weight coding models on Runpod.")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("models", help="list model presets and GPU recommendations").set_defaults(func=cmd_models)
    sub.add_parser("gpus", help="show live Runpod GPU availability and prices").set_defaults(func=cmd_gpus)
    sub.add_parser("status", help="list all Pods").set_defaults(func=cmd_status)
    wait = sub.add_parser("wait", help="poll vLLM readiness and GPU model-loading progress")
    wait.add_argument("pod_id")
    wait.add_argument("--timeout", type=duration_seconds, default=1800, metavar="DURATION", help="maximum wait; default: 30m")
    wait.set_defaults(func=cmd_wait)
    logs = sub.add_parser("logs", help="stream detailed container logs for a Pod")
    logs.add_argument("pod_id")
    logs.set_defaults(func=cmd_logs)
    sub.add_parser("usage", help="show Pods and billing guidance").set_defaults(func=cmd_usage)

    def add_deploy_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("model", choices=sorted(MODELS))
        target.add_argument("--name")
        target.add_argument("--gpu", help="override the recommended Runpod GPU name")
        target.add_argument("--gpu-count", type=int, help="override number of GPUs")
        target.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"))
        target.add_argument("--volume-gb", type=int, help="Pod-attached volume size (ignored with --network-volume-id)")
        target.add_argument("--network-volume-id", help="reusable Runpod network volume; preserves the model cache across swaps")
        target.add_argument("--ephemeral", action="store_true", help="do not create a persistent volume; model cache is discarded with the Pod")
        target.add_argument("--container-disk-gb", type=int)
        target.add_argument("--stop-after", help="auto-stop duration, e.g. 8h")
        target.add_argument("--terminate-after", help="auto-delete duration, e.g. 24h")
        target.add_argument("--estimate-hours", type=float, default=1, help="hours used for the preflight cost estimate (default: 1)")
        target.add_argument("--wait-timeout", type=duration_seconds, default=1800, metavar="DURATION", help="wait for vLLM readiness; default: 30m")

    deploy = sub.add_parser("deploy", help="create a vLLM Pod and load a model")
    add_deploy_options(deploy)
    deploy.set_defaults(func=cmd_deploy)

    up = sub.add_parser("up", help="single command: cost preflight, deploy, then configure local OpenCode")
    add_deploy_options(up)
    up.add_argument("--opencode-output", default="opencode.json", help="local OpenCode config path")
    up.add_argument("--force-opencode-config", action="store_true", help="replace the generated OpenCode config if it exists")
    up.set_defaults(func=cmd_up)

    swap = sub.add_parser("swap", help="delete a Pod and replace it with another model")
    swap.add_argument("pod_id", help="Pod to replace")
    add_deploy_options(swap)
    swap.set_defaults(func=cmd_swap)

    opencode = sub.add_parser("opencode-config", help="write project-local OpenCode config for a Pod")
    opencode.add_argument("pod_id")
    opencode.add_argument("model", choices=sorted(MODELS))
    opencode.add_argument("--output", default="opencode.json")
    opencode.add_argument("--force", action="store_true")
    opencode.set_defaults(func=cmd_opencode)

    for action in ("start", "stop", "restart"):
        control = sub.add_parser(action, help=f"{action} a Pod")
        control.add_argument("pod_id")
        control.set_defaults(func=cmd_control, action=action)
    destroy = sub.add_parser("destroy", help="delete a Pod, optionally also a network volume")
    destroy.add_argument("pod_id")
    destroy.add_argument("--delete-network-volume", metavar="VOLUME_ID", help="also permanently delete this network volume")
    destroy.set_defaults(func=cmd_destroy)
    volume = sub.add_parser("volume", help="pass through a network-volume action to runpodctl")
    volume.add_argument("runpodctl_args", nargs=argparse.REMAINDER, help="e.g. list, create ..., delete VOLUME_ID")
    volume.set_defaults(func=cmd_volume)
    return p


def main() -> None:
    load_dotenv()
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
