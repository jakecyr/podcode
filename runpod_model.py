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
# Pin the image used during validation; `latest` can change its CLI or runtime
# behavior without a podcode release.
VLLM_IMAGE = "vllm/vllm-openai:v0.30.0"
DEFAULT_MODEL = "qwen3-coder-next"


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
    recommended_container_disk_gb: int = 30
    max_num_seqs: int | None = None
    gpu_memory_utilization: float | None = None
    tool_call_parser: str | None = None


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
    "qwen3-coder-next": Model("qwen3-coder-next", "RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16", "80B MoE / 3B active", 48, "NVIDIA RTX A6000", max_model_len=32768, note="Default coding-agent choice; verified vLLM-compatible 4-bit checkpoint.", recommended_container_disk_gb=100, max_num_seqs=8, gpu_memory_utilization=0.98, tool_call_parser="qwen3_coder"),
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


def redact_secrets(text: str) -> str:
    """Prevent Pod-create responses from echoing local API tokens to the terminal."""
    for name in ("RUNPOD_API_KEY", "RUNPOD_VLLM_API_KEY", "HF_TOKEN"):
        value = os.getenv(name)
        if value:
            text = text.replace(value, "***")
    return re.sub(r"\bhf_[A-Za-z0-9_-]+\b", "***", text)


def print_pod_submission(output: str) -> None:
    """Render Runpod's JSON create response without exposing noisy metadata."""
    try:
        pod = json.loads(output)
    except json.JSONDecodeError:
        print(redact_secrets(output), end="" if output.endswith("\n") else "\n")
        return
    if not isinstance(pod, dict):
        print(redact_secrets(output), end="" if output.endswith("\n") else "\n")
        return
    machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
    print(color("\n✓ Pod submitted", "32"))
    for label, value in (
        ("Pod ID", pod.get("id")),
        ("Status", pod.get("desiredStatus")),
        ("GPU", f"{pod.get('gpuCount', 1)}x {machine.get('gpuDisplayName', pod.get('gpuDisplayName', 'requested GPU'))}"),
        ("Location", machine.get("location")),
        ("Rate", f"${pod['costPerHr']}/hour" if pod.get("costPerHr") is not None else None),
        ("Container disk", f"{pod['containerDiskInGb']} GB" if pod.get("containerDiskInGb") is not None else None),
    ):
        if value is not None:
            print(f"  {color(label + ':', '2')} {value}")


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


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def concise_log_line(source: str, line: str) -> tuple[str, str] | None:
    """Return a readable startup event, omitting banners and Python stack frames."""
    text = line.strip().replace("\r", "")
    lowered = text.lower()
    if source == "system" and "start container" in lowered:
        return ("info", "Container started")
    if "resolved architecture" in lowered:
        return ("info", "Model architecture recognized")
    if "initializing a v1 llm engine" in lowered:
        return ("info", "Initializing GPU inference engine")
    if "loading model from scratch" in lowered:
        return ("load", "Loading model weights onto GPU")
    if "time spent downloading weights" in lowered:
        return ("info", "Model weights downloaded")
    if "flashattention version" in lowered:
        return ("info", "GPU attention kernels initialized")
    if "torch.compile took" in lowered:
        return ("info", "GPU model compilation finished")
    if "initial profiling/warmup run took" in lowered:
        return ("info", "GPU warmup finished")
    if "application startup complete" in lowered or "uvicorn running on" in lowered:
        return ("ready", "vLLM API is ready")
    if "not enough free disk space" in lowered:
        return ("error", "Insufficient container disk while downloading model")
    if "background writer channel closed" in lowered:
        return ("error", "Model download failed; Hugging Face cache writer stopped")
    if "engine core initialization failed" in lowered:
        return ("error", "vLLM engine failed to start")
    return None


def vllm_loading_stage(pod_id: str) -> str:
    """Summarize the most recent useful vLLM startup state without printing logs."""
    result = subprocess.run(
        [require_ctl(), "pod", "logs", pod_id, "--tail", "200", "--source", "container"],
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
    if "application startup complete" in recent or "uvicorn running on" in recent:
        return "vLLM API is ready"
    if "starting vllm server" in recent or "started server process" in recent:
        return "vLLM API server is starting"
    if "initial profiling/warmup run" in recent or "warming up" in recent or "capturing cuda graphs" in recent:
        return "GPU is warming kernels and capturing graphs"
    if "torch.compile" in recent or "compiling a graph" in recent:
        return "GPU is compiling the model"
    if "loading safetensors checkpoint shards" in recent or "loading weights took" in recent:
        return "GPU is loading model weights"
    if "downloading weights" in recent:
        return "Downloading model weights"
    if "loading model from scratch" in recent or "loading model weights" in recent:
        return "GPU is loading model weights"
    if "resolved architecture" in recent or "initializing a v1 llm engine" in recent:
        return "vLLM is initializing the GPU engine"
    return "Pod is starting"


def wait_for_vllm(pod_id: str, timeout_seconds: float) -> None:
    """Show deploy progress until the public vLLM health endpoint responds."""
    vllm_key = os.getenv("RUNPOD_VLLM_API_KEY")
    if not vllm_key:
        raise SystemExit("RUNPOD_VLLM_API_KEY is required to check vLLM readiness.")
    # Probe an authenticated OpenAI-compatible endpoint.  /health is also
    # protected when vLLM is started with --api-key, so an unauthenticated
    # health probe would wait forever even after the server has started.
    url = f"https://{pod_id}-8000.proxy.runpod.net/v1/models"
    started = time.monotonic()
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    index = 0
    stage = "Pod is starting"
    print("\nWaiting for vLLM to load model weights", end="", flush=True)
    try:
        while time.monotonic() - started < timeout_seconds:
            try:
                # Runpod's Cloudflare proxy rejects Python's default urllib
                # user agent with error 1010 even when vLLM is healthy.
                request = urllib.request.Request(url, headers={
                    "Authorization": f"Bearer {vllm_key}",
                    "User-Agent": "podcode/0.1",
                })
                with urllib.request.urlopen(request, timeout=10) as response:
                    if response.status == 200:
                        elapsed = int(time.monotonic() - started)
                        print(f"\r✓ vLLM is ready after {elapsed // 60}m {elapsed % 60:02d}s.{' ' * 20}")
                        return
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                pass
            elapsed = int(time.monotonic() - started)
            if index % 5 == 0:
                stage = vllm_loading_stage(pod_id)
            print(f"\r\033[2K{color(frames[index % len(frames)], '36')} {stage} ({elapsed // 60}m {elapsed % 60:02d}s)", end="", flush=True)
            index += 1
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nReadiness monitor cancelled; the Pod was left running.")
        return
    print()
    raise SystemExit(
        f"vLLM was not ready after {int(timeout_seconds // 60)} minute(s). The Pod may still be loading; "
        f"inspect it with `runpodctl pod logs {pod_id} --follow` or delete it with `podcode destroy {pod_id}`."
    )


def show_quote(model: Model, gpu: str, count: int, cloud: str, hours: float) -> None:
    """Render a compact deployment card from Runpod's live GPU inventory."""
    try:
        rows = json.loads(ctl_output(["gpu", "list"]))
    except json.JSONDecodeError:
        rows = []
    matches = [row for row in rows if isinstance(row, dict) and row.get("gpuId", "").lower() == gpu.lower()] if isinstance(rows, list) else []
    price_field = "securePricePerHr" if cloud.upper() == "SECURE" else "communityPricePerHr"
    selected = next((row for row in matches if isinstance(row.get(price_field), (int, float))), matches[0] if matches else {})
    price = selected.get(price_field) if isinstance(selected, dict) else None
    stock = selected.get("stockStatus") if isinstance(selected, dict) else None

    print(color(f"\n╭─ Deploy {model.key}", "36"))
    print(f"│  Model     {model.huggingface_id}")
    print(f"│  Compute   {count}× {gpu} · {cloud.upper()} cloud")
    if stock:
        print(f"│  Availability  {color(str(stock).upper(), '33' if str(stock).lower() != 'available' else '32')}")
    if isinstance(price, (int, float)):
        hourly = price * count
        print(f"│  Live rate ${hourly:.2f}/hour")
        print(f"│  Estimate  ${hourly * hours:.2f} for {hours:g} hour(s)")
    else:
        print("│  Live rate unavailable · check `podcode gpus`")
    print(color("╰────────────────────────────────────────", "36"))


def fallback_gpu(model: Model, requested_gpu: str, cloud: str) -> str | None:
    """Return the lowest-cost currently listed compatible GPU, if one exists."""
    try:
        rows = json.loads(ctl_output(["gpu", "list"]))
    except json.JSONDecodeError:
        return None
    if not isinstance(rows, list):
        return None
    cloud_key = "secureCloud" if cloud.upper() == "SECURE" else "communityCloud"
    price_key = "securePricePerHr" if cloud.upper() == "SECURE" else "communityPricePerHr"
    candidates = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("available")
        and row.get(cloud_key)
        and isinstance(row.get("memoryInGb"), (int, float))
        and row["memoryInGb"] >= model.min_vram_gb
        and row.get("gpuId") != requested_gpu
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (
        float(row[price_key]) if isinstance(row.get(price_key), (int, float)) else float("inf"),
        str(row.get("gpuId", "")),
    ))
    candidate = candidates[0].get("gpuId")
    return str(candidate) if candidate else None


def confirm(word: str, message: str) -> None:
    print(f"\n{color('!', '33')} {message}")
    try:
        answer = input(f"Type {color(word, '33')} to continue › ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. No Runpod resources were created or deleted.")
        raise SystemExit(0)
    if answer != word:
        raise SystemExit("Cancelled. No Runpod resources were created or deleted.")


def confirm_yes(message: str) -> None:
    """Continue only for an explicit yes; every other response cancels."""
    print(f"\n{color('!', '33')} {message}")
    try:
        answer = input(f"Continue? {color('[y/N]', '33')} › ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled. No Runpod resources were created or deleted.")
        raise SystemExit(0)
    if answer not in {"y", "yes"}:
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
    print(color("\nAvailable model presets", "36"))
    print(color("MODEL                     PARAMS  MIN VRAM  RECOMMENDED GPU                 NOTES", "2"))
    for model in MODELS.values():
        gpu = f"{model.gpu_count}x {model.preferred_gpu}" if model.gpu_count > 1 else model.preferred_gpu
        print(f"{model.key:<25} {model.parameters:<7} {model.min_vram_gb:>4} GB  {gpu:<31} {model.note}")


def list_pods() -> list[dict]:
    """Return Runpod's Pod list with a consistent error for malformed output."""
    try:
        pods = json.loads(ctl_output(["pod", "list", "--all"]))
    except json.JSONDecodeError as error:
        raise SystemExit("Runpod returned an unreadable Pod list.") from error
    if not isinstance(pods, list):
        raise SystemExit("Runpod returned an unexpected Pod list.")
    return [pod for pod in pods if isinstance(pod, dict)]


def print_pods(pods: list[dict], *, heading: str = "Runpod Pods") -> None:
    """Render Pod inventory without dumping Runpod's raw JSON."""
    print(color(f"\n{heading}", "36"))
    if not pods:
        print("No Pods found.")
        return
    print(color("ID                    STATUS      GPU                              NAME", "2"))
    for pod in pods:
        pod_id = str(pod.get("id") or "—")
        status = str(pod.get("desiredStatus") or pod.get("status") or "unknown").upper()
        machine = pod.get("machine") if isinstance(pod.get("machine"), dict) else {}
        gpu = str(machine.get("gpuDisplayName") or pod.get("gpuDisplayName") or "—")
        count = pod.get("gpuCount")
        if isinstance(count, int) and count > 1:
            gpu = f"{count}x {gpu}"
        name = str(pod.get("name") or "—")
        shade = "32" if status == "RUNNING" else "33" if status in {"EXITED", "STOPPED"} else "2"
        print(f"{pod_id:<21} {color(f'{status:<11}', shade)} {gpu[:32]:<32} {name}")


def cmd_gpus(args: argparse.Namespace) -> None:
    """Show the useful portion of Runpod's live GPU inventory."""
    raw = ctl_output(["gpu", "list"])
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        print(raw, end="" if raw.endswith("\n") else "\n")
        return
    if not isinstance(rows, list):
        print(raw, end="" if raw.endswith("\n") else "\n")
        return
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    def number(row: dict, key: str) -> float:
        value = row.get(key)
        return float(value) if isinstance(value, (int, float)) else float("inf")

    rows = [row for row in rows if isinstance(row, dict) and row.get("available")]
    rows.sort(key=lambda row: (number(row, "securePricePerHr"), str(row.get("displayName", ""))))
    print(color("\nLive Runpod GPU availability", "36"))
    print(color("GPU                              VRAM   SECURE    COMMUNITY  STOCK  AVAILABLE LOCATIONS", "2"))
    if not rows:
        print("No currently available GPU types were returned. Try again shortly.")
        return
    for row in rows:
        name = str(row.get("displayName") or row.get("gpuId") or "Unknown GPU")[:32]
        memory = row.get("memoryInGb")
        secure = row.get("securePricePerHr")
        community = row.get("communityPricePerHr")
        stock = str(row.get("stockStatus") or "unknown")
        locations = [str(item.get("dataCenterId")) for item in row.get("dataCenterAvailability", [])
                     if isinstance(item, dict) and str(item.get("stockStatus", "")).lower() not in {"none", "", "unavailable"}]
        location_text = ", ".join(locations[:3]) or "—"
        if len(locations) > 3:
            location_text += f" +{len(locations) - 3}"
        secure_text = f"${secure:.2f}/h" if isinstance(secure, (int, float)) else "—"
        community_text = f"${community:.2f}/h" if isinstance(community, (int, float)) else "—"
        stock_color = "32" if stock.lower() == "available" else "33"
        print(f"{name:<32} {str(memory or '—') + ' GB':>6}  {secure_text:>8}  {community_text:>9}  "
              f"{color(f'{stock.upper():<5}', stock_color)}  {location_text}")
    print(color("\nRates are per GPU per hour. Availability can change before Pod creation.", "2"))


def cmd_deploy(args: argparse.Namespace) -> None:
    model = choose_model(args.model)
    gpu = args.gpu or model.preferred_gpu
    count = args.gpu_count or model.gpu_count
    cloud = args.cloud_type or os.getenv("RUNPOD_CLOUD_TYPE", "SECURE")
    volume_gb = args.volume_gb or int(os.getenv("RUNPOD_VOLUME_GB", str(model.recommended_volume_gb)))
    requested_container_gb = args.container_disk_gb or int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "0"))
    container_gb = max(requested_container_gb, model.recommended_container_disk_gb)
    mount = os.getenv("RUNPOD_VOLUME_MOUNT_PATH", "/workspace")
    cache = "/root/.cache/huggingface" if args.ephemeral else os.getenv("RUNPOD_MODEL_CACHE", "/workspace/huggingface")
    name = args.name or f"llm-{model.key}"
    require_timer_support(args.stop_after, args.terminate_after)
    if not getattr(args, "confirmed", False):
        show_quote(model, gpu, count, cloud, args.estimate_hours)
        message = f"This will create a billable Runpod Pod named '{name}' and download {model.huggingface_id}."
        if getattr(args, "yes_no_confirm", False):
            confirm_yes(message)
        else:
            confirm("DEPLOY", message)
    vllm_key = os.getenv("RUNPOD_VLLM_API_KEY")
    if not vllm_key:
        raise SystemExit("RUNPOD_VLLM_API_KEY is required to protect the public vLLM endpoint. Set it in .env.")

    # vllm/vllm-openai has a vLLM entrypoint, so these are its arguments rather
    # than a nested `vllm serve` command.
    serve = [model.huggingface_id, "--host", "0.0.0.0", "--port", "8000", "--download-dir", cache]
    if model.quantization:
        serve += ["--quantization", model.quantization]
    if model.max_model_len:
        serve += ["--max-model-len", str(model.max_model_len)]
    if model.max_num_seqs:
        serve += ["--max-num-seqs", str(model.max_num_seqs)]
    if model.gpu_memory_utilization:
        serve += ["--gpu-memory-utilization", str(model.gpu_memory_utilization)]
    if model.tool_call_parser:
        serve += ["--enable-auto-tool-choice", "--tool-call-parser", model.tool_call_parser]
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
          Storage: {f"{container_gb} GB ephemeral container disk" if args.ephemeral else f"{volume_gb} GB persistent volume mounted at {mount}"}
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
    unavailable = "no longer any instances available" in (result.stderr + result.stdout).lower()
    # Only the default choice falls back: an explicit --gpu is a user decision.
    if result.returncode and unavailable and args.gpu is None:
        replacement_gpu = fallback_gpu(model, gpu, cloud)
        if replacement_gpu:
            print(color(f"\n{gpu} became unavailable before Runpod could reserve it.", "33"))
            print("Trying the next compatible live GPU:")
            show_quote(model, replacement_gpu, count, cloud, args.estimate_hours)
            confirm("DEPLOY", f"This creates the same billable Pod using {replacement_gpu}.")
            command[command.index("--gpu-id") + 1] = replacement_gpu
            result = ctl(command, capture=True, check=False)
    if result.returncode:
        message = result.stderr.strip() or result.stdout.strip() or "runpodctl could not create the Pod."
        raise SystemExit(message)
    if result.stdout:
        print_pod_submission(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="" if result.stderr.endswith("\n") else "\n")
    print(color("\nWaiting for vLLM readiness…", "36"))
    print("First boot downloads the model to the selected Pod storage.")
    # runpodctl output varies by version; accept its explicit id fields only.
    match = re.search(r'(?im)(?:pod\s*(?:id)?|"id")\s*[:=]\s*["\']?([a-z0-9]{6,})', result.stdout)
    pod_id = match.group(1) if match else None
    if pod_id:
        wait_for_vllm(pod_id, args.wait_timeout)
    return pod_id


def cmd_status(_: argparse.Namespace) -> None:
    print_pods(list_pods())
    print(color("\nTip: `runpodctl pod get POD_ID` shows SSH and proxy details.", "2"))


def resolve_pod_id(pod_id: str | None) -> str:
    if pod_id:
        return pod_id
    pods = list_pods()
    if len(pods) == 1 and isinstance(pods[0], dict) and isinstance(pods[0].get("id"), str):
        inferred = pods[0]["id"]
        print(f"Using the only Pod: {inferred}")
        return inferred
    if not pods:
        raise SystemExit("No Pods found. Create one with `podcode up MODEL` first.")
    raise SystemExit("Multiple Pods found; specify POD_ID (see `podcode status`).")


def resolve_running_pod_id(pod_id: str | None) -> str:
    """Resolve the sole running Pod while ignoring stopped Pods."""
    if pod_id:
        return pod_id
    pods = list_pods()
    running = [
        pod for pod in pods
        if isinstance(pod, dict)
        and str(pod.get("desiredStatus") or pod.get("status") or "").upper() == "RUNNING"
        and isinstance(pod.get("id"), str)
    ]
    if len(running) == 1:
        inferred = running[0]["id"]
        print(f"Using the only running Pod: {inferred}")
        return inferred
    if not running:
        raise SystemExit("No running Pods found. Start one with `podcode start POD_ID` or create one with `podcode up`.")
    raise SystemExit("Multiple running Pods found; specify POD_ID (see `podcode status`).")


def cmd_wait(args: argparse.Namespace) -> None:
    """Poll a Pod's vLLM health endpoint and concise GPU-loading stage."""
    wait_for_vllm(resolve_pod_id(args.pod_id), args.timeout)


def cmd_logs(args: argparse.Namespace) -> None:
    """Stream readable Pod startup events, with raw logs available via --verbose."""
    pod_id = resolve_pod_id(args.pod_id)
    command = [require_ctl(), "pod", "logs", pod_id, "--follow"]
    print(color(f"Streaming Pod {pod_id} logs (Ctrl-C to stop)", "36"))
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy())
    try:
        assert process.stdout is not None
        for raw in process.stdout:
            try:
                event = json.loads(raw)
                source, line = str(event.get("source", "container")), str(event.get("line", ""))
            except json.JSONDecodeError:
                if args.verbose:
                    print(raw.rstrip())
                continue
            summary = concise_log_line(source, line)
            if summary:
                kind, message = summary
                marker, shade = {"info": ("●", "36"), "load": ("◌", "33"), "ready": ("✓", "32"), "error": ("✗", "31")}[kind]
                print(f"{color(marker, shade)} {message}")
            elif args.verbose:
                print(color(f"{source}: ", "2") + line)
    except KeyboardInterrupt:
        print("\nLog stream stopped.")
    finally:
        process.terminate()
        process.wait(timeout=5)


def cmd_control(args: argparse.Namespace) -> None:
    pod_id = resolve_running_pod_id(args.pod_id) if args.action == "stop" else args.pod_id
    ctl(["pod", args.action, pod_id], capture=True)
    verb = {"start": "started", "stop": "stopped", "restart": "restarted"}[args.action]
    print(color(f"\n✓ Pod {verb}", "32"))
    print(f"  {color('Pod ID:', '2')} {pod_id}")


def cmd_destroy(args: argparse.Namespace) -> None:
    pod_id = resolve_pod_id(args.pod_id)
    print(f"Deleting Pod {pod_id}. This is permanent; Pod-attached volume data will not be recoverable.")
    ctl(["pod", "delete", pod_id], capture=True)
    print(color("\n✓ Pod deleted", "32"))
    print(f"  {color('Pod ID:', '2')} {pod_id}")
    if args.delete_network_volume:
        print(f"Deleting network volume {args.delete_network_volume}.")
        ctl(["network-volume", "delete", args.delete_network_volume], capture=True)
        print(color("✓ Network volume deleted", "32"))
        print(f"  {color('Volume ID:', '2')} {args.delete_network_volume}")


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
    vllm_key = os.getenv("RUNPOD_VLLM_API_KEY")
    if not vllm_key:
        raise SystemExit("RUNPOD_VLLM_API_KEY is required to configure OpenCode.")
    target = Path(args.output).resolve()
    config = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"runpod/{model.huggingface_id}",
        "provider": {"runpod": {"npm": "@ai-sdk/openai-compatible", "name": "Runpod vLLM", "options": {
            "baseURL": f"https://{args.pod_id}-8000.proxy.runpod.net/v1", "apiKey": vllm_key},
            "models": {model.huggingface_id: {"name": model.key, "limit": {
                "context": model.max_model_len or 32768, "output": 8192}}}}},
    }
    if target.exists() and not args.force:
        try:
            current = json.loads(target.read_text())
            provider = current.get("provider", {}) if isinstance(current, dict) else {}
        except json.JSONDecodeError:
            provider = {}
        # This is a config previously generated by podcode. Refresh only its
        # Runpod connection after the replacement Pod becomes ready.
        if not (isinstance(provider, dict) and "runpod" in provider):
            raise SystemExit(f"{target} already exists; use --force to replace it, or choose --output.")
        current["model"] = config["model"]
        current["provider"]["runpod"] = config["provider"]["runpod"]
        config = current
    target.write_text(json.dumps(config, indent=2) + "\n")
    # This generated config contains the local vLLM key and is gitignored.
    target.chmod(0o600)
    print(f"Wrote {target}\nStart OpenCode in this directory; default model: runpod/{model.huggingface_id}")
    opencode = shutil.which("opencode")
    if opencode:
        reload_result = subprocess.run([opencode, "reload"], text=True, capture_output=True, check=False)
        if reload_result.returncode == 0:
            print("Reloaded OpenCode configuration for the new Pod.")


def cmd_up(args: argparse.Namespace) -> None:
    """Single-command path: deploy the model, then create local OpenCode config."""
    pod_id = cmd_deploy(args)
    if not pod_id:
        # Some runpodctl releases change their create-response shape. Recover
        # automatically when the new Pod is the only Pod rather than silently
        # leaving OpenCode pointed at an older deployment.
        print("\nThe create response did not contain a recognizable Pod ID; checking the Pod list…")
        pod_id = resolve_pod_id(None)
        wait_for_vllm(pod_id, args.wait_timeout)
    print(color("\nConfiguring OpenCode automatically…", "36"))
    config_args = argparse.Namespace(pod_id=pod_id, model=args.model, output=args.opencode_output, force=args.force_opencode_config)
    cmd_opencode(config_args)


def cmd_volume(args: argparse.Namespace) -> None:
    # Pass through exactly so this stays compatible with new runpodctl volume flags.
    ctl(["network-volume", *args.runpodctl_args])


def cmd_usage(_: argparse.Namespace) -> None:
    print_pods(list_pods(), heading="Current billable resources")
    print(textwrap.dedent("""\

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
    gpus = sub.add_parser("gpus", help="show live Runpod GPU availability and prices")
    gpus.add_argument("--json", action="store_true", help="print the unformatted Runpod inventory for scripts")
    gpus.set_defaults(func=cmd_gpus)
    sub.add_parser("status", help="list all Pods").set_defaults(func=cmd_status)
    wait = sub.add_parser("wait", help="poll vLLM readiness and GPU model-loading progress")
    wait.add_argument("pod_id", nargs="?", help="Pod ID; inferred when exactly one Pod exists")
    wait.add_argument("--timeout", type=duration_seconds, default=1800, metavar="DURATION", help="maximum wait; default: 30m")
    wait.set_defaults(func=cmd_wait)
    logs = sub.add_parser("logs", help="stream detailed container logs for a Pod")
    logs.add_argument("pod_id", nargs="?", help="Pod ID; inferred when exactly one Pod exists")
    logs.add_argument("--verbose", action="store_true", help="also show unfiltered raw log lines")
    logs.set_defaults(func=cmd_logs)
    sub.add_parser("usage", help="show Pods and billing guidance").set_defaults(func=cmd_usage)

    def add_deploy_options(target: argparse.ArgumentParser, *, up_defaults: bool = False) -> None:
        target.add_argument("model", nargs="?" if up_defaults else None, choices=sorted(MODELS),
                            default=DEFAULT_MODEL if up_defaults else None,
                            help=f"model preset (default: {DEFAULT_MODEL})" if up_defaults else None)
        target.add_argument("--name")
        target.add_argument("--gpu", help="override the recommended Runpod GPU name")
        target.add_argument("--gpu-count", type=int, help="override number of GPUs")
        target.add_argument("--cloud-type", choices=("SECURE", "COMMUNITY"))
        target.add_argument("--volume-gb", type=int, help="Pod-attached volume size (ignored with --network-volume-id)")
        target.add_argument("--network-volume-id", help="reusable Runpod network volume; preserves the model cache across swaps")
        if up_defaults:
            storage = target.add_mutually_exclusive_group()
            storage.add_argument("--ephemeral", dest="ephemeral", action="store_true", help="discard model storage when the Pod is deleted (default)")
            storage.add_argument("--persistent", dest="ephemeral", action="store_false", help="create a persistent Pod volume for the model cache")
            target.set_defaults(ephemeral=True)
        else:
            target.add_argument("--ephemeral", action="store_true", help="do not create a persistent volume; model cache is discarded with the Pod")
        target.add_argument("--container-disk-gb", type=int)
        target.add_argument("--stop-after", help="auto-stop duration, e.g. 8h")
        target.add_argument("--terminate-after", help="auto-delete duration, e.g. 24h")
        target.add_argument("--estimate-hours", type=float, default=10 if up_defaults else 1,
                            help=f"hours used for the preflight cost estimate (default: {10 if up_defaults else 1})")
        target.add_argument("--wait-timeout", type=duration_seconds, default=1800, metavar="DURATION", help="wait for vLLM readiness; default: 30m")

    deploy = sub.add_parser("deploy", help="create a vLLM Pod and load a model")
    add_deploy_options(deploy)
    deploy.set_defaults(func=cmd_deploy)

    up = sub.add_parser("up", help="single command: cost preflight, deploy, then configure local OpenCode")
    add_deploy_options(up, up_defaults=True)
    up.add_argument("--opencode-output", default="opencode.json", help="local OpenCode config path")
    up.add_argument("--force-opencode-config", action="store_true", help="replace the generated OpenCode config if it exists")
    up.set_defaults(func=cmd_up, yes_no_confirm=True)

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
        control.add_argument("pod_id", nargs="?" if action == "stop" else None,
                             help="Pod ID; for stop, inferred when exactly one Pod is running")
        control.set_defaults(func=cmd_control, action=action)
    destroy = sub.add_parser("destroy", help="delete a Pod, optionally also a network volume")
    destroy.add_argument("pod_id", nargs="?", help="Pod ID; inferred when exactly one Pod exists")
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
