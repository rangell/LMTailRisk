"""vLLM server lifecycle utilities."""

from contextlib import contextmanager
import os
import select
import socket
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

import requests
from openai import AsyncOpenAI


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@contextmanager
def vllm_server(
    model: str,
    port: Optional[int] = None,
    tokenizer: Optional[str] = None,
    extra_args: Optional[list] = None,
    adapters: Optional[Dict[str, str]] = None,
) -> Tuple[AsyncOpenAI, int]:
    """Spawn a vLLM server and yield (AsyncOpenAI client, port)."""
    if port is None:
        port = find_free_port()

    cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        # "--generation-config",
        # "vllm",
        "--enable-sleep-mode",
        *(["--tokenizer", tokenizer] if tokenizer else []),
        *(extra_args or []),
    ]
    if adapters:
        cmd += ["--enable-lora"]
        for name, path in adapters.items():
            cmd += ["--lora-modules", f"{name}={path}"]

    print("[vllm] launch command:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ},
    )

    quiet_event = threading.Event()

    def _drain():
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if ready:
                line = proc.stdout.readline()
                if not line:
                    break
                if not quiet_event.is_set():
                    print("[vllm]", line.decode(errors="replace"), end="", flush=True)

    threading.Thread(target=_drain, daemon=True).start()

    try:
        url = f"http://localhost:{port}/health"
        for _ in range(1000):
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(1)
        else:
            proc.kill()
            raise RuntimeError(f"vLLM server on port {port} failed to start")

        quiet_event.set()
        sleep_server(port)
        yield AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY"), port
    finally:
        proc.terminate()
        proc.wait()


def sleep_server(port: int, level: int = 1) -> None:
    requests.post(f"http://localhost:{port}/sleep", json={"level": level})


def wake_server(port: int) -> None:
    requests.post(f"http://localhost:{port}/wake_up")
