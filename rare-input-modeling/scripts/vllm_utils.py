"""Utilities for spawning and communicating with a vLLM server."""

from contextlib import contextmanager
import os
import select
import socket
import subprocess
import threading
import time

import requests
from openai import AsyncOpenAI


def find_free_port() -> int:
    """Return an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@contextmanager
def vllm_server(
    model: str, port: int | None = None, extra_args: list[str] | None = None
):
    """Spawn a vLLM server and yield an OpenAI client pointed at it."""
    if port is None:
        port = find_free_port()
    proc = subprocess.Popen(
        [
            "vllm",
            "serve",
            model,
            "--port",
            str(port),
            "--generation-config",
            "vllm",
            "--enable-chunked-prefill",
            "--max-num-batched-tokens",
            "65536",
            *(extra_args or []),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={
            **os.environ
        },  # let vLLM auto-select backend (Triton for sinks on non-Hopper)
    )

    quiet_event = threading.Event()

    def _drain_stdout():
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0.5)
            if ready:
                line = proc.stdout.readline()
                if not line:  # EOF
                    break
                if not quiet_event.is_set():
                    print("[vllm]", line.decode(errors="replace"), end="", flush=True)

    drain_thread = threading.Thread(target=_drain_stdout, daemon=True)
    drain_thread.start()

    try:
        url = f"http://localhost:{port}/health"
        for _ in range(1000):
            try:
                r = requests.get(url, timeout=2)
                if r.status_code == 200:
                    break
            except requests.ConnectionError:
                pass
            time.sleep(1)
        else:
            proc.kill()
            raise RuntimeError("vLLM server failed to start")

        quiet_event.set()

        yield AsyncOpenAI(base_url=f"http://localhost:{port}/v1", api_key="EMPTY")
    finally:
        proc.terminate()
        proc.wait()
