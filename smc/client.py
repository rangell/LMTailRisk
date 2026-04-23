import time
import os
import re
import signal
import subprocess
import threading
from openai import AsyncOpenAI

from smc.utils import is_main_process, wait_for_everyone


class OutputStreamReader(threading.Thread):
    def __init__(self, proc, stop_reading_event, verbose):
        super().__init__()
        self._stop_event = stop_reading_event
        self._proc = proc
        self._verbose = verbose

    def run(self):
        while not self._stop_event.is_set():
            try:
                lines = self._proc.stdout.readlines()
                for line in lines:
                    # if self._verbose:
                    #    print(f"[JUDGE SERVER] {line.strip()}", end="\r\n")
                    pass
            except BlockingIOError:
                # No data available to read at this moment
                pass


class JudgeClient:
    def __init__(self, launch_script_path: str, verbose: bool = False) -> None:
        super().__init__()
        self.launch_script_path = launch_script_path
        self.verbose = verbose

    def __enter__(self) -> AsyncOpenAI:
        client = None
        if is_main_process():
            client = self.enter_body()
        wait_for_everyone()
        return client

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        exit_val = None
        if is_main_process():
            exit_val = self.exit_body(exc_type, exc_value, traceback)
        wait_for_everyone()
        return exit_val

    def enter_body(self) -> AsyncOpenAI:
        # TODO: if launch_script_path is empty we try using together ai
        # TODO: tensor parallel size in launch script

        # busy_gpu_subprocess = subprocess.Popen(
        #    ["python", "-u", "misc/run_50xALL.py", "2>&1", "/dev/null"],
        #    stdout=subprocess.PIPE,
        #    stderr=subprocess.PIPE,
        #    bufsize=1,
        #    text=True,
        #    start_new_session=True,
        # )

        command = ["sh", self.launch_script_path]

        # Start the judge server
        print(f"Starting vLLM server with command: {' '.join(command)}")
        self.proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            start_new_session=True,
        )

        # The stdout stream from proc is blocking currently, since we are waiting for the judge to start this is fine
        ip_address = None
        for line in self.proc.stdout:
            if self.verbose:
                print(f"[JUDGE SERVER] {line.strip()}", end="\r\n")
            match = re.search(
                r"^([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+) ",
                line.strip(),
            )
            if match:
                ip_address = match.group(1)
                if self.verbose:
                    print(
                        f"Got the judge server IP address! ({ip_address})", end="\r\n"
                    )
            if "Available routes are:" in line:
                break

        # If we can't extract the ip address, we raise an error
        if ip_address is None:
            raise RuntimeError(
                "IP address could not be extracted from judge launch script output..."
            )

        # If the judge fails to start, we raise an error
        if self.proc.poll() is not None:
            raise RuntimeError("Judge failed to start...")

        # Set stdout to non-blocking
        os.set_blocking(self.proc.stdout.fileno(), False)

        # Create stop event
        self.stop_reading_event = threading.Event()

        # Start output buffer thread
        self.stream_reader = OutputStreamReader(
            self.proc, self.stop_reading_event, verbose=self.verbose
        )
        self.stream_reader.start()

        # Create the client
        client = AsyncOpenAI(
            base_url=f"http://{ip_address}:8000/v1", api_key="sk-no-key-required"
        )

        # busy_gpu_subprocess.terminate()
        # busy_gpu_subprocess.wait()

        return client

    def exit_body(self, exc_type, exc_value, traceback) -> bool | None:
        # Stop reading from the output stream of the judge server
        self.stop_reading_event.set()
        self.stream_reader.join()

        # Send SIGTERM to proc
        os.killpg(self.proc.pid, signal.SIGINT)
        time.sleep(5)

        if self.proc.poll() is None:
            self.proc.terminate()
            time.sleep(5)
            if self.proc.poll() is None:
                self.proc.kill()

        self.proc.wait()

        assert self.proc.poll() is not None
