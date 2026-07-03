"""Remote job execution over SSH (e.g. driving MD/QM jobs on a WSL box). Requires the
`remote` extra (paramiko).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class JobResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class SSHJobRunner:
    """Upload inputs, run a command, and download outputs on a remote host over SSH.

    Opens a fresh connection per call rather than holding one open, since MD/QM jobs are
    submitted infrequently relative to connection overhead — this keeps the class simple
    and trivially mockable (patch `paramiko.SSHClient`).
    """

    def __init__(
        self,
        host: str,
        user: str,
        *,
        key_path: str | Path | None = None,
        port: int = 22,
        remote_workdir: str = ".",
    ) -> None:
        self.host = host
        self.user = user
        self.key_path = Path(key_path) if key_path else None
        self.port = port
        self.remote_workdir = remote_workdir

    def _connect(self):
        import paramiko

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=self.host,
            port=self.port,
            username=self.user,
            key_filename=str(self.key_path) if self.key_path else None,
        )
        return client

    def run(self, cmd: str, *, timeout: float | None = None) -> JobResult:
        client = self._connect()
        try:
            _, stdout, stderr = client.exec_command(f"cd {self.remote_workdir} && {cmd}", timeout=timeout)
            returncode = stdout.channel.recv_exit_status()
            return JobResult(
                stdout=stdout.read().decode(),
                stderr=stderr.read().decode(),
                returncode=returncode,
            )
        finally:
            client.close()

    def upload(self, local_paths: Sequence[str | Path], remote_dir: str | None = None) -> None:
        client = self._connect()
        try:
            sftp = client.open_sftp()
            try:
                target_dir = remote_dir or self.remote_workdir
                for local_path in local_paths:
                    local_path = Path(local_path)
                    sftp.put(str(local_path), f"{target_dir}/{local_path.name}")
            finally:
                sftp.close()
        finally:
            client.close()

    def download(self, remote_paths: Sequence[str], local_dir: str | Path) -> list[Path]:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        client = self._connect()
        try:
            sftp = client.open_sftp()
            try:
                downloaded = []
                for remote_path in remote_paths:
                    local_path = local_dir / Path(remote_path).name
                    sftp.get(remote_path, str(local_path))
                    downloaded.append(local_path)
                return downloaded
            finally:
                sftp.close()
        finally:
            client.close()

    def run_job(
        self,
        cmd: str,
        *,
        inputs: Sequence[str | Path] = (),
        outputs: Sequence[str] = (),
        local_output_dir: str | Path = ".",
        timeout: float | None = None,
    ) -> tuple[JobResult, list[Path]]:
        """Convenience wrapper for the common case: upload inputs, run `cmd`, fetch outputs."""
        if inputs:
            self.upload(inputs)
        result = self.run(cmd, timeout=timeout)
        downloaded = self.download(outputs, local_output_dir) if outputs else []
        return result, downloaded
