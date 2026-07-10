from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("paramiko")

from mlbag.remote import JobResult, SSHJobRunner


def _mock_client(stdout_text: str = "ok\n", stderr_text: str = "", returncode: int = 0) -> MagicMock:
    client = MagicMock()
    stdout = MagicMock()
    stdout.read.return_value = stdout_text.encode()
    stdout.channel.recv_exit_status.return_value = returncode
    stderr = MagicMock()
    stderr.read.return_value = stderr_text.encode()
    client.exec_command.return_value = (MagicMock(), stdout, stderr)
    return client


def test_job_result_ok_true_for_zero_returncode():
    assert JobResult(stdout="", stderr="", returncode=0).ok


def test_job_result_ok_false_for_nonzero_returncode():
    assert not JobResult(stdout="", stderr="", returncode=1).ok


@patch("paramiko.SSHClient")
def test_run_returns_job_result(mock_ssh_client_cls):
    mock_ssh_client_cls.return_value = _mock_client("hello\n", "", 0)
    runner = SSHJobRunner(host="wsl-box", user="cyrus", remote_workdir="/home/cyrus/jobs")

    result = runner.run("echo hello")

    assert isinstance(result, JobResult)
    assert result.ok
    assert result.stdout == "hello\n"


@patch("paramiko.SSHClient")
def test_run_reports_nonzero_returncode_as_not_ok(mock_ssh_client_cls):
    mock_ssh_client_cls.return_value = _mock_client("", "boom\n", 1)
    runner = SSHJobRunner(host="wsl-box", user="cyrus")

    result = runner.run("false")

    assert not result.ok
    assert result.returncode == 1
    assert "boom" in result.stderr


@patch("paramiko.SSHClient")
def test_upload_puts_each_local_file_into_remote_workdir(mock_ssh_client_cls, tmp_path):
    client = MagicMock()
    sftp = MagicMock()
    client.open_sftp.return_value = sftp
    mock_ssh_client_cls.return_value = client

    local_file = tmp_path / "job.lammps"
    local_file.write_text("# lammps input")
    runner = SSHJobRunner(host="wsl-box", user="cyrus", remote_workdir="/home/cyrus/jobs")

    runner.upload([local_file])

    sftp.put.assert_called_once_with(str(local_file), "/home/cyrus/jobs/job.lammps")


@patch("paramiko.SSHClient")
def test_download_fetches_each_remote_path_into_local_dir(mock_ssh_client_cls, tmp_path):
    client = MagicMock()
    sftp = MagicMock()
    client.open_sftp.return_value = sftp
    mock_ssh_client_cls.return_value = client

    runner = SSHJobRunner(host="wsl-box", user="cyrus")
    downloaded = runner.download(["/home/cyrus/jobs/log.lammps"], tmp_path / "out")

    assert downloaded == [tmp_path / "out" / "log.lammps"]
    sftp.get.assert_called_once_with("/home/cyrus/jobs/log.lammps", str(tmp_path / "out" / "log.lammps"))


@patch("paramiko.SSHClient")
def test_run_job_uploads_runs_and_downloads(mock_ssh_client_cls, tmp_path):
    client = _mock_client("done\n", "", 0)
    sftp = MagicMock()
    client.open_sftp.return_value = sftp
    mock_ssh_client_cls.return_value = client

    local_input = tmp_path / "in.lammps"
    local_input.write_text("# input")
    runner = SSHJobRunner(host="wsl-box", user="cyrus", remote_workdir="/jobs")

    result, downloaded = runner.run_job(
        "lammps -in in.lammps",
        inputs=[local_input],
        outputs=["/jobs/log.lammps"],
        local_output_dir=tmp_path / "out",
    )

    assert result.ok
    assert downloaded == [tmp_path / "out" / "log.lammps"]
    sftp.put.assert_called_once()
    sftp.get.assert_called_once()


@patch("paramiko.SSHClient")
def test_run_job_skips_upload_when_no_inputs(mock_ssh_client_cls, tmp_path):
    client = _mock_client("done\n", "", 0)
    sftp = MagicMock()
    client.open_sftp.return_value = sftp
    mock_ssh_client_cls.return_value = client

    runner = SSHJobRunner(host="wsl-box", user="cyrus")
    result, downloaded = runner.run_job("echo hi")

    assert result.ok
    assert downloaded == []
    sftp.put.assert_not_called()
