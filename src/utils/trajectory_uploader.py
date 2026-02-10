import os
import tarfile
import tempfile
import logging

import requests

logger = logging.getLogger(__name__)


class TrajectoryUploader:
    def __init__(self, server_url: str):
        self.server_url = server_url

    def upload(self, log_dir: str) -> str:
        """Package log_dir as tar.gz and upload to server.

        Returns the trajectory_id assigned by the server.
        """
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with tarfile.open(tmp_path, "w:gz") as tar:
                for fname in os.listdir(log_dir):
                    fpath = os.path.join(log_dir, fname)
                    if os.path.isfile(fpath):
                        tar.add(fpath, arcname=fname)

            with open(tmp_path, "rb") as f:
                resp = requests.post(
                    f"{self.server_url}/upload_trajectory",
                    files={"file": ("trajectory.tar.gz", f, "application/gzip")},
                    timeout=120,
                )
            resp.raise_for_status()
            traj_id = resp.json()["trajectory_id"]
            logger.info(f"Trajectory uploaded: {traj_id}")
            return traj_id
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
