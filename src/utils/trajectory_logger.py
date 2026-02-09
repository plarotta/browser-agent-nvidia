import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from PIL import Image
import os

@dataclass
class TimeStep:
    step_id: int
    timestamp: float
    observation_dom: str
    action: Dict[str, Any]
    reward: float
    done: bool
    # Images are stored separately to avoid massive JSONs
    screenshot_path: Optional[str] = None

class TrajectoryLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_trajectory: List[TimeStep] = []
        self.trajectory_id = int(time.time())

    def log_step(self, step_id: int, dom: str, screenshot: Image.Image, action: Dict[str, Any], reward: float, done: bool):
        """Logs a single step."""
        timestamp = time.time()
        
        # Save screenshot
        screenshot_filename = f"traj_{self.trajectory_id}_step_{step_id}.png"
        screenshot_path = os.path.join(self.log_dir, screenshot_filename)
        screenshot.save(screenshot_path)
        
        step = TimeStep(
            step_id=step_id,
            timestamp=timestamp,
            observation_dom=dom, # In real app, we might hash or compress this
            action=action,
            reward=reward,
            done=done,
            screenshot_path=screenshot_path
        )
        self.current_trajectory.append(step)

    def save_trajectory(self):
        """Saves the full trajectory to JSON."""
        filepath = os.path.join(self.log_dir, f"traj_{self.trajectory_id}.json")
        data = [asdict(step) for step in self.current_trajectory]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath
