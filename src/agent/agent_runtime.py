import time
from typing import Optional, Dict, Any
from src.browser_interaction.session_manager import SessionManager
from src.browser_interaction.action_executor import ActionExecutor
from src.observation.visual_capture import VisualCapture
from src.observation.dom_snapshotter import DOMSnapshotter
from src.utils.trajectory_logger import TrajectoryLogger
from src.policy.multimodal_policy import MultimodalPolicy
from src.sdft.sdft_module import SDFTModule
from src.agent.action_parser import ActionParser
from src.utils.config import AgentConfig
import logging

logger = logging.getLogger(__name__)

class AgentRuntime:
    def __init__(self, config: AgentConfig, policy: Optional[MultimodalPolicy] = None, sdft: Optional[SDFTModule] = None):
        self.config = config
        self.session_manager = SessionManager(headless=config.headless)
        self.action_executor = None 
        self.visual_capture = None
        self.dom_snapshotter = None
        self.logger = TrajectoryLogger(config.log_dir)
        self.step_count = 0
        
        # Policy & Learning
        self.policy = policy
        self.sdft = sdft
        self.action_parser = ActionParser()

    def start(self, start_url: str):
        """Initializes the browser session."""
        page = self.session_manager.start()
        # Ensure viewport matches config
        page.set_viewport_size({"width": self.config.viewport_width, "height": self.config.viewport_height})
        
        self.action_executor = ActionExecutor(page)
        self.visual_capture = VisualCapture(page)
        self.dom_snapshotter = DOMSnapshotter(page)
        self.session_manager.navigate(start_url)
        
        if self.policy:
            # Ensure model is ready (lazy load if needed)
            if not self.policy.model:
                 self.policy.load_model()


    def step(self, action: Optional[Dict[str, Any]] = None) -> bool:
        """Executes one step: Observe -> Predict -> Act -> Log."""
        if not self.action_executor:
            raise RuntimeError("Agent not started")
            
        # 1. Observe
        dom = self.dom_snapshotter.capture()
        screenshot = self.visual_capture.capture()
        
        # 2. Predict Action (if not provided manually)
        action_dict = action
        if not action_dict and self.policy:
            # Construct prompt - this is simplified. In reality, we'd add history/goal.
            prompt = "<|image|><|begin_of_text|>User: Goal: Navigate the page.\nAssistant: I will" 
            try:
                raw_action_text = self.policy.forward(screenshot, prompt)
                logger.info(f"Model Predicted: {raw_action_text}")
                action_dict = self.action_parser.parse(raw_action_text)
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
        
        # Fallback if no policy or parsing failed (for testing/debugging or manual override)
        if not action_dict:
            logger.warning("No valid action predicted. Waiting...")
            action_dict = {"type": "wait", "params": {"duration": 1000}}

        # 3. Act
        success = self.action_executor.execute(action_dict.get("type"), action_dict.get("params", {}))
        
        # 4. Log
        # Reward placeholder
        reward = 1.0 if success else 0.0
        self.logger.log_step(
            step_id=self.step_count,
            dom=dom, # We might want to save token usage here too
            screenshot=screenshot,
            action=action_dict,
            reward=reward,
            done=False 
        )
        
        # 5. Learn (SDFT) - Implementation of the hook
        if self.sdft and self.policy:
             # Logic to calculate loss and update adapter would go here
             # We need logits from the forward pass for this, which we'd need to expose from Policy
             # For this prototype step, we'll just check if we SHOULD update
             if self.sdft.should_update(entropy=0.0, success=success): # Entropy mocked
                 self.sdft.update_teacher()
                 logger.info("SDFT: Updated Teacher Model")

        self.step_count += 1
        return success

    def stop(self):
        """Cleanup."""
        path = self.logger.save_trajectory()
        logger.info(f"Trajectory saved to {path}")
        self.session_manager.close()
