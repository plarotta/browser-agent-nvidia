import re
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ActionParser:
    """
    Parses natural language or structured text actions from the model 
    into the dictionary format required by ActionExecutor.
    Expected format examples:
    - click(selector="#submit")
    - type(selector="#search", text="hello world")
    - scroll(direction="down", amount=500)
    - wait(duration=1000)
    """
    
    @staticmethod
    def parse(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        
        # Regex to capture function name and arguments inside parentheses
        # matches: name(args)
        match = re.match(r"^(\w+)\((.*)\)$", text)
        if not match:
            logger.warning(f"Could not parse action: {text}")
            return None
        
        action_type = match.group(1).lower()
        args_str = match.group(2)
        
        params = {}
        if args_str:
            # Simple attribute parser: key="value" or key=number
            # This is a basic parser and might need to be more robust for complex strings
            # handling escaped quotes etc.
            # strict regex for key="value" or key=123
            arg_matches = re.findall(r'(\w+)=(?:"([^"]*)"|(\d+))', args_str)
            for key, val_str, val_int in arg_matches:
                if val_str is not None and val_str != "":
                    params[key] = val_str
                elif val_int is not None and val_int != "":
                    params[key] = int(val_int)
        
        return {"type": action_type, "params": params}
