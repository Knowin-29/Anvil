import time
from typing import Dict


class RobotBridge:
    def __init__(self, mode: str = "mock") -> None:
        self.mode = mode

    def execute(self, decision: Dict) -> Dict:
        if self.mode == "mock":
            time.sleep(0.1)
            return {
                "mode": "mock",
                "accepted": True,
                "message": "Mock robot accepted grasp plan.",
                "decision": decision,
            }

        # Placeholder for real robot execution adapter.
        return {
            "mode": "real",
            "accepted": False,
            "message": "Real robot adapter not configured yet.",
            "decision": decision,
        }

