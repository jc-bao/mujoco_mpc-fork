#!/usr/bin/env python3
"""Evaluate a controller using GUI visualization with UI agent server."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import shutil
import re
import time
from loop_rate_limiters import RateLimiter
import numpy as np
import mujoco
from mujoco_mpc import agent as agent_lib
import tyro


@dataclass
class EvaluationConfig:
    """Configuration for GUI-based controller evaluation."""
    
    task: str = "Acrobot"
    """Task ID (e.g., 'Cartpole')."""
    
    controller: str = "Feedback Sampling"
    """Controller ID (e.g., 'Sampling', 'iLQG', 'Feedback Sampling')."""
    
    model_path: Optional[Path] = None
    """Optional path to the task XML model."""

    total_time: float = 60.0
    """Total time to run the simulation."""

    record_dt: float = 0.02
    """Time step to record data."""

# Planner ID mapping
planner_id_map = {
    "Sampling": 0,
    "Gradient": 1,
    "iLQG": 2,
    "iLQS": 3,
    "Robust Sampling": 4,
    "Cross Entropy": 5,
    "Sample Gradient": 6,
    "Feedback Sampling": 7,
}


def update_planner_config(model_path: Path, planner_id: int) -> Path:
    """Update the planner configuration in the XML file.
    
    Args:
        model_path: Path to the original XML model file.
        planner_id: ID of the planner to use.
        
    Returns:
        Path to the modified XML file.
    """
    # Verify source file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Source file not found: {model_path}")

    # Create the benchmark filename
    benchmark_path = model_path.with_name("task_benchmark.xml")

    # Copy the file
    shutil.copy(model_path, benchmark_path)

    # Read and modify the file
    with open(benchmark_path, "r") as file:
        content = file.read()

    # Update planner configuration
    modified_content = re.sub(
        r'<numeric\s+name="agent_planner"\s+data="\d+"\s*/>', 
        f'<numeric name="agent_planner" data="{planner_id}" />', 
        content
    )

    # Write the modified content
    with open(benchmark_path, "w") as file:
        file.write(modified_content)

    return benchmark_path


def main(config: EvaluationConfig) -> None:
    """Main function to run evaluation.
    
    Args:
        config: Configuration dataclass containing evaluation parameters.
    """
    # Resolve model path
    if config.model_path is None:
        config.model_path = (
            Path(__file__).parent.parent
            / "../../build/mjpc/tasks"
            / config.task.lower()
            / "task.xml"
        )

    # Update planner configuration
    planner_id = planner_id_map[config.controller]
    try:
        model_path = update_planner_config(config.model_path, planner_id)
        print(f"Successfully created and updated: {model_path}")
    except Exception as e:
        print(f"Failed to update planner configuration: {e}")
        return

    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Run GUI with agent server
    with agent_lib.Agent(
        server_binary_path=Path(agent_lib.__file__).parent / "mjpc" / "ui_agent_server",
        task_id=config.task,
        model=model,
    ) as agent:
        # activate planner
        t0 = time.time()
        rate_limiter = RateLimiter(frequency=1 / config.record_dt)
        agent.plan_enabled = True
        costs = []
        try:
            print(f"Running {config.task} with {config.controller} controller")
            print("Press Ctrl+C to stop")
            while time.time() - t0 < config.total_time:
                costs.append(agent.get_total_cost())
                rate_limiter.sleep()
        except KeyboardInterrupt:
            print("\nStopping evaluation")

    costs = np.array(costs)
    print(f"{config.task} {config.controller} cost: {costs.mean()}")

if __name__ == "__main__":
    main(tyro.cli(EvaluationConfig))