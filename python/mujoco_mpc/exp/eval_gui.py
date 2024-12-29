#!/usr/bin/env python3
"""Evaluate a controller using GUI visualization with UI agent server, 
with the ability to load planner parameters from a YAML config.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict
import shutil
import re
import time
import sys

import numpy as np
import mujoco
from mujoco_mpc import agent as agent_lib

import tyro

# Add PyYAML import
import yaml

from loop_rate_limiters import RateLimiter


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

    config_file: Optional[Path] = None
    """Path to a YAML file containing planner parameters."""

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


def update_planner_config(
    model_path: Path, 
    planner_id: int, 
    yaml_config: Dict[str, Any] = None
) -> Path:
    """Update the planner configuration in the XML file.
    
    Args:
        model_path: Path to the original XML model file.
        planner_id: ID of the planner to use.
        yaml_config: Dictionary of parameters loaded from the config.yaml file.
        
    Returns:
        Path to the modified XML file.
    """
    # Verify source file exists
    print(model_path)
    # make model_path a path object
    # model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Source file not found: {model_path}")

    # Create the benchmark filename
    benchmark_path = model_path.with_name("task_benchmark.xml")

    # Copy the file
    shutil.copy(model_path, benchmark_path)

    # 1) Update the planner ID
    # Example: <numeric name="agent_planner" data="7" />
    content = re.sub(
        r'<numeric\s+name="agent_planner"\s+data="\d+"\s*/>',
        f'<numeric name="agent_planner" data="{planner_id}" />',
        content
    )


    # 2) Use the YAML config (if provided) to override numeric tags in the XML
    if yaml_config is not None:
        for key, value in yaml_config.items():
            # Convert to a string representation that MuJoCo can read
            # If it's a list, join with spaces; otherwise convert directly to str
            if isinstance(value, list):
                value_str = " ".join(str(x) for x in value)
            else:
                value_str = str(value)

            # Regex pattern to find an existing numeric tag with name="key"
            # e.g. <numeric name="sampling_scale" data="..."/>
            pattern = rf'<numeric\s+name="{key}"\s+data="[^"]*"\s*/>'
            replacement = f'<numeric name="{key}" data="{value_str}" />'

            # Attempt to replace if it exists
            content, num_subs = re.subn(pattern, replacement, content)
            if num_subs > 0:
                print(f"Updated <numeric name=\"{key}\" data=\"...\" /> → {value_str}")
            else:
                print(f"Warning: No XML tag found for <numeric name=\"{key}\" .../>.")

    # Write the modified content
    with open(benchmark_path, "w") as file:
        file.write(content)

    return benchmark_path

def walker_task_init(agent,yaml_config = None):
    """Walker task initialization."""
    print('Walker task initialization')
    agent.set_task_parameters({"Speed Goal": yaml_config["speed_goal"]})
    print('--------------------------------')
    print(agent.get_task_parameters())

def allegro_task(agent, reset_time, t0):
    """Allegro task behavior."""
    if time.time() - t0 >= reset_time:
        # Randomize a target quaternion
        random_quat = np.random.rand(4)
        random_quat /= np.linalg.norm(random_quat)  # Normalize to make it a valid quaternion
        # Put it into the first 4 elements in array of qpos
        model_state = agent.get_state()
        model_state.qpos[:4] = random_quat
        agent.set_state(qpos=model_state.qpos, qvel=model_state.qvel)
        reset_time += reset_time  # Schedule next reset
        print(f"Reset state at time {time.time() - t0}")
    return reset_time

# Add more task-specific functions as needed
# def another_task(agent, t0):
#     ...

def default_init(agent,yaml_config = None):
    """Default task initialization function."""
    print('--------------------------------')
    # print(agent.get_task_parameters())
    print(agent.get_cost_weights())
    # agent.set_cost_weights({"Distance": 30.0})
    print('Default task initialization')
    pass

def default_task(agent, t0):
    """Default task behavior when no specific task function is found."""
    # Implement default behavior here, if needed
    pass

def main(config: EvaluationConfig) -> Optional[float]:
    """Main function to run evaluation.
    
    Args:
        config: Configuration dataclass containing evaluation parameters.
    
    Returns:
        The mean cost if successful, otherwise None.
    """
    # Resolve model path
    if config.model_path is None:
        config.model_path = (
            Path(__file__).parent.parent
            / "../../build/mjpc/tasks"
            / config.task.lower()
            / "task.xml"
        )

    # Prepare YAML config if provided
    yaml_config = None
    if config.config_file is not None and config.config_file.exists():
        with open(config.config_file, "r") as f:
            yaml_config = yaml.safe_load(f)
            print(f"Loaded config from {config.config_file}:")
            print(yaml_config)
            # Only try to update total_time if yaml_config exists
            try:
                config.total_time = yaml_config["environment"]["total_time"]
            except (KeyError, TypeError):
                print('No total_time found in the config file. Using default value.')
    else:
        print("No YAML config file provided or file does not exist. Proceeding without overrides.")

    # Update planner configuration
    print(config.controller)
    planner_id = planner_id_map[config.controller]
    try:
        if planner_id == 7:
            yaml_config_feedback = yaml_config["feedback_sampling"]
            print(yaml_config_feedback)
            model_path = update_planner_config(config.model_path, planner_id, yaml_config_feedback)
            print(f"Successfully created and updated: {model_path}")
        else:
            # do not pass yaml_config for non-feedback sampling planners
            model_path = update_planner_config(config.model_path, planner_id,yaml_config["environment"])
            print("Load default config for non-feedback sampling planners.")
    except Exception as e:
        print(f"Failed to update planner configuration: {e}")
        return

    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    print("Agent server binary path:", Path(agent_lib.__file__).parent / "mjpc" / "ui_agent_server")
    print("Task ID:", config.task)
    print("Model Path:", model_path)
    # Run GUI with agent server
    with agent_lib.Agent(
        server_binary_path=Path(agent_lib.__file__).parent / "mjpc" / "ui_agent_server",
        task_id=config.task,
        model=model,
    ) as agent:
        # list agent's attributes
        # print(agent.__dict__)
        # list agent's planner attributes
        # print(agent.planner.__dict__)
        # list agent's methods
        print(agent.get_task_parameters())
        task_init_function = f"{config.task.lower()}_task_init"
        task_init_function = getattr(sys.modules[__name__], task_init_function, default_init)
        task_init_function(agent,yaml_config["environment"])
        t0 = time.time()
        rate_limiter = RateLimiter(frequency=1 / config.record_dt)
        agent.plan_enabled = True
        costs = []
        try:
            print(f"Running {config.task} with {config.controller} controller.")
            print("Press Ctrl+C to stop.")
            # print('YAML config:', yaml_config)
            if 'reset_state_time' in yaml_config["environment"]:
                reset_time = yaml_config["environment"]["reset_state_time"]

            # Get the task-specific function
            task_function_name = f"{config.task.lower()}_task"
            task_function = getattr(sys.modules[__name__], task_function_name, default_task)

            while time.time() - t0 < config.total_time:
                # Call the task function with the necessary parameters
                if 'reset_time' in task_function.__code__.co_varnames:
                    reset_time = task_function(agent, reset_time, t0)
                else:
                    task_function(agent, t0)
                costs.append(agent.get_total_cost())
                rate_limiter.sleep()
                    
        except KeyboardInterrupt:
            print("\nStopping evaluation...")

    costs = np.array(costs)
    mean_cost = costs.mean()
    print(f"{config.task} {config.controller} cost: {mean_cost}")
    return mean_cost


if __name__ == "__main__":
    # Use Tyro to parse CLI arguments into EvaluationConfig,
    # including the new `config_file` argument.
    main(tyro.cli(EvaluationConfig))
