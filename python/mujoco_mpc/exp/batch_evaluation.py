#!/usr/bin/env python3
import subprocess
import statistics
import re
import csv
from pathlib import Path

# Add imports from eval_gui.py
from eval_gui import main as eval_main
from eval_gui import EvaluationConfig

def parse_cost_from_output(output, task, controller):
    """
    Given the stdout from `evaluation.py`, extract the final cost.

    We expect a line like: 
        <TASK> <CONTROLLER> cost: 123.45
    We'll use a regex to capture that number.
    """
    pattern = rf"{task}\s+{controller}\s+cost:\s+([0-9\.\-eE]+)"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    return None


def main():
    """
    Batch evaluation script that:
      1. Runs multiple tasks with different controllers.
      2. Uses a distinct total_time per task.
      3. Collects and prints final results (mean ± std).
      4. Saves all run data to a CSV file.
    """

    # Dictionary mapping tasks to their total_time
    # You can add more tasks here, or remove tasks you don’t need
    tasks_config = {
        # "Cartpole",
        "Acrobot",
        # "Quadruped Hill",
        # "allegro",
        # "swimmer",
        # "walker",

    }

    # List of controllers to test
    controllers = [
        # "Sampling",
        # "Feedback Sampling",
        "iLQG"
    ]

    # Number of runs per (task, controller) pair
    num_runs = 3

    # Prepare a data structure to store all cost results
    # Example shape: results[task][controller] = [run1_cost, run2_cost, ...]
    results = {
        task: {controller: [] for controller in controllers}
        for task in tasks_config
    }

    # We'll store each individual run in this list of dicts for later CSV usage
    # Each entry will be: {
    #     "Task": <task_name>,
    #     "Controller": <controller_name>,
    #     "RunIndex": 1..5,
    #     "Cost": <float>
    # }
    run_data_records = []

    for task in tasks_config:
        for controller in controllers:
            print(f"\n=== Evaluating {task} - {controller} for {num_runs} runs ===")
            for i in range(num_runs):
                # Read config file from ./config/<task>/config.yaml
                # if there is space in the task name, replace it with underscore
                task_file_name = task.replace(" ", "_")
                config_file = Path("./config") / task_file_name / "config.yaml"
                
                config = EvaluationConfig(
                    task=task,
                    controller=controller,
                    config_file=config_file
                )
                if task == "Quadruped Hill":
                    config.model_path = Path('/home/pcy/Research/code/mujoco_mpc-fork/build/mjpc/tasks/quadruped/task_hill.xml')
                try:
                    cost = eval_main(config)
                    if cost is not None:
                        results[task][controller].append(cost)
                        run_data_records.append({
                            "Task": task,
                            "Controller": controller,
                            "RunIndex": i + 1,
                            "Cost": cost
                        })
                        print(f"Run {i+1}/{num_runs} => cost: {cost:.4f}")
                    else:
                        print(f"Run {i+1}/{num_runs} => Invalid cost returned.")
                except Exception as e:
                    print(f"Run {i+1}/{num_runs} failed with error:\n{e}")
                    continue

    # ================== Analyze results and print a table ==================
    print("\n================== Final Results Table ==================")
    print("Task              Controller                Mean ± Std Dev (over 5 runs)")
    for task in tasks_config:
        for controller in controllers:
            cost_list = results[task][controller]
            if len(cost_list) == num_runs:
                mean_val = statistics.mean(cost_list)
                std_val = statistics.pstdev(cost_list)  # population stdev or sample stdev
                print(f"{task:<18} {controller:<25} {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"{task:<18} {controller:<25} Not enough data")

    # ================== Save all data to CSV ==================
    # We'll create or overwrite a CSV file in the current directory
    csv_path = Path("evaluation_results.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Task", "Controller", "RunIndex", "Cost"])
        writer.writeheader()
        writer.writerows(run_data_records)

    print(f"\nAll run data has been saved to: {csv_path.absolute()}")
    print("Done.")


if __name__ == "__main__":
    main()
