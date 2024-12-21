#!/usr/bin/env python3
import subprocess
import statistics
import re

def parse_cost_from_output(output, task, controller):
    """
    Given the stdout from `evaluation.py`, extract the final cost.

    The line we expect to see is something like:
        "Acrobot Feedback Sampling cost: 123.45"
    We'll use a regex to capture that 123.45.
    """
    # Regex pattern that matches for example: "Acrobot Feedback Sampling cost: 1.2345"
    pattern = rf"{task}\s+{controller}\s+cost:\s+([0-9\.\-eE]+)"
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    else:
        return None

def main():
    tasks = ["Cartpole", "Acrobot"]  # Add more tasks as desired
    controllers = ["Sampling", "iLQG", "Feedback Sampling"]  # Add more controllers as desired

    # A nested dict for storing cost results
    # results[task][controller] = list of 5 costs
    results = {
        task: {controller: [] for controller in controllers}
        for task in tasks
    }

    # Number of runs per (task, controller) pair
    num_runs = 3

    # Run each (task, controller) pair num_runs times
    for task in tasks:
        for controller in controllers:
            print(f"\n=== Evaluating {task} - {controller} for {num_runs} runs ===")
            for i in range(num_runs):
                # Build the command to call the evaluation script
                # We pass in the arguments via command-line to override the default
                cmd = [
                    "python",
                    "eval_gui.py",
                    "--task", task,
                    "--controller", controller,
                    "--total-time", "5",  # Specify total time as 10
                    # Optionally add overrides, e.g., --record-dt 0.02
                ]
                process = subprocess.run(cmd, capture_output=True, text=True)
                stdout = process.stdout
                stderr = process.stderr

                # Check if something went wrong
                if process.returncode != 0:
                    print(f"Run {i+1}/{num_runs} failed with error:")
                    print(stderr)
                    continue

                # Parse cost from output
                cost = parse_cost_from_output(stdout, task, controller)
                if cost is not None:
                    results[task][controller].append(cost)
                    print(f"Run {i+1}/{num_runs} => cost: {cost:.4f}")
                else:
                    print(f"Run {i+1}/{num_runs} => Could not parse cost from output:")
                    print(stdout)

    # Now analyze results and print a table
    print("\n================== Final Results Table ==================")
    print("Task / Controller          Mean ± Std Dev (over 5 runs)")
    for task in tasks:
        for controller in controllers:
            cost_list = results[task][controller]
            if len(cost_list) == num_runs:
                mean_val = statistics.mean(cost_list)
                std_val = statistics.pstdev(cost_list)  # population stdev or pstdev
                print(f"{task:<20} / {controller:<20} => {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"{task:<20} / {controller:<20} => Not enough data collected")


if __name__ == "__main__":
    main()
