import tkinter as tk
from tkinter import ttk
import time

# Create a tkinter GUI with a scale (slider) to represent the bar
def create_bar(params_dict):
    root = tk.Tk()
    root.title("Input Bars")

    # Create a dictionary to store the scale widgets
    bars = {}

    # Create a scale widget for each parameter
    for param_name, param_config in params_dict.items():
        lower_bound = param_config.get('lower', 0)
        upper_bound = param_config.get('upper', 1) 
        step_size = param_config.get('step', 0.01)
        
        # Create scale (slider) with the specified parameters
        bar = tk.Scale(
            root,
            from_=lower_bound,
            to=upper_bound,
            resolution=step_size,
            orient="horizontal",
            length=300,
            label=f"Adjust {param_name}"
        )
        bar.pack(pady=10)
        bars[param_name] = bar

    # Create a function to keep the tkinter window running in a non-blocking way
    def update_gui():
        root.update_idletasks()
        root.update()

    return root, bars, update_gui

# Main program
def main():
    # Create the GUI elements
    params_dict = {
        "kp": {"lower": 0, "upper": 100, "step": 1},
        "kd": {"lower": 0, "upper": 100, "step": 1},
    }
    root, bars, update_gui = create_bar(params_dict)

    try:
        for i in range(1000):  # Your own for loop
            # Update the GUI to process user interactions
            update_gui()

            # Read the values from both bars (sliders)
            value1 = bars["kp"].get()
            value2 = bars["kd"].get()

            # Print the values read from the bars
            print(f"Loop {i + 1}: Value from bar1 = {value1}, Value from bar2 = {value2}")

            # Simulate some work in the loop
            time.sleep(0.01)

    finally:
        # Ensure the tkinter window is closed after the loop
        root.destroy()

if __name__ == "__main__":
    main()
