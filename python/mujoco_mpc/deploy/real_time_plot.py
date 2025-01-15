import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import time

# Number of points for the sine and cosine data
num_points = 100

# Generate x values
x = np.linspace(0, 2 * np.pi, num_points)

# Static sine data
y_sin = np.sin(x)

# Buffer for dynamic cosine data
y_cos = np.zeros(num_points)

# Signal handling for graceful termination
def signal_handler(sig, frame):
    print("\nTerminating the plot...")
    plt.close('all')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Create a 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# Flatten the axes array for easy iteration
axs = axs.flatten()

# Initialize lines for sine and cosine on each subplot
lines_sin = []
lines_cos = []
for ax in axs:
    line_sin, = ax.plot(x, y_sin, label='sin(x)')
    line_cos, = ax.plot(x, y_cos, label='cos(x)', linestyle='--')
    ax.legend()
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(-1.5, 1.5)
    lines_sin.append(line_sin)
    lines_cos.append(line_cos)

# Main update loop
def run_realtime_plot():
    global y_cos
    frame = 0
    plt.ion()  # Turn on interactive mode
    try:
        while True:
            # Update cosine data buffer
            y_cos[frame % num_points] = np.cos(x[frame % num_points])

            # Reset buffer after completing a full cycle
            if frame % num_points == num_points - 1:
                y_cos = np.zeros(num_points)

            # Update all cosine curves
            for line_cos in lines_cos:
                line_cos.set_ydata(y_cos)

            # Redraw the figure
            plt.pause(0.05)

            # Increment frame counter
            frame += 1
    except KeyboardInterrupt:
        print("\nTerminating the plot...")
        plt.close('all')

# Run the plot
run_realtime_plot()
