import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
from pathlib import Path
from termcolor import colored
import time

class MotionReader:
    def __init__(self, csv_path):
        self.offset = np.array([0.0, 0.0, 0.0])
        self.data = pd.read_csv(csv_path)
        self.body_parts = ["pelvis", "head", "ltoe", "rtoe", "lheel", "rheel", 
                          "lknee", "rknee", "lhand", "rhand", "lelbow", "relbow", 
                          "lshoulder", "rshoulder", "lhip", "rhip"]
        
        # Define connections for skeleton visualization
        self.connections = [
            ("head", "pelvis"),
            ("lshoulder", "rshoulder"),
            ("lshoulder", "lelbow"),
            ("rshoulder", "relbow"),
            ("lelbow", "lhand"),
            ("relbow", "rhand"),
            ("pelvis", "lhip"),
            ("pelvis", "rhip"),
            ("lhip", "lknee"),
            ("rhip", "rknee"),
            ("lknee", "lheel"),
            ("rknee", "rheel"),
            ("lheel", "ltoe"),
            ("rheel", "rtoe")
        ]
        
        print(colored(f"Loaded motion data with {len(self.data)} frames", "green"))
        print(colored(f"Motion keys: {self.data['motion_key'].unique()}", "green"))

    def get_body_position(self, frame, body_part):
        """Get the position of a body part at a specific frame."""
        return np.array([
            self.data.loc[frame, f"{body_part}_x"],
            self.data.loc[frame, f"{body_part}_y"],
            self.data.loc[frame, f"{body_part}_z"]
        ])

    def visualize_frame(self, frame_idx, ax=None, show=True):
        """Visualize a single frame of the motion."""
        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')

        # Plot each body part
        for body_part in self.body_parts:
            pos = self.get_body_position(frame_idx, body_part)
            ax.scatter(*pos, marker='o')

        # Draw connections between body parts
        for start_part, end_part in self.connections:
            start_pos = self.get_body_position(frame_idx, start_part)
            end_pos = self.get_body_position(frame_idx, end_part)
            ax.plot([start_pos[0], end_pos[0]],
                   [start_pos[1], end_pos[1]],
                   [start_pos[2], end_pos[2]], 'r-')

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx} (Motion: {self.data.loc[frame_idx, "motion_key"]})')

        # Set consistent axis limits
        max_range = self.get_axis_limits()
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 2*max_range])  # Assuming Z is up

        if show:
            plt.show()

    def get_axis_limits(self):
        """Calculate consistent axis limits across all frames."""
        all_coords = []
        for body_part in self.body_parts:
            all_coords.extend([
                self.data[f"{body_part}_x"].values,
                self.data[f"{body_part}_y"].values,
                self.data[f"{body_part}_z"].values
            ])
        all_coords = np.concatenate(all_coords)
        return np.max(np.abs(all_coords)) * 1.2

    def animate_motion(self, output_path=None, fps=30):
        """Create an animation of the motion."""
        print(colored("Creating animation...", "green"))
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        if output_path:
            from matplotlib.animation import FuncAnimation, PillowWriter
            
            def update(frame):
                ax.clear()
                self.visualize_frame(frame, ax, show=False)
                return ax,

            anim = FuncAnimation(fig, update, frames=len(self.data),
                               interval=1000/fps, blit=False)
            
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
            print(colored(f"Animation saved to: {output_path}", "green"))
        else:
            # Real-time visualization
            for frame in range(len(self.data)):
                ax.clear()
                self.visualize_frame(frame, ax, show=False)
                plt.pause(1/fps)
                
        plt.close()

    def get_qpos_data(self, frame_idx):
        """Get the qpos data for a specific frame."""
        frame_data = self.data.iloc[frame_idx]
        qpos_cols = [col for col in self.data.columns if col.startswith('qpos_')]
        return np.array([frame_data[col] for col in qpos_cols])

    def export_to_mujoco_xml(self, output_path, motion_key=None):
        """Export motion data to MuJoCo XML format with keyframes.
        Args:
            output_path: Path to save the XML file
            motion_key: Optional specific motion key to export. If None, exports all.
        """
        data_to_export = self.data if motion_key is None else self.data[self.data['motion_key'] == motion_key]
        
        # XML header and root
        xml_content = ['<?xml version="1.0" encoding="UTF-8"?>',
                      '<mujoco>',
                      '  <keyframe>']
        
        for idx, frame in data_to_export.iterrows():
            # Get mocap positions
            mpos_values = []
            for body_part in self.body_parts:
                mpos = np.array([
                    frame[f"{body_part}_x"],
                    frame[f"{body_part}_y"],
                    frame[f"{body_part}_z"]
                ])
                mpos_values.extend(mpos + self.offset)
            mpos_str = " ".join(f"{val:.5f}" for val in mpos_values)
            
            # Get qpos values
            qpos_cols = [col for col in self.data.columns if col.startswith('qpos_')]
            qpos_str = " ".join(f"{frame[col]:.5f}" for col in qpos_cols)
            
            # Default qvel (zeros) matching qpos dimension
            qvel_dim = len(qpos_cols) - 1
            qvel_str = " ".join(["0.0"] * qvel_dim)

            # check if frame is the same as the previous frame, if so, skip
            # current_frame = frame['frame']
            # last_frame = self.data.iloc[idx - 1]['frame']
            # if current_frame == last_frame:
            #     continue
            # if current_frame - last_frame > 1:
            #     print(f"[Warning] Frame {current_frame} is {current_frame - last_frame} frames away from the previous frame")

            # Create key element
            key_name = f"{frame['motion_key']}_{idx + 1}"
            key_line = f'    <key name="{key_name}" mpos="{mpos_str}" qpos="{qpos_str}" qvel="{qvel_str}" />'
            xml_content.append(key_line)
        
        # Close tags
        xml_content.extend(['  </keyframe>',
                          '</mujoco>'])
        
        # Write to file
        output_path = Path(output_path)
        output_path.write_text('\n'.join(xml_content))
        print(colored(f"Exported motion data to: {output_path}", "green"))

    def print_frame_info(self, frame_idx):
        """Print detailed information about a specific frame."""
        frame_data = self.data.iloc[frame_idx]
        print(colored(f"\nFrame {frame_idx} Information:", "cyan"))
        print(f"Motion Key: {frame_data['motion_key']}")
        print(f"Timestamp: {frame_data['timestamp']:.3f}")
        print(f"Motion ID: {frame_data['motion_id']}")
        
        print("\nBody Part Positions:")
        for body_part in self.body_parts:
            pos = self.get_body_position(frame_idx, body_part)
            print(f"{body_part:10s}: ({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")
            
        print("\nQPos Data:")
        qpos = self.get_qpos_data(frame_idx)
        for i, val in enumerate(qpos):
            print(f"qpos_{i:02d}: {val:6.3f}")


def main():
    # parser = argparse.ArgumentParser(description='Read and visualize motion tracking data')
    # parser.add_argument('csv_file', type=str, help='Path to the CSV file containing motion data')
    # parser.add_argument('--frame', type=int, help='Visualize a specific frame')
    # parser.add_argument('--animate', action='store_true', help='Create an animation of the motion')
    # parser.add_argument('--output', type=str, help='Output path for animation (if --animate is used)')
    # parser.add_argument('--fps', type=int, default=30, help='FPS for animation (default: 30)')
    # parser.add_argument('--export-xml', type=str, help='Export to MuJoCo XML file')
    # parser.add_argument('--motion-key', type=str, help='Specific motion key to export to XML')

    file_name = "down_box_to_walk"
    fps = 30
    csv_file = f"../data/{file_name}.csv"
    motion_key = None
    animate = False
    export_xml = f"../data/{file_name}.xml"
    
    reader = MotionReader(csv_file)
    
    # if args.frame is not None:
    #     if 0 <= args.frame < len(reader.data):
    #         reader.print_frame_info(args.frame)
    #         reader.visualize_frame(args.frame)
    #     else:
    #         print(colored(f"Error: Frame index {args.frame} is out of range", "red"))
    
    if animate:
        reader.animate_motion(None, fps)
        
    if export_xml:
        reader.export_to_mujoco_xml(export_xml, motion_key)

if __name__ == "__main__":
    main()