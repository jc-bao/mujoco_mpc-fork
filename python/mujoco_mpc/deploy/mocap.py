import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
from multiprocessing import shared_memory
from pyvicon_datastream import tools
import struct
from loop_rate_limiters import RateLimiter
import tyro
from dataclasses import dataclass

from utils import pack_mocap_data
from config import G1PositionConfig, Go2PositionConfig, H1_2PositionConfig, H1_2_simpleConfig, H1Config, ObjectConfig, H1_maniConfig

@dataclass
class Args:
    robot_name: str = "h1_mani"
class ViconDemo:
    """
    Vicon data acquisition and filtering

    Note:
    - The quaternion is in the order of [x, y, z, w]. So need to convert it to the order of [w, x, y, z] before packing.
    - Vicon's euler is absolute, which we should use "XYZ" in scipy.spatial.transform.Rotation, instead of "xyz", which is relative.
    """
    def __init__(self, robot_name="g1"):
        if robot_name == "g1":
            self.config = G1PositionConfig()
        elif robot_name == "go2":
            self.config = Go2PositionConfig()
        elif robot_name == "h1_2":
            self.config = H1_2PositionConfig()
        elif robot_name == "h1_2_simple":
            self.config = H1_2_simpleConfig()
        elif robot_name == "h1":
            self.config = H1Config()
        elif robot_name == "h1_mani":
            self.config = H1_maniConfig()
        elif robot_name == "object":
            self.config = ObjectConfig()
        else:
            raise ValueError(f"Robot {robot_name} not supported")

        # Vicon DataStream IP and object name
        self.vicon_tracker_ip = self.config.vicon_tracker_ip
        self.vicon_object_name = self.config.vicon_object_name

        # Connect to Vicon DataStream
        self.tracker = tools.ObjectTracker(self.vicon_tracker_ip)
        if self.tracker.is_connected:
            print(f"Connected to Vicon DataStream at {self.vicon_tracker_ip}")
        else:
            print(f"Failed to connect to Vicon DataStream at {self.vicon_tracker_ip}")
            raise Exception(f"Connection to {self.vicon_tracker_ip} failed")

        # Initialize previous values for velocity computation
        self.prev_time = None
        self.prev_position = None
        self.prev_quaternion = None

        # Low-pass filter parameters
        self.cutoff_freq = 5.0  # Cut-off frequency of the filter (Hz)
        self.filter_order = 2
        self.fs = 1 / self.config.dt_real  # Sampling frequency (Hz)
        self.b, self.a = butter(
            self.filter_order, self.cutoff_freq / (0.5 * self.fs), btype="low"
        )

        # Initialize data buffers for filtering
        self.vel_buffer = []
        self.omega_buffer = []

        
        # Initialize shared memory
        if robot_name == "object":
            self.shared_mem_name = "object_state_shm"
            self.shared_mem_size = 8 + 13 * 8  # 8 bytes for utime (int64), 13 float64s (13*8 bytes)
        else:
            self.shared_mem_name = "mocap_state_shm"
            self.shared_mem_size = 8 + 13 * 8  # 8 bytes for utime (int64), 13 float64s (13*8 bytes)
        try:
            self.state_shm = shared_memory.SharedMemory(name=self.shared_mem_name, create=True, size=self.shared_mem_size)
            print(f"Attach to shared memory '{self.shared_mem_name}' of size {self.shared_mem_size} bytes.")
        except FileExistsError:
            print(f"shared memory does not exist")
        self.state_buffer = self.state_shm.buf

    def get_vicon_data(self):
        position = self.tracker.get_position(self.vicon_object_name)
        # print(f"position: {position[2][0]}")
        if not position:
            print(f"Cannot get the pose of `{self.vicon_object_name}`.")
            return None, None, None

        try:
            obj = position[2][0]
            _, _, x, y, z, roll, pitch, yaw = obj
            current_time = time.time()

            # Position and orientation
            position = np.array([x, y, z]) / 1000.0

            position += self.config.mocap_offset
            rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
            quaternion = rotation.as_quat()  # [x, y, z, w]

            return current_time, position, quaternion
        except Exception as e:
            print(f"Error retrieving Vicon data: {e}")
            return None, None, None

    def compute_velocities(self, current_time, position, quaternion):
        # Initialize velocities
        linear_velocity = np.zeros(3)
        angular_velocity = np.zeros(3)

        if (
            self.prev_time is not None
            and self.prev_position is not None
            and self.prev_quaternion is not None
        ):
            dt = current_time - self.prev_time
            if dt > 0:
                # Linear velocity
                dp = position - self.prev_position
                linear_velocity = dp / dt

                # Angular velocity
                prev_rot = R.from_quat(self.prev_quaternion)
                curr_rot = R.from_quat(quaternion)
                delta_rot = curr_rot * prev_rot.inv()
                delta_angle = delta_rot.as_rotvec()
                angular_velocity = delta_angle / dt
        else:
            # First data point; velocities remain zero
            pass

        # Update previous values
        self.prev_time = current_time
        self.prev_position = position
        self.prev_quaternion = quaternion

        return linear_velocity, angular_velocity

    def low_pass_filter(self, data_buffer, new_data):
        # Append new data to the buffer
        data_buffer.append(new_data)
        # Keep only the last N samples (buffer size)
        buffer_size = int(self.fs / self.cutoff_freq) * 3
        if len(data_buffer) > buffer_size:
            data_buffer.pop(0)
        # Apply low-pass filter if enough data points are available
        if len(data_buffer) >= self.filter_order + 1:
            data_array = np.array(data_buffer)
            filtered_data = lfilter(self.b, self.a, data_array, axis=0)[-1]
            return filtered_data
        else:
            return new_data  # Not enough data to filter; return the new data as is

    def main_loop(self):
        print("Starting Vicon data acquisition...")
        rate_limiter = RateLimiter(frequency=1 / self.config.dt_real)
        try:
            while True:
                # Get Vicon data
                current_time, position, quaternion = self.get_vicon_data()
                if position is None:
                    rate_limiter.sleep()
                    continue

                # Compute velocities
                linear_velocity, angular_velocity = self.compute_velocities(
                    current_time, position, quaternion
                )

                # Apply low-pass filter to velocities
                filtered_linear_velocity = self.low_pass_filter(
                    self.vel_buffer, linear_velocity
                )
                filtered_angular_velocity = self.low_pass_filter(
                    self.omega_buffer, angular_velocity
                )

                # Prepare data to pack
                x, y, z, w = quaternion
                qmocap = np.concatenate([position, np.array([w, x, y, z])])
                qdmocap = np.concatenate([filtered_linear_velocity, filtered_angular_velocity])
                self.state_buffer[:] = pack_mocap_data(self.state_buffer, current_time, qmocap, qdmocap)

                # Sleep to mimic sampling rate
                rate_limiter.sleep()

        except KeyboardInterrupt:
            print("Exiting Vicon data acquisition.")
        finally:
            # Close and unlink shared memory
            try:
                self.state_shm.close()
                print(f"Shared memory '{self.shared_mem_name}' closed")
            except:
                pass


if __name__ == "__main__":
    args = tyro.cli(Args)
    vicon_demo = ViconDemo(robot_name=args.robot_name)
    vicon_demo.main_loop()
