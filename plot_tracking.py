#!/usr/bin/env python3
"""
Plot desired vs actual states for the hand tracking simulation.
Creates position and orientation plots for palm and each finger.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def quaternion_to_euler(w, x, y, z):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in radians."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1,
                     np.copysign(np.pi / 2, sinp),
                     np.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def plot_position_tracking(df, body_name, ax_x, ax_y, ax_z):
    """Plot position tracking for a specific body."""
    body_df = df[df['body_name'] == body_name]

    if len(body_df) == 0:
        print(f"Warning: No data found for {body_name}")
        return

    # Convert pandas Series to numpy arrays
    time = np.array(body_df['timestamp_ms'].values) / 1000.0  # Convert to seconds

    des_pos_x = np.array(body_df['des_pos_x'].values)
    des_pos_y = np.array(body_df['des_pos_y'].values)
    des_pos_z = np.array(body_df['des_pos_z'].values)

    act_pos_x = np.array(body_df['act_pos_x'].values)
    act_pos_y = np.array(body_df['act_pos_y'].values)
    act_pos_z = np.array(body_df['act_pos_z'].values)

    # X position
    ax_x.plot(time, des_pos_x, 'b-', label='Desired', linewidth=2, alpha=0.7)
    ax_x.plot(time, act_pos_x, 'r--', label='Actual', linewidth=2)
    ax_x.set_ylabel('X Position (m)', fontsize=10)
    ax_x.grid(True, alpha=0.3)
    ax_x.legend(loc='upper right', fontsize=8)

    # Y position
    ax_y.plot(time, des_pos_y, 'b-', label='Desired', linewidth=2, alpha=0.7)
    ax_y.plot(time, act_pos_y, 'r--', label='Actual', linewidth=2)
    ax_y.set_ylabel('Y Position (m)', fontsize=10)
    ax_y.grid(True, alpha=0.3)

    # Z position
    ax_z.plot(time, des_pos_z, 'b-', label='Desired', linewidth=2, alpha=0.7)
    ax_z.plot(time, act_pos_z, 'r--', label='Actual', linewidth=2)
    ax_z.set_ylabel('Z Position (m)', fontsize=10)
    ax_z.set_xlabel('Time (s)', fontsize=10)
    ax_z.grid(True, alpha=0.3)

def plot_orientation_tracking(df, body_name, ax_roll, ax_pitch, ax_yaw):
    """Plot orientation tracking for a specific body."""
    body_df = df[df['body_name'] == body_name]

    if len(body_df) == 0:
        print(f"Warning: No data found for {body_name}")
        return

    # Convert pandas Series to numpy arrays
    time = np.array(body_df['timestamp_ms'].values) / 1000.0  # Convert to seconds

    # Extract quaternion data as numpy arrays
    des_quat_w = np.array(body_df['des_quat_w'].values)
    des_quat_x = np.array(body_df['des_quat_x'].values)
    des_quat_y = np.array(body_df['des_quat_y'].values)
    des_quat_z = np.array(body_df['des_quat_z'].values)

    act_quat_w = np.array(body_df['act_quat_w'].values)
    act_quat_x = np.array(body_df['act_quat_x'].values)
    act_quat_y = np.array(body_df['act_quat_y'].values)
    act_quat_z = np.array(body_df['act_quat_z'].values)

    # Convert quaternions to Euler angles (in radians)
    des_roll, des_pitch, des_yaw = quaternion_to_euler(
        des_quat_w, des_quat_x, des_quat_y, des_quat_z
    )

    act_roll, act_pitch, act_yaw = quaternion_to_euler(
        act_quat_w, act_quat_x, act_quat_y, act_quat_z
    )

    # Convert to degrees (element-wise for numpy arrays)
    des_roll = np.rad2deg(des_roll)
    des_pitch = np.rad2deg(des_pitch)
    des_yaw = np.rad2deg(des_yaw)
    act_roll = np.rad2deg(act_roll)
    act_pitch = np.rad2deg(act_pitch)
    act_yaw = np.rad2deg(act_yaw)

    # Roll
    ax_roll.plot(time, des_roll, 'b-', label='Desired', linewidth=2, alpha=0.7)
    ax_roll.plot(time, act_roll, 'r--', label='Actual', linewidth=2)
    ax_roll.set_ylabel('Roll (deg)', fontsize=10)
    ax_roll.grid(True, alpha=0.3)
    ax_roll.legend(loc='upper right', fontsize=8)

    # Pitch
    ax_pitch.plot(time, des_pitch, 'b-', label='Desired', linewidth=2, alpha=0.7)
    ax_pitch.plot(time, act_pitch, 'r--', label='Actual', linewidth=2)
    ax_pitch.set_ylabel('Pitch (deg)', fontsize=10)
    ax_pitch.grid(True, alpha=0.3)

    # Yaw
    ax_yaw.plot(time, des_yaw, 'b-', label='Desired', linewidth=2, alpha=0.7)
    ax_yaw.plot(time, act_yaw, 'r--', label='Actual', linewidth=2)
    ax_yaw.set_ylabel('Yaw (deg)', fontsize=10)
    ax_yaw.set_xlabel('Time (s)', fontsize=10)
    ax_yaw.grid(True, alpha=0.3)

def main():
    # Load the CSV data
    log_file = Path(__file__).parent / "logs" / "state_tracking.csv"

    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        print("Please run the simulation first to generate the log file.")
        return

    df = pd.read_csv(log_file)
    print(f"Loaded {len(df)} data points from {log_file}")

    # Get unique body names
    bodies = df['body_name'].unique()
    print(f"Bodies in log: {bodies}")

    # Create figure for position tracking
    fig_pos, axes_pos = plt.subplots(len(bodies), 3, figsize=(15, 3*len(bodies)))
    fig_pos.suptitle('Position Tracking: Desired vs Actual', fontsize=14, fontweight='bold')

    # Create figure for orientation tracking
    fig_ori, axes_ori = plt.subplots(len(bodies), 3, figsize=(15, 3*len(bodies)))
    fig_ori.suptitle('Orientation Tracking: Desired vs Actual', fontsize=14, fontweight='bold')

    # Handle single body case (axes won't be 2D array)
    if len(bodies) == 1:
        axes_pos = axes_pos.reshape(1, -1)
        axes_ori = axes_ori.reshape(1, -1)

    # Plot each body
    for i, body in enumerate(bodies):
        # Position plots
        ax_x, ax_y, ax_z = axes_pos[i]
        ax_x.set_title(f'{body} Position', fontweight='bold', fontsize=11)
        plot_position_tracking(df, body, ax_x, ax_y, ax_z)

        # Orientation plots
        ax_roll, ax_pitch, ax_yaw = axes_ori[i]
        ax_roll.set_title(f'{body} Orientation', fontweight='bold', fontsize=11)
        plot_orientation_tracking(df, body, ax_roll, ax_pitch, ax_yaw)

    # Adjust layout
    fig_pos.tight_layout()
    fig_ori.tight_layout()

    # Save figures
    output_dir = Path(__file__).parent / "logs"
    output_dir.mkdir(exist_ok=True)

    pos_file = output_dir / "position_tracking.png"
    ori_file = output_dir / "orientation_tracking.png"

    fig_pos.savefig(pos_file, dpi=150, bbox_inches='tight')
    fig_ori.savefig(ori_file, dpi=150, bbox_inches='tight')

    print(f"\nPlots saved to:")
    print(f"  - {pos_file}")
    print(f"  - {ori_file}")

    # Show plots
    plt.show()

if __name__ == "__main__":
    main()
