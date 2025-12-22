"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - Filp the main image

Usage:
    python regenerate.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

"""

import argparse
import json
import os
import time
from io import BytesIO

import h5py
import numpy as np
from PIL import Image
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def encode_frames_to_jpeg_bytes(frames):
    """Compress RGB frames back to JPEG byte arrays for HDF5 storage."""
    encoded = np.empty(len(frames), dtype=object)
    for idx, frame in enumerate(frames):
        assert frame.dtype == np.uint8
        # Convert numpy array to PIL Image (expects RGB format)
        pil_image = Image.fromarray(frame)
        # Encode to JPEG bytes
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=100)
        encoded[idx] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return encoded


IMAGE_RESOLUTION = 256


def is_noop(action, prev_action=None, threshold=1e-4):
    """
    Returns whether an action is a no-op action.

    A no-op action satisfies two criteria:
        (1) All action dimensions, except for the last one (gripper action), are near zero.
        (2) The gripper action is equal to the previous timestep's gripper action.

    Explanation of (2):
        Naively filtering out actions with just criterion (1) is not good because you will
        remove actions where the robot is staying still but opening/closing its gripper.
        So you also need to consider the current state (by checking the previous timestep's
        gripper action as a proxy) to determine whether the action really is a no-op.
    """
    # Special case: Previous action is None if this is the first action in the episode
    # Then we only care about criterion (1)
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold

    # Normal case: Check both criteria (1) and (2)
    gripper_action = action[-1]
    prev_gripper_action = prev_action[-1]
    return np.linalg.norm(action[:-1]) < threshold and gripper_action == prev_gripper_action


def main(args):
    print("Regenerating dataset", args.libero_task_suite)

    os.makedirs(args.libero_target_dir, exist_ok=True)

    meta_json = {
        "dataset_name": args.libero_task_suite,
        "language_instruction_key": "language_instruction",
        "observation_key": ['rgb_main', 'rgb_wrist'],
        "num_ep": 0,
        "datalist": []
    }
    metainfo_json_out_path = f"./{args.libero_task_suite}_metainfo.json"

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks

    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path)
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        for i in range(len(orig_data.keys())):
            new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo_{i}.hdf5")

            # Get demo data
            demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            actions = []
            agentview_images = [np.flip(np.flip(obs['agentview_image'], 0), 1)]
            eye_in_hand_images = [obs['robot0_eye_in_hand_image']]
            abs_action_6d = []

            # Replay original demo actions in environment and record observations
            for _, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                prev_action = actions[-1] if len(actions) > 0 else None
                if is_noop(action, prev_action):
                    print(f"\tSkipping no-op action: {action}")
                    num_noops += 1
                    continue

                # Execute demo action in environment
                obs, reward, done, info = env.step(action.tolist())

                goal_pos = env.env.robots[0].controller.goal_pos
                goal_ori = env.env.robots[0].controller.goal_ori
                assert goal_ori.shape == (3, 3)
            
                # state_ori = T.quat2axisangle(T.mat2quat(goal_ori))
                Rotate6D = np.concatenate([goal_ori[:3, 0], goal_ori[:3, 1]], axis=-1)
                abs_action_6d.append(np.concatenate([goal_pos, Rotate6D, action[-1:]]))
                agentview_images.append(np.flip(np.flip(obs['agentview_image'], 0), 1))
                eye_in_hand_images.append(obs['robot0_eye_in_hand_image'])

            # At end of episode, save replayed trajectories to new HDF5 files (only keep successes)
            if done:
                rgb_main_bytes = encode_frames_to_jpeg_bytes(agentview_images)
                rgb_wrist_bytes = encode_frames_to_jpeg_bytes(eye_in_hand_images)

                str_dtype = h5py.string_dtype(encoding="utf-8")

                with h5py.File(new_data_path, "w") as h5_file:
                    h5_file.create_dataset("language_instruction", data=task_description, dtype=str_dtype)
                    vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
                    h5_file.create_dataset("rgb_main", data=rgb_main_bytes, dtype=vlen_uint8)
                    h5_file.create_dataset("rgb_wrist", data=rgb_wrist_bytes, dtype=vlen_uint8)
                    h5_file.create_dataset("abs_action_6d", data=abs_action_6d)

                meta_json["datalist"].append(new_data_path)

                num_success += 1

            num_replays += 1

            # Count total number of successful replays so far
            print(
                f"Total # episodes replayed: {num_replays}, Total # successes: {num_success} ({num_success / num_replays * 100:.1f} %)",
                flush=True,
            )

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")

        # Close HDF5 files
        orig_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")


    meta_json["num_ep"] = num_success
    with open(metainfo_json_out_path, 'w') as meta_json_f:
        json.dump(meta_json, meta_json_f, indent=4)

    print(f"Dataset regeneration complete! Saved new dataset at: {args.libero_target_dir}")
    print(f"Saved metainfo JSON at: {metainfo_json_out_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    args = parser.parse_args()

    # Start data regeneration
    main(args)