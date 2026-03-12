import os
import argparse
import json
import h5py
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import mediapy as media
import sys

# Add ShowVLA to path to import datasets_vla.utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets_vla.utils import euler_to_rotate6d

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

def combine_main_wrist_views(main_img, wrist_0_img, wrist_1_img,
        main_tgt_size=(224, 320), wrist_tgt_size=(112, 160), comb_size=(336, 320)):
    """ 
    Combine the main view image and two wrist view images into one image.
    wrist_0 (left wrist) at bottom-left, wrist_1 (right wrist) at bottom-right.
    """
    assert comb_size[0] == main_tgt_size[0] + wrist_tgt_size[0]
    assert comb_size[1] == main_tgt_size[1]
    assert wrist_tgt_size[1] * 2 == main_tgt_size[1]

    # Resize
    main_img = np.array(Image.fromarray(main_img).resize((main_tgt_size[1], main_tgt_size[0]), Image.BILINEAR))
    wrist_0_img = np.array(Image.fromarray(wrist_0_img).resize((wrist_tgt_size[1], wrist_tgt_size[0]), Image.BILINEAR))
    wrist_1_img = np.array(Image.fromarray(wrist_1_img).resize((wrist_tgt_size[1], wrist_tgt_size[0]), Image.BILINEAR))

    comb_img = np.zeros((comb_size[0], comb_size[1], 3), dtype=np.uint8)
    # Main view on top
    comb_img[:main_tgt_size[0], :] = main_img
    # Wrist 0 (left) on bottom-left
    comb_img[main_tgt_size[0]:, :wrist_tgt_size[1]] = wrist_0_img
    # Wrist 1 (right) on bottom-right
    comb_img[main_tgt_size[0]:, wrist_tgt_size[1]:] = wrist_1_img

    return comb_img

def setup_seed(seed):
    np.random.seed(seed)

def convert_jaka_to_hdf5(data_dir, output_dir, metainfo_json_out_path, crop_main, speed_up=2, image_stream_offset=1):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        with open(metainfo_json_out_path, 'r') as f:
            meta_json = json.load(f)
        cur_episode = meta_json['num_ep']
    except:
        meta_json = {
            "dataset_name": "JAKA",
            "data_dirs": [],
            "language_instruction_key": "language_instruction",
            "observation_key": ['rgb_comb'],
            "num_ep": 0,
            "datalist": [],
        }
        cur_episode = 0

    if "data_dirs" not in meta_json:
        meta_json["data_dirs"] = []
    
    if data_dir not in meta_json["data_dirs"]:
        meta_json["data_dirs"].append(data_dir)
    else:
        print(f"Warning: data_dir {data_dir} already processed!")

    print(f"Converting {data_dir}")

    list_ep_folders = []
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            if d.startswith('episode_'):
                list_ep_folders.append(os.path.join(root, d))
    list_ep_folders.sort()
    
    data_dir_basename = os.path.basename(data_dir.rstrip(os.sep))

    pbar = tqdm(list_ep_folders)
    for ep_cnt, episode_dir in enumerate(pbar):
        pbar.set_description(f"[{ep_cnt+1}/{len(list_ep_folders)}] {episode_dir}")
        # Create h5_filename based on data_dir basename and relative path
        rel_path = os.path.relpath(episode_dir, data_dir)
        h5_filename = f"{data_dir_basename}_{rel_path.replace(os.sep, '_')}.hdf5"
        h5_path = os.path.join(output_dir, h5_filename)
        
        if h5_path in meta_json["datalist"]:
            print(f"Warning: h5_path {h5_path} already in datalist. Skipping!")
            continue

        # Read data.json
        with open(os.path.join(episode_dir, 'data.json')) as json_f:
            json_data = json.load(json_f)
        
        task = json_data['info']['task']
        
        list_rgb_comb = []
        list_eef_xyz_rotate6d_grip = []
        
        camera_names = ['rgb_main', 'rgb_wrist_0', 'rgb_wrist_1']
        
        # Process frames
        for i, item in enumerate(json_data['data']):
            # Proprioception: eef_pose_l + act_grip_l (assuming single arm for now, or need to clarify)
            # Based on convert_lmdb_jaka.py, it's a dual-arm setup. 
            # The prompt asks for a 10-dim vector: xyz (3) + rotate6d (6) + grip (1) = 10.
            # This implies we might only be tracking one arm or a specific combined state.
            # Given the 10-dim requirement, I will use the left arm's eef_pose and gripper state.
            
            state = item['states']
            eef_pose_l = state['eef_pose_l'] # [x, y, z, roll, pitch, yaw]
            xyz = eef_pose_l[:3]
            euler = eef_pose_l[3:]
            
            # Convert euler to rotate6d using datasets_vla.utils.euler_to_rotate6d
            rotate6d = euler_to_rotate6d(np.array(euler), pattern="xyz")

            grip = 1 if item['actions']['act_grip_l'] else 0
            
            eef_10d = np.concatenate([xyz, rotate6d, [grip]])
            list_eef_xyz_rotate6d_grip.append(eef_10d)
            
            # Images
            imgs = {}
            for cam_name in camera_names:
                img_path = os.path.join(episode_dir, 'colors', f'{str(i).zfill(6)}_{cam_name}.jpg')
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1].copy() # BGR -> RGB
                
                if cam_name == 'rgb_main':
                    if img.shape == (720, 1280, 3):
                        img = img[crop_main[0]:crop_main[1], crop_main[2]:crop_main[3]]
                
                imgs[cam_name] = img
            
            comb_img = combine_main_wrist_views(imgs['rgb_main'], imgs['rgb_wrist_0'], imgs['rgb_wrist_1'])
            list_rgb_comb.append(comb_img)

        # Apply speed_up and offset (matching convert_lmdb_jaka.py logic)
        if speed_up != 1:
            list_eef_xyz_rotate6d_grip = list_eef_xyz_rotate6d_grip[::speed_up]
            list_rgb_comb = list_rgb_comb[image_stream_offset:][::speed_up]

        # Align lengths (matching convert_lmdb_jaka.py logic)
        num_steps = min(len(list_eef_xyz_rotate6d_grip) - 1, len(list_rgb_comb))
        if len(list_rgb_comb) > num_steps:
            list_rgb_comb = list_rgb_comb[:num_steps]
        list_eef_xyz_rotate6d_grip = list_eef_xyz_rotate6d_grip[:num_steps]

        if num_steps <= 0:
            continue

        rgb_comb_bytes = encode_frames_to_jpeg_bytes(list_rgb_comb)
        str_dtype = h5py.string_dtype(encoding="utf-8")
        
        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("language_instruction", data=task, dtype=str_dtype)
            vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
            h5_file.create_dataset("rgb_comb", data=rgb_comb_bytes, dtype=vlen_uint8)
            h5_file.create_dataset("eef_xyz_rotate6d_grip", data=np.array(list_eef_xyz_rotate6d_grip))

        # Save to MP4
        mp4_path = h5_path.replace(".hdf5", ".mp4")
        media.write_video(mp4_path, list_rgb_comb, fps=30)

        meta_json["datalist"].append(h5_path)
        cur_episode += 1
        print(f"[{cur_episode}][{ep_cnt+1}/{len(list_ep_folders)}] {episode_dir}", flush=True)

    meta_json["num_ep"] = cur_episode
    with open(metainfo_json_out_path, 'w') as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of raw JAKA data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output HDF5 files')
    parser.add_argument('--meta_prefix', type=str, default='JAKA', help='Prefix for metainfo json file')
    parser.add_argument('--speed_up', type=int, default=2)
    parser.add_argument('--image_stream_offset', type=int, default=1)
    parser.add_argument('--crop_main', type=str, default="(40, -140, 300, -300)")
    args = parser.parse_args()

    setup_seed(0)
    
    crop_main = eval(args.crop_main)
    metainfo_json_out_path = f"{args.meta_prefix}_metainfo.json"
    
    convert_jaka_to_hdf5(
        args.data_dir, 
        args.output_dir, 
        metainfo_json_out_path, 
        crop_main, 
        args.speed_up, 
        args.image_stream_offset
    )
