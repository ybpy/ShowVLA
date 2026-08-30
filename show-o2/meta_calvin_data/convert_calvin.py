import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import json
import re
import h5py
import time
import numpy as np
from pickle import dumps, loads
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import mediapy as media


def normalize_inst(inst, max_len=120):
    """Normalize language instruction into a filesystem-safe filename stem."""
    s = str(inst).strip().lower()
    s = re.sub(r'[^\w\-]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s[:max_len]


def encode_frames_to_jpeg_bytes(frames):
    """Compress RGB frames back to JPEG byte arrays for HDF5 storage."""
    encoded = np.empty(len(frames), dtype=object)
    for idx, frame in enumerate(frames):
        assert frame.dtype == np.uint8
        # Convert numpy array to PIL Image (expects RGB format)
        pil_image = Image.fromarray(frame)
        # Encode to JPEG bytes
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        encoded[idx] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return encoded

def combine_main_wrist_views(main_img, wrist_img,
        main_tgt_size=(224, 320), wrist_tgt_size=(112, 160), comb_size=(336, 320), wrist_at_left=False):
    """ Combine the main view image and the wrist view image into an image. """
    assert comb_size[0] == main_tgt_size[0] + wrist_tgt_size[0]
    assert comb_size[1] == main_tgt_size[1]

    # Resize
    main_img = np.array(Image.fromarray(main_img).resize((main_tgt_size[1], main_tgt_size[0]), Image.BILINEAR))
    wrist_img = np.array(Image.fromarray(wrist_img).resize((wrist_tgt_size[1], wrist_tgt_size[0]), Image.BILINEAR))

    comb_img = np.zeros((comb_size[0], comb_size[1], 3), dtype=np.uint8)
    comb_img[:main_tgt_size[0]] = main_img
    if wrist_at_left:
        comb_img[main_tgt_size[0]: , :wrist_tgt_size[1]] = wrist_img
    else:
        comb_img[main_tgt_size[0]: , wrist_tgt_size[1]:] = wrist_img

    return comb_img


def setup_seed(seed):
    np.random.seed(seed)

def save_to_lmdb(output_dir, input_dir, dataset_name, metainfo_json_out_path, start_index=0):
    annotations = np.load(os.path.join(input_dir, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True).tolist()['language']['ann']
    start_end_ids = np.load(os.path.join(input_dir, 'lang_annotations/auto_lang_ann.npy'), allow_pickle=True).tolist()['info']['indx']

    src_name = '_'.join(input_dir.split('/')[-2:])

    try:
        meta_json = json.load(open(metainfo_json_out_path))
        cur_episode = meta_json['num_ep']
    except:
        meta_json = {
            "dataset_name": dataset_name,
            "language_instruction_key": "language_instruction",
            # "observation_key": ['rgb_main', 'rgb_wrist'],
            "observation_key": ['rgb_comb'],
            "num_ep": 0,
            "datalist": []
        }
        cur_episode = 0
    
    pbar = tqdm(initial=start_index, total=len(start_end_ids))
    for index, (start, end) in enumerate(start_end_ids):
        if index < start_index:
            continue

        new_data_path = os.path.join(output_dir, f"{src_name}_{index}.hdf5")

        print(f'cur_episode: {cur_episode}')
        print(f'[{index}/{len(start_end_ids)}]', flush=True)
        inst = annotations[index]

        print(f'{inst}', flush=True)
        
        list_rgb_static = []
        list_rgb_gripper = []
        list_proprio = []
        for i in range(start, end+1):
            frame = np.load(os.path.join(input_dir, f'episode_{i:07}.npz'))
            list_rgb_static.append(frame['rgb_static'])
            list_rgb_gripper.append(frame['rgb_gripper'])
            list_proprio.append(frame['robot_obs'])

        # rgb_main_bytes = encode_frames_to_jpeg_bytes(list_rgb_static)
        # rgb_wrist_bytes = encode_frames_to_jpeg_bytes(list_rgb_gripper)

        comb_images = [combine_main_wrist_views(x, y) for x, y in zip(
            list_rgb_static, list_rgb_gripper)]
        rgb_comb_bytes = encode_frames_to_jpeg_bytes(comb_images)

        str_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(new_data_path, "w") as h5_file:
            h5_file.create_dataset("language_instruction", data=inst, dtype=str_dtype)
            vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
            # h5_file.create_dataset("rgb_main", data=rgb_main_bytes, dtype=vlen_uint8)
            # h5_file.create_dataset("rgb_wrist", data=rgb_wrist_bytes, dtype=vlen_uint8)
            h5_file.create_dataset("rgb_comb", data=rgb_comb_bytes, dtype=vlen_uint8)
            h5_file.create_dataset("proprio", data=list_proprio)

        # Save combined RGB frames to MP4 (fps=10)
        h5_stem = os.path.splitext(os.path.basename(new_data_path))[0]
        mp4_path = os.path.join(output_dir, f"{h5_stem}_{normalize_inst(inst)}.mp4")
        media.write_video(mp4_path, comb_images, fps=10)

        meta_json["datalist"].append(new_data_path)
        
        cur_episode += 1
        pbar.update(1)

    meta_json["num_ep"] = cur_episode
    with open(metainfo_json_out_path, 'w') as meta_json_f:
        json.dump(meta_json, meta_json_f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transfer CALVIN dataset to lmdb format.")
    parser.add_argument("--input_dir", default='/datasets/public_data/calvin/task_ABC_D/', type=str, help="Original dataset directory.")
    parser.add_argument("--train_dir", default='/home/hyx/datasets/CalvinABC_D/Train', type=str)
    parser.add_argument("--test_dir", default='/home/hyx/datasets/CalvinABC_D/Test', type=str)
    parser.add_argument("--dataset_name", default='Calvin', type=str)
    args = parser.parse_args()

    setup_seed(0)

    input_dir = args.input_dir
    train_dir = args.train_dir
    test_dir = args.test_dir
    dataset_name = args.dataset_name
    
    metainfo_json_out_path = f"{dataset_name}_train_metainfo.json"
    os.makedirs(train_dir, exist_ok=True)
    save_to_lmdb(train_dir, os.path.join(input_dir, 'training'), dataset_name, metainfo_json_out_path, start_index=0)

    metainfo_json_out_path = f"{dataset_name}_test_metainfo.json"
    os.makedirs(test_dir, exist_ok=True)
    save_to_lmdb(test_dir, os.path.join(input_dir, 'validation'), dataset_name, metainfo_json_out_path, start_index=0)
