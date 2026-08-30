import os
import re
import argparse
import json
import h5py
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import mediapy as media
import tensorflow_datasets as tfds

BRIDGE_FPS = 5  # Bridge control frequency (Hz), matches BridgeHandler


def encode_frames_to_jpeg_bytes(frames):
    """Compress RGB frames back to JPEG byte arrays for HDF5 storage."""
    encoded = np.empty(len(frames), dtype=object)
    for idx, frame in enumerate(frames):
        assert frame.dtype == np.uint8
        pil_image = Image.fromarray(frame)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG")
        encoded[idx] = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return encoded


def combine_main_zero_bottom(
    main_img,
    main_tgt_size=(224, 320),
    wrist_tgt_size=(112, 160),
    comb_size=(336, 320),
):
    """Combine main view (image_0) on top; bottom wrist region is zeros."""
    assert comb_size[0] == main_tgt_size[0] + wrist_tgt_size[0]
    assert comb_size[1] == main_tgt_size[1]

    main_img = np.array(
        Image.fromarray(main_img).resize((main_tgt_size[1], main_tgt_size[0]), Image.BILINEAR)
    )
    comb_img = np.zeros((comb_size[0], comb_size[1], 3), dtype=np.uint8)
    comb_img[: main_tgt_size[0]] = main_img
    return comb_img


def sanitize_lang_for_filename(lang: str, max_len: int = 80) -> str:
    """Turn language instruction into a filesystem-safe slug."""
    slug = lang.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug, flags=re.UNICODE)
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    if not slug:
        slug = "no_instruction"
    return slug[:max_len].rstrip("_")


def mp4_path_from_h5(h5_path: str, language_instruction: str) -> str:
    base, _ = os.path.splitext(h5_path)
    lang_slug = sanitize_lang_for_filename(language_instruction)
    return f"{base}_{lang_slug}.mp4"


def setup_seed(seed):
    np.random.seed(seed)


def is_single_word_instruction(lang: str) -> bool:
    """True if the instruction has no spaces (only one word)."""
    return lang.count(" ") == 0


def episode_to_arrays(episode):
    steps = list(episode["steps"])
    if len(steps) < 2:
        return None

    lang = steps[0]["language_instruction"].numpy().decode().strip()
    if not lang or is_single_word_instruction(lang):
        return None

    proprio_list = []
    action_list = []
    rgb_images = []

    for step in steps:
        obs = step["observation"]
        rgb_images.append(combine_main_zero_bottom(obs["image_0"].numpy()))
        proprio_list.append(np.array(obs["state"], dtype=np.float32))
        action_list.append(np.array(step["action"], dtype=np.float32))

    return {
        "language_instruction": lang,
        "rgb_comb": rgb_images,
        "proprio": np.stack(proprio_list, axis=0),
        "action": np.stack(action_list, axis=0),
    }


def convert_bridge_split(
    data_dir,
    output_dir,
    split,
    dataset_name,
    metainfo_json_out_path,
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(metainfo_json_out_path), exist_ok=True)

    meta_json = {
        "dataset_name": dataset_name,
        "language_instruction_key": "language_instruction",
        "observation_key": ["rgb_comb"],
        "num_ep": 0,
        "datalist": [],
    }
    cur_episode = 0

    builder = tfds.builder_from_directory(data_dir)
    split_info = builder.info.splits[split]
    num_examples = split_info.num_examples
    ds = builder.as_dataset(split=split, shuffle_files=False)

    pbar = tqdm(total=num_examples)
    for index, episode in enumerate(ds):
        h5_path = os.path.join(output_dir, f"bridge_{split}_{index}.hdf5")

        arrays = episode_to_arrays(episode)
        if arrays is None:
            pbar.update(1)
            continue

        rgb_comb_bytes = encode_frames_to_jpeg_bytes(arrays["rgb_comb"])
        str_dtype = h5py.string_dtype(encoding="utf-8")

        with h5py.File(h5_path, "w") as h5_file:
            h5_file.create_dataset("language_instruction", data=arrays["language_instruction"], dtype=str_dtype)
            vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
            h5_file.create_dataset("rgb_comb", data=rgb_comb_bytes, dtype=vlen_uint8)
            h5_file.create_dataset("proprio", data=arrays["proprio"])
            h5_file.create_dataset("action", data=arrays["action"])

        mp4_path = mp4_path_from_h5(h5_path, arrays["language_instruction"])
        media.write_video(mp4_path, arrays["rgb_comb"], fps=BRIDGE_FPS)

        meta_json["datalist"].append(h5_path)
        cur_episode += 1
        meta_json["num_ep"] = cur_episode

        num_frames = len(arrays["rgb_comb"])
        print(
            f"[{cur_episode}][{index}/{num_examples}] frames={num_frames} {arrays['language_instruction']}",
            flush=True,
        )
        pbar.update(1)

    with open(metainfo_json_out_path, "w") as meta_json_f:
        json.dump(meta_json, meta_json_f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Bridge RLDS dataset to ShowVLA HDF5 format.")
    parser.add_argument(
        "--input_dir",
        default="/datasets2/hyx_data/bridge_orig/1.0.0",
        type=str,
        help="Original Bridge RLDS dataset directory.",
    )
    parser.add_argument(
        "--train_dir",
        default="/home/hyx/datasets/Bridge/Train",
        type=str,
        help="Output directory for converted train HDF5 files.",
    )
    parser.add_argument(
        "--val_dir",
        default="/home/hyx/datasets/Bridge/Val",
        type=str,
        help="Output directory for converted validation HDF5 files.",
    )
    parser.add_argument("--dataset_name", default="Bridge", type=str)
    parser.add_argument("--split", default="both", choices=["train", "val", "both"], help="Which split to convert.")
    args = parser.parse_args()

    setup_seed(0)

    if args.split in ("train", "both"):
        os.makedirs(args.train_dir, exist_ok=True)
        convert_bridge_split(
            args.input_dir,
            args.train_dir,
            "train",
            args.dataset_name,
            f"./train/{args.dataset_name}_train_metainfo.json",
        )

    if args.split in ("val", "both"):
        os.makedirs(args.val_dir, exist_ok=True)
        convert_bridge_split(
            args.input_dir,
            args.val_dir,
            "val",
            args.dataset_name,
            f"./val/{args.dataset_name}_val_metainfo.json",
        )
