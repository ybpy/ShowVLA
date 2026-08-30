import os
import json
import argparse
import re
from collections import defaultdict


SPECIAL_SYNONYM_OVERRIDES = {
    "monitor_(computer_equipment) computer_monitor": "computer monitor",
}


def clean_lvis_category_name(name, keep_parenthetical_text=False):
    """Make LVIS synonyms shorter and more natural for text instructions."""
    if name in SPECIAL_SYNONYM_OVERRIDES:
        return SPECIAL_SYNONYM_OVERRIDES[name]
    name = name.replace("_", " ")
    if keep_parenthetical_text:
        name = re.sub(r"\s*\(([^)]*)\)", r" \1", name)
    else:
        name = re.sub(r"\s*\([^)]*\)", "", name)
    return " ".join(name.split())


def get_clean_synonyms(synonyms, keep_parenthetical_text=False):
    clean_synonyms = []
    for synonym in synonyms:
        clean_synonym = clean_lvis_category_name(synonym, keep_parenthetical_text)
        if clean_synonym and len(clean_synonym) <= 30 and clean_synonym not in clean_synonyms:
            clean_synonyms.append(clean_synonym)
    return clean_synonyms


def parse_coco_url(coco_url):
    """Extract COCO split directory and file name from coco_url.

    LVIS V1.0 splits no longer match COCO splits one-to-one (e.g. the LVIS val
    set contains images from COCO train2017). Use coco_url instead of file_name
    or a fixed img_dir to locate the image.

    Example:
        http://images.cocodataset.org/train2017/000000391895.jpg
        -> ("train2017", "000000391895.jpg")
    """
    parts = coco_url.rstrip("/").split("/")
    file_name = parts[-1]
    coco_split = parts[-2]
    return coco_split, file_name


def convert_anns_for_image(image, coco_img_root, img_id_2_anns, cat_synonyms_dict,
                            out_json_dir, num_instances_ths=8, area_ratio_ths=0.0043,
                            area_ratio_ths_big=0.008):
    """
    Convert LVIS annotations for a single image to per-image JSON format.

    Image location is derived from coco_url (COCO split + file name), not from
    a fixed img_dir tied to the LVIS split.
    """
    coco_split, file_name = parse_coco_url(image["coco_url"])
    height, width = image["height"], image["width"]
    img_id = image["id"]

    img_path = os.path.join(coco_img_root, coco_split, file_name)
    print(f"[img_id: {img_id}] {img_path}")

    cat_2_instances = img_id_2_anns[img_id]
    if len(cat_2_instances) == 0:
        print(f"[Invalid] No annotations found for image {img_id}")
        return None

    cat_2_instances_filtered = {}
    cat_synonyms_filtered = {}
    cat_is_small_filtered = {}
    for cat_name, instances in cat_2_instances.items():
        area = sum(ann["area"] for ann in instances)
        area_ratio = area / (height * width)
        is_small = area_ratio < area_ratio_ths_big
        print(f"# Instances of {cat_name}: {len(instances)}, area_ratio: {area_ratio:.4f}, is_small: {is_small}")
        if len(instances) >= num_instances_ths:
            print(f"Too many instances (>={num_instances_ths}). Discard!!!")
            continue
        if area_ratio < area_ratio_ths:
            print(f"Too small (<{area_ratio_ths}). Discard!!!")
            continue
        cat_2_instances_filtered[cat_name] = instances
        cat_is_small_filtered[cat_name] = is_small
        if cat_name in cat_synonyms_dict:
            cat_synonyms_filtered[cat_name] = cat_synonyms_dict[cat_name]

    if len(cat_2_instances_filtered) == 0:
        print(f"[Invalid] No valid annotations found for image {img_id}")
        return None

    data_dict = {
        "img_path": img_path,
        "height": height,
        "width": width,
        "anns": cat_2_instances_filtered,
        "is_small": cat_is_small_filtered,
        # cat_synonyms[cat][0] is used as the display name in text instructions
        "cat_synonyms": cat_synonyms_filtered,
    }
    out_json_path = os.path.join(out_json_dir, f"{img_id}.json")
    with open(out_json_path, "w") as json_f:
        json.dump(data_dict, json_f, indent=4)

    return out_json_path


def get_img_id_2_anns(annotations, cat_dict):
    """Build mapping: image_id -> {cat_name -> [annotation, ...]}."""
    img_id_2_anns = defaultdict(dict)
    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        cat_name = cat_dict[category_id]

        if cat_name not in img_id_2_anns[image_id]:
            img_id_2_anns[image_id][cat_name] = []
        img_id_2_anns[image_id][cat_name].append(ann)

    return img_id_2_anns


def convert_lvis(dataset_name, ann_json_path, coco_img_root, out_json_dir, out_meta_path,
                 area_ratio_ths_big=0.008):
    meta_dict = {
        "dataset_name": dataset_name,
        "ann_json_path": ann_json_path,
        "coco_img_root": coco_img_root,
        "num_images": 0,
        "datalist": [],
    }
    os.makedirs(out_json_dir, exist_ok=True)

    print(f"Loading annotation file: {ann_json_path}")
    d = json.load(open(ann_json_path))

    # LVIS categories use synonyms instead of a single name field.
    # We use the first synonym as the canonical category name.
    cat_dict = {}          # category_id -> canonical_name (synonyms[0])
    cat_synonyms_dict = {} # canonical_name -> [all synonyms]
    cat_clean_synonyms = {}
    canonical_counts = defaultdict(int)
    for category in d["categories"]:
        cat_id = category["id"]
        synonyms = category["synonyms"]
        assert len(synonyms) > 0, f"No synonyms found for category {cat_id}"
        clean_synonyms = get_clean_synonyms(synonyms)
        if len(clean_synonyms) == 0:
            raise ValueError(f"All synonyms exceed 30 chars for category {cat_id}: {synonyms}")
        cat_clean_synonyms[cat_id] = clean_synonyms
        canonical_counts[clean_synonyms[0]] += 1

    for category in d["categories"]:
        cat_id = category["id"]
        synonyms = category["synonyms"]
        clean_synonyms = cat_clean_synonyms[cat_id]
        if canonical_counts[clean_synonyms[0]] > 1:
            clean_synonyms = get_clean_synonyms(synonyms, keep_parenthetical_text=True)
            if len(clean_synonyms) == 0:
                raise ValueError(f"All synonyms exceed 30 chars for category {cat_id}: {synonyms}")
        canonical = clean_synonyms[0]
        if canonical in cat_synonyms_dict:
            raise ValueError(f"Duplicate canonical category name after cleaning: {canonical}")
        cat_dict[cat_id] = canonical
        cat_synonyms_dict[canonical] = clean_synonyms

    annotations = d.get("annotations", [])
    img_id_2_anns = get_img_id_2_anns(annotations, cat_dict)

    images = d["images"]
    num_images = 0
    num_invalid_images = 0
    for image in images:
        out_json_path = convert_anns_for_image(
            image, coco_img_root, img_id_2_anns, cat_synonyms_dict, out_json_dir,
            area_ratio_ths_big=area_ratio_ths_big
        )
        if out_json_path:
            num_images += 1
            meta_dict["datalist"].append(out_json_path)
        else:
            num_invalid_images += 1

    meta_dict["num_images"] = num_images
    print(f"\nTotal images with valid annotations: {num_images}")
    print(f"Total invalid images: {num_invalid_images}")
    print(f"Total images: {len(images)}")

    with open(out_meta_path, "w") as meta_json_f:
        json.dump(meta_dict, meta_json_f, indent=4)
    print(f"Meta file saved to: {out_meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert LVIS dataset annotations to per-image JSON")
    parser.add_argument(
        "--data_root", type=str, default="/home/hyx/datasets/lvis",
        help="Root directory of LVIS annotations (should contain lvis_v1_*.json)"
    )
    parser.add_argument(
        "--coco_img_root", type=str, default=None,
        help=(
            "Root directory of COCO images, containing train2017/ and val2017/ "
            "subdirectories. Image paths are resolved per-sample from coco_url. "
            "Defaults to <data_root>/images if not set."
        ),
    )
    parser.add_argument(
        "--split_name", type=str, default="train",
        choices=["train", "val"],
        help="Dataset split: train or val",
    )
    parser.add_argument(
        "--area_ratio_ths_big", type=float, default=0.008,
        help="Category area ratio below this threshold is marked as is_small=True",
    )

    args = parser.parse_args()

    split_name = args.split_name
    data_root = args.data_root

    # Annotation file: lvis_v1_train.json / lvis_v1_val.json
    ann_json_path = os.path.join(data_root, f"lvis_v1_{split_name}.json")

    coco_img_root = args.coco_img_root or os.path.join(data_root, "images")

    dataset_name = f"lvis_{split_name}2017"
    out_json_dir = os.path.join(data_root, f"{split_name}2017_json_")
    out_meta_path = os.path.join(os.path.dirname(__file__), f"lvis_{split_name}2017_meta.json")

    convert_lvis(
        dataset_name, ann_json_path, coco_img_root, out_json_dir, out_meta_path,
        area_ratio_ths_big=args.area_ratio_ths_big,
    )


if __name__ == "__main__":
    main()
