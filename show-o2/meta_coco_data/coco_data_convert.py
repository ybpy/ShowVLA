import os
import json
from collections import defaultdict



def convert_anns_for_image(image, img_dir, img_id_2_anns, out_json_dir,
                            num_instances_ths=8, area_ratio_ths=0.0043):
    file_name = image["file_name"]
    assert image["coco_url"].endswith(file_name)
    height, width = image["height"], image["width"]
    img_id = image["id"]

    img_path = os.path.join(img_dir, file_name)
    print(f"[img_id: {img_id}] {img_path}")

    cat_2_instances = img_id_2_anns[img_id]
    if len(cat_2_instances) == 0:
        return None
    
    cat_2_instances_filtered = dict()
    for cat, instances in cat_2_instances.items():
        area = 0
        for ann in instances:
            area += ann['area']
        area_ratio = area / (height*width)
        print(f"# Instances of {cat}: {len(instances)}")
        if len(instances) >= num_instances_ths:
            print(f"Too many instances (>={num_instances_ths}). Discard!!!")
            continue
        print(f"Area Ratio: {area_ratio}")
        if area_ratio < area_ratio_ths:
            print(f"Area Ratio is too small (<{area_ratio_ths})). Discard!!!")
            continue
        cat_2_instances_filtered[cat] = instances

    if len(cat_2_instances_filtered) == 0:
        return None
    
    data_dict = {
        "img_path": img_path,
        "anns": cat_2_instances_filtered,
    }
    out_json_path = os.path.join(out_json_dir, f"{img_id}.json")
    with open(out_json_path, 'w') as json_f:
        json.dump(data_dict, json_f, indent=4)
    
    return out_json_path


def get_img_id_2_anns(annotations, cat_dict):
    img_id_2_anns = defaultdict(dict)
    for ann in annotations:
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        cat = cat_dict[category_id]

        if cat not in img_id_2_anns[image_id]:
            img_id_2_anns[image_id][cat] = []
        img_id_2_anns[image_id][cat].append(ann)

    return img_id_2_anns

def convert_coco(dataset_name, ann_json_path, img_dir, out_json_dir, out_meta_path):
    meta_dict = {
        "dataset_name": dataset_name,
        "ann_json_path": ann_json_path,
        "img_dir": img_dir,
        "num_images": 0,
        "datalist": []
    }
    os.makedirs(out_json_dir)

    d = json.load(open(ann_json_path))

    categories = d['categories']
    cat_dict = dict()
    for category in categories:
        cat = category["name"]
        cat_id = category["id"]
        cat_dict[cat_id] = cat
    
    annotations = d["annotations"]
    img_id_2_anns = get_img_id_2_anns(annotations, cat_dict)

    images = d["images"]
    num_images = 0
    for image in images:
        out_json_path = convert_anns_for_image(image, img_dir, img_id_2_anns, out_json_dir)
        if out_json_path:
            num_images += 1
            meta_dict["datalist"].append(out_json_path)
    
    meta_dict["num_images"] = num_images
    with open(out_meta_path, 'w') as meta_json_f:
        json.dump(meta_dict, meta_json_f, indent=4)

split_name = "train"

dataset_name = f"coco_{split_name}2017"
ann_json_path = f"/home/hyx/datasets/coco/annotations/instances_{split_name}2017.json"
img_dir = f"/home/hyx/datasets/coco/{split_name}2017"
out_json_dir = f"/home/hyx/datasets/coco/{split_name}2017_json"
out_meta_path = f"./coco_{split_name}2017_meta.json"
convert_coco(dataset_name, ann_json_path, img_dir, out_json_dir, out_meta_path)
