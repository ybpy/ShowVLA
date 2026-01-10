from calendar import c
import os
import json
import csv
import numpy as np
import argparse
from collections import defaultdict
from PIL import Image

def set_seed(seed=0):
    np.random.seed(seed)

def convert_anns_for_image(img_name, img_dir, ann_dir, class_dict, out_json_dir, 
                            area_ratio_ths=0.0039):

    img_path = os.path.join(img_dir, img_name)
    print(f"\n{img_path}")

    ann_name = img_name.replace('.jpg', '.png')
    ann_path = os.path.join(ann_dir, ann_name)

    segm = Image.open(ann_path)
    segm = np.array(segm)
    height, width = segm.shape

    class_2_area = {class_id: 0 for class_id in class_dict.keys()}
    for i in range(height):
        for j in range(width):
            v = segm[i, j]
            if v == 0:
                continue
            if v in class_2_area:
                class_2_area[v] += 1
    
    list_categories = []
    for class_id, area in class_2_area.items():
        if area == 0:
            continue
        class_name = class_dict[class_id]

        area_ratio = area / (height*width)
        print(f"Area Ratio of {class_name}: {area_ratio}")
        if area_ratio < area_ratio_ths:
            print(f"Too small (<{area_ratio_ths}). Discard!!!")
            continue
        list_categories.append((class_id, class_name))

    if len(list_categories) == 0:
        return None
    
    data_dict = {
        "img_path": img_path,
        "height": height,
        "width": width,
        "segm_path": ann_path,
        "list_categories": list_categories,
    }
    img_id = os.path.splitext(img_name)[0]
    out_json_path = os.path.join(out_json_dir, f"{img_id}.json")
    with open(out_json_path, 'w') as json_f:
        json.dump(data_dict, json_f, indent=4)
    
    return out_json_path


def convert_ade20k(data_root, split_name):
    dataset_name = f"ade20k_{split_name}"
    out_json_dir = f"{data_root}/{split_name}_json_"
    out_meta_path = f"./ade20k_{split_name}_meta.json"
    meta_dict = {
        "dataset_name": dataset_name,
        "num_images": 0,
        "datalist": []
    }
    os.makedirs(out_json_dir)

    img_dir = os.path.join(data_root, "images", split_name)
    ann_dir = os.path.join(data_root, "annotations", split_name)

    # 读取class_info
    list_invalid = []
    class_dict = dict()
    with open(os.path.join(data_root, "object150_info.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id = int(row['Idx'])
            num_in_train = int(row['Train'])
            num_in_val = int(row['Val'])
            stuff = int(row['Stuff'])
            if (num_in_train + num_in_val) < 100:
                list_invalid.append(class_id)
            if stuff:
                list_invalid.append(class_id)
    with open(os.path.join(data_root, "objectInfo150.txt")) as f:
        for line in f:
            if line.startswith("Idx"):
                continue
            line = line.strip()
            class_id = int(line.split('\t')[0])
            if class_id in list_invalid:
                continue

            class_name = line.split('\t')[-1].split(',')[0]
            if any(k in class_name for k in ["blind", "lamp", "light", "screen", "table", "chair", "counter", "stair", "step", "earth", "land", "stool", "pool", "pot", "case", "sea", "water", "tree", "palm", "glass", "buffet", "plant", "shelf", "desk", "base", "column", "bench", "bar", "apparel", "van", "truck", "boat", "ship", "trade name", "monitor"]):
                continue
            class_dict[class_id] = class_name
            print(f"[{len(class_dict)}] Class {class_id:03d}: {class_name}")
    
    # 遍历所有图像
    num_images = 0
    for img_name in sorted(os.listdir(img_dir)):
        assert img_name.endswith('.jpg')
        out_json_path = convert_anns_for_image(img_name, img_dir, ann_dir, class_dict, out_json_dir)
        if out_json_path:
            num_images += 1
            meta_dict["datalist"].append(out_json_path)
    
    meta_dict["num_images"] = num_images
    with open(out_meta_path, 'w') as meta_json_f:
        json.dump(meta_dict, meta_json_f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Convert ADE20K dataset annotations')
    parser.add_argument('--data_root', type=str, default='/home/hyx/datasets/ADEChallengeData2016',
                        help='Root directory of ADE20K dataset')
    parser.add_argument('--split_name', type=str, default='validation',
                        choices=['training', 'validation'],
                        help='Split name')
    
    
    args = parser.parse_args()
    
    split_name = args.split_name
    data_root = args.data_root

    # 设置随机种子
    set_seed(0)

    convert_ade20k(data_root, split_name)


if __name__ == "__main__":
    main()
