import argparse
import json
import os


def merge_metainfo(input_files, output_file, dataset_name="Bridge"):
    merged_data = {
        "dataset_name": dataset_name,
        "language_instruction_key": None,
        "observation_key": None,
        "num_ep": 0,
        "datalist": [],
    }

    for i, file_path in enumerate(input_files):
        with open(file_path, "r") as f:
            data = json.load(f)

        if i == 0:
            merged_data["language_instruction_key"] = data.get("language_instruction_key")
            merged_data["observation_key"] = data.get("observation_key")
            if data.get("dataset_name"):
                merged_data["dataset_name"] = data["dataset_name"]
        else:
            if merged_data["language_instruction_key"] != data.get("language_instruction_key"):
                print(f"Warning: language_instruction_key mismatch in {file_path}")
            if merged_data["observation_key"] != data.get("observation_key"):
                print(f"Warning: observation_key mismatch in {file_path}")

        merged_data["num_ep"] += data.get("num_ep", 0)
        merged_data["datalist"].extend(data.get("datalist", []))

    if merged_data["num_ep"] != len(merged_data["datalist"]):
        print(
            f"Warning: num_ep ({merged_data['num_ep']}) does not match "
            f"datalist length ({len(merged_data['datalist'])})"
        )
        merged_data["num_ep"] = len(merged_data["datalist"])

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(
        f"Successfully merged {len(input_files)} files into {output_file} "
        f"(num_ep={merged_data['num_ep']})"
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_inputs = [
        os.path.join(script_dir, "train", "Bridge_train_metainfo.json"),
        os.path.join(script_dir, "val", "Bridge_val_metainfo.json"),
    ]
    default_output = os.path.join(script_dir, "Bridge_all_metainfo.json")

    parser = argparse.ArgumentParser(description="Merge Bridge train/val metainfo JSON files.")
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=default_inputs,
        help="Input metainfo JSON files (default: train + val).",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=default_output,
        help="Output metainfo JSON path (default: Bridge_all_metainfo.json).",
    )
    args = parser.parse_args()

    merge_metainfo(args.input, args.output)
