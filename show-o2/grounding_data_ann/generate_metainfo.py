import os
import json
import argparse
from pathlib import Path

def generate_metainfo(dataset_name, data_dirs, language_instruction_key, observation_key, output_path):
    """
    生成数据集的元数据文件。
    
    Args:
        dataset_name (str): 数据集名称。
        data_dirs (list): 包含 .hdf5 文件的目录列表。
        language_instruction_key (str): 语言指令的键名。
        observation_key (list): 观测值的键名列表。
        output_path (str): 输出的 JSON 文件路径。
    """
    datalist = []
    
    # 遍历指定的目录，查找所有 .hdf5 文件
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Warning: Directory {data_dir} does not exist. Skipping.")
            continue
            
        # 递归查找所有 .hdf5 文件，并按字母顺序排序
        hdf5_files = sorted(list(data_path.rglob("*.hdf5")))
        for hdf5_file in hdf5_files:
            # 使用绝对路径
            datalist.append(str(hdf5_file.absolute()))
    
    # 构建元数据字典
    metainfo = {
        "dataset_name": dataset_name,
        "data_dirs": data_dirs,
        "language_instruction_key": language_instruction_key,
        "observation_key": observation_key,
        "num_ep": len(datalist),
        "datalist": datalist
    }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    # 将结果写入 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metainfo, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully generated {output_path} with {len(datalist)} episodes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metainfo JSON for HDF5 datasets.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True, help="Directories containing .hdf5 files")
    parser.add_argument("--language_instruction_key", type=str, default="language_instruction", help="Key for language instructions")
    parser.add_argument("--observation_key", type=str, nargs='+', default=["rgb_comb"], help="Keys for observations")
    parser.add_argument("--output", type=str, default="metainfo.json", help="Output JSON file path")

    args = parser.parse_args()

    generate_metainfo(
        args.dataset_name,
        args.data_dirs,
        args.language_instruction_key,
        args.observation_key,
        args.output
    )
