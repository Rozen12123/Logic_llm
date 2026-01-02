#!/usr/bin/env python3
"""
根据推理结果筛选原始数据：
1) flag == "success" 且 answer 与 predicted_answer 一致 -> 视为可用训练样本
2) 其余样本 -> 归类为需要进一步处理的样本
全部输出到 outputs/data_err_train 目录
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

LOGIC_INFERENCE_DIR = Path("./outputs/logic_inference")
DATA_ROOT = Path("./data")
OUTPUT_DIR = Path("./outputs/data_err_train")
# ============================================================================
# 默认配置：将常用的推理结果文件路径写在这里，直接运行脚本即可批处理
# ============================================================================
DEFAULT_INFERENCE_FILES: List[str] = [
    # 示例：

    #"./outputs/logic_inference/ProntoQA_dev_glm-4.6_backup-random.json",
    #"./outputs/logic_inference/ProofWriter_dev_glm-4.6_backup-random.json",
    #"./outputs/logic_inference/LogicalDeduction_dev_glm-4.6_backup-random.json",
    #"./outputs/logic_inference/FOLIO_dev_glm-4.6_backup-random.json",
    #"./outputs/logic_inference/AR-LSAT_dev_glm-4.6_backup-random.json"
    "./outputs/logic_inference/FOLIO_train_deepseek-v3.2_backup-random.json",
    "./outputs/logic_inference/AR-LSAT_train_deepseek-v3.2_backup-random.json"
    


]
# 若某些推理文件无法通过文件名自动推断数据集，可在此处提供映射
# Key 为推理文件名（不含路径），Value 为对应数据集 JSON 路径
DEFAULT_DATASET_FILES: Dict[str, str] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按推理结果筛选原始数据样本。")
    parser.add_argument(
        "--inference_file",
        "-i",
        nargs="*",
        help="要处理的推理结果 JSON 文件，可一次传多个；若缺省则使用默认配置。",
    )
    parser.add_argument(
        "--dataset_file",
        "-d",
        help="原始数据集 JSON 文件（默认根据推理文件名自动推断 data/<dataset>/<split>.json）",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(OUTPUT_DIR),
        help=f"输出目录，默认 {OUTPUT_DIR}",
    )
    return parser.parse_args()


def infer_dataset_path(inference_path: Path) -> Tuple[str, str, Path]:
    stem = inference_path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"无法从文件名 {inference_path.name} 推断 <dataset>_<split> 格式。")
    dataset = parts[0]
    split = parts[1]
    dataset_path = DATA_ROOT / dataset / f"{split}.json"
    return dataset, split, dataset_path


def load_json_list(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 内容不是列表。")
    return data


def build_id_map(entries: List[dict]) -> Dict[str, dict]:
    return {entry["id"]: entry for entry in entries if "id" in entry}


def split_ids(inference_entries: List[dict]) -> Tuple[List[str], List[str]]:
    success_ids: List[str] = []
    other_ids: List[str] = []
    for sample in inference_entries:
        sid = sample.get("id")
        if sid is None:
            continue
        flag_ok = sample.get("flag") == "success"
        answers_match = sample.get("answer") == sample.get("predicted_answer") and sample.get("answer") is not None
        if flag_ok and answers_match:
            success_ids.append(sid)
        else:
            other_ids.append(sid)
    return success_ids, other_ids


def select_dataset_entries(id_map: Dict[str, dict], ids: List[str], dataset_name: str = "") -> List[dict]:
    selected: List[dict] = []
    missing: List[str] = []
    for sid in ids:
        entry = id_map.get(sid)
        if entry is None:
            missing.append(sid)
        else:
            selected.append(entry)
    if missing:
        # 尝试在其他split中查找
        found_in_other_splits: Dict[str, List[str]] = {}
        if dataset_name:
            for split in ['dev', 'test']:
                other_file = DATA_ROOT / dataset_name / f"{split}.json"
                if other_file.exists():
                    try:
                        with other_file.open("r", encoding="utf-8") as f:
                            other_data = json.load(f)
                        other_ids = {item["id"]: item for item in other_data if "id" in item}
                        for mid in missing:
                            if mid in other_ids:
                                if split not in found_in_other_splits:
                                    found_in_other_splits[split] = []
                                found_in_other_splits[split].append(mid)
                    except Exception:
                        pass
        
        still_missing = [mid for mid in missing if not any(mid in found_in_other_splits.get(s, []) for s in found_in_other_splits)]
        
        if found_in_other_splits:
            print(f"[Warning] 数据集中缺少 {len(missing)} 个样本，其中：")
            for split, found_ids in found_in_other_splits.items():
                print(f"  - {len(found_ids)} 个样本存在于 {split}.json 中（示例：{found_ids[0]}）")
            if still_missing:
                print(f"  - {len(still_missing)} 个样本在所有split中都不存在（示例：{still_missing[0]}）")
        else:
            print(f"[Warning] 数据集中缺少 {len(missing)} 个样本：{missing[:5]}{'...' if len(missing) > 5 else ''}")
    return selected


def write_output(base_name: str, output_dir: Path, suffix: str, entries: List[dict]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{base_name}{suffix}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    return output_path


def resolve_inference_files(arg_files: Optional[List[str]]) -> List[Path]:
    if arg_files:
        files = arg_files
    elif DEFAULT_INFERENCE_FILES:
        files = DEFAULT_INFERENCE_FILES
    else:
        raise SystemExit("请通过 --inference_file 传入文件，或在 DEFAULT_INFERENCE_FILES 中配置。")
    paths = [Path(path).expanduser().resolve() for path in files]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"以下推理结果文件不存在：{missing}")
    return paths


def resolve_dataset_path(
    inference_path: Path, dataset_override: Optional[str]
) -> Path:
    if dataset_override:
        dataset_path = Path(dataset_override).expanduser().resolve()
        return dataset_path

    override = DEFAULT_DATASET_FILES.get(inference_path.name)
    if override:
        return Path(override).expanduser().resolve()

    _, _, dataset_path = infer_dataset_path(inference_path)
    return dataset_path


def process_single_file(
    inference_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> None:
    if not dataset_path.exists():
        raise SystemExit(f"找不到数据集文件：{dataset_path}")

    inference_entries = load_json_list(inference_path)
    dataset_entries = load_json_list(dataset_path)
    dataset_map = build_id_map(dataset_entries)

    # 从 dataset_path 推断数据集名称（例如 data/ProofWriter/train.json -> ProofWriter）
    dataset_name = dataset_path.parent.name if dataset_path.parent.name != "data" else ""

    success_ids, other_ids = split_ids(inference_entries)
    success_entries = select_dataset_entries(dataset_map, success_ids, dataset_name=dataset_name)
    other_entries = select_dataset_entries(dataset_map, other_ids, dataset_name=dataset_name)

    base_name = inference_path.stem
    success_path = write_output(base_name, output_dir, "__success_dataset", success_entries)
    others_path = write_output(base_name, output_dir, "__others_dataset", other_entries)

    print(f"[完成] {inference_path.name}: 成功样本 {len(success_entries)} -> {success_path}")
    print(f"[完成] {inference_path.name}: 其他样本 {len(other_entries)} -> {others_path}")


def main() -> None:
    args = parse_args()
    inference_files = resolve_inference_files(args.inference_file)
    output_dir = Path(args.output_dir).expanduser().resolve()

    for inference_path in inference_files:
        dataset_path = resolve_dataset_path(inference_path, args.dataset_file)
        process_single_file(inference_path, dataset_path, output_dir)


if __name__ == "__main__":
    main()

