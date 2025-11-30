#!/usr/bin/env python3
"""
将逻辑推理结果拆分成两份：
1) flag 为 "success" 且 answer 与 predicted_answer 一致的样本
2) 其余所有样本
并同时把对应的逻辑程序（来自 outputs/logic_programs）同步分类输出。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple, Union, Any, Optional, Dict

# ============================================================================
# 默认配置：在这里填写常用的文件/筛选关键词，直接运行脚本即可处理
# ============================================================================
DEFAULT_INPUT_DIR = Path("./outputs/logic_inference")
DEFAULT_OUTPUT_DIR = Path("./outputs/data")
LOGIC_PROGRAM_DIR = Path("./outputs/logic_programs")
LOGIC_PROGRAM_SUCCESS_SUBDIR = "logic_programs_success"
LOGIC_PROGRAM_OTHERS_SUBDIR = "logic_programs_others"
# 如果你想固定处理某几个文件，直接把路径填进下面列表即可（留空表示使用目录扫描）
DEFAULT_FILE_PATHS: List[str] = [

    "./outputs/logic_inference/ProofWriter_train_glm-4.6_backup-random.json",
    "./outputs/logic_inference/FOLIO_train_glm-4.6_backup-random.json",
    "./outputs/logic_inference/AR-LSAT_train_glm-4.6_backup-random.json"


]
# 如果想按关键词自动筛文件（当 DEFAULT_FILE_PATHS 为空时生效）
DEFAULT_SELECT_KEYWORDS: List[str] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="把满足条件的推理样本单独导出，便于后续分析。"
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="*",
        help="需要处理的 JSON 文件路径，默认会遍历 outputs/logic_inference/*.json。",
    )
    parser.add_argument(
        "--select",
        "-s",
        nargs="*",
        help="当未指定 --input 时，用文件名关键字筛选（任意关键字命中即可）。",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出目录，默认保存在 {DEFAULT_OUTPUT_DIR}。",
    )
    parser.add_argument(
        "--include-non-success",
        action="store_true",
        help="即使 flag 不是 success，只要答案一致也收入 success_match 文件。",
    )
    parser.add_argument(
        "--logic-program-file",
        help="显式指定逻辑程序 JSON 文件（默认依据推理文件名自动推断）。",
    )
    return parser.parse_args()


def collect_input_files(
    inputs: Union[None, List[str]], selects: Optional[List[str]]
) -> List[Path]:
    if inputs:
        files = [Path(path).expanduser().resolve() for path in inputs]
    elif DEFAULT_FILE_PATHS:
        files = [Path(path).expanduser().resolve() for path in DEFAULT_FILE_PATHS]
    else:
        candidates = sorted(p.resolve() for p in DEFAULT_INPUT_DIR.glob("*.json"))
        select_pool = selects or DEFAULT_SELECT_KEYWORDS
        if select_pool:
            lowered = [s.lower() for s in select_pool]
            files = [p for p in candidates if any(k in p.name.lower() for k in lowered)]
        else:
            files = candidates
    return [p for p in files if p.exists()]


def load_entries(file_path: Path) -> List[dict]:
    with file_path.open("r", encoding="utf-8") as f:
        data: Any = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            return data["results"]
    raise ValueError(f"{file_path} 的 JSON 结构不受支持")


def split_entries(entries: Iterable[dict], require_success: bool = True) -> Tuple[List[dict], List[dict]]:
    success_matches: List[dict] = []
    others: List[dict] = []
    for sample in entries:
        flag_ok = sample.get("flag") == "success"
        answers_match = sample.get("answer") is not None and sample.get("answer") == sample.get("predicted_answer")
        if answers_match and (flag_ok or not require_success):
            success_matches.append(sample)
        else:
            others.append(sample)
    return success_matches, others


def write_json(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_logic_program_file(inference_file: Path, override: Optional[str]) -> Path:
    if override:
        lp_path = Path(override).expanduser().resolve()
        if not lp_path.exists():
            raise SystemExit(f"指定的逻辑程序文件不存在：{lp_path}")
        return lp_path

    stem = inference_file.stem
    candidates = [stem]
    if "_backup" in stem:
        candidates.append(stem.split("_backup", 1)[0])
    if stem.endswith("_llm-symbolic"):
        candidates.append(stem[: -len("_llm-symbolic")])
    if "__" in stem:
        candidates.append(stem.split("__", 1)[0])

    seen = set()
    ordered_candidates = []
    for cand in candidates:
        if cand not in seen:
            ordered_candidates.append(cand)
            seen.add(cand)

    for cand in ordered_candidates:
        lp_path = LOGIC_PROGRAM_DIR / f"{cand}.json"
        if lp_path.exists():
            return lp_path

    raise SystemExit(
        f"无法为 {inference_file.name} 找到逻辑程序文件，请使用 --logic-program-file 指定。"
    )


def select_entries_by_ids(id_map: Dict[str, dict], ids: List[str], label: str) -> List[dict]:
    selected: List[dict] = []
    missing: List[str] = []
    for sid in ids:
        entry = id_map.get(sid)
        if entry is None:
            missing.append(sid)
        else:
            selected.append(entry)
    if missing:
        preview = ", ".join(missing[:5])
        more = "..." if len(missing) > 5 else ""
        print(f"[Warning] {label} 缺少 {len(missing)} 个样本：{preview}{more}")
    return selected


def process_file(
    file_path: Path,
    output_dir: Path,
    include_non_success: bool,
    logic_program_override: Optional[str],
) -> Tuple[Path, Path, int, int]:
    entries = load_entries(file_path)
    success_matches, others = split_entries(entries, require_success=not include_non_success)
    stem = file_path.stem

    success_ids = [item.get("id") for item in success_matches if item.get("id")]
    others_ids = [item.get("id") for item in others if item.get("id")]

    success_path = output_dir / f"{stem}__success_match.json"
    others_path = output_dir / f"{stem}__others.json"
    write_json(success_path, success_matches)
    write_json(others_path, others)

    logic_program_path = resolve_logic_program_file(file_path, logic_program_override)
    logic_entries = load_entries(logic_program_path)
    logic_id_map = {entry.get("id"): entry for entry in logic_entries if entry.get("id")}

    logic_success_entries = select_entries_by_ids(
        logic_id_map, success_ids, f"{logic_program_path.name} / success"
    )
    logic_others_entries = select_entries_by_ids(
        logic_id_map, others_ids, f"{logic_program_path.name} / others"
    )

    logic_success_dir = output_dir / LOGIC_PROGRAM_SUCCESS_SUBDIR
    logic_others_dir = output_dir / LOGIC_PROGRAM_OTHERS_SUBDIR
    logic_success_path = logic_success_dir / f"{stem}__logic_programs_success.json"
    logic_others_path = logic_others_dir / f"{stem}__logic_programs_others.json"
    write_json(logic_success_path, logic_success_entries)
    write_json(logic_others_path, logic_others_entries)

    return success_path, others_path, len(success_matches), len(others)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    input_files = collect_input_files(args.input, args.select)
    if not input_files:
        raise SystemExit(f"在 {DEFAULT_INPUT_DIR} 下或 --input 参数中未找到可用的 JSON 文件。")

    for file_path in input_files:
        success_path, others_path, success_count, others_count = process_file(
            file_path,
            output_dir,
            include_non_success=args.include_non_success,
            logic_program_override=args.logic_program_file,
        )
        print(
            f"[{file_path.name}] 成功&匹配: {success_count} -> {success_path.name}, "
            f"其余: {others_count} -> {others_path.name}"
        )


if __name__ == "__main__":
    main()

