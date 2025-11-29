#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 LLM 推理结果与求解器推理结果：
  - LLM 预测正确时保留原结果
  - LLM 预测错误且求解器预测正确时采用求解器版本
最终合并后的 JSON 写入 ./outputs/Compare 目录
"""

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

DEFAULT_LLM_FILE = './outputs/logic_inference/self-refine-1_ProntoQA_dev_glm-4-flash-250414_llm-symbolic.json'
DEFAULT_SOLVER_FILE = './outputs/logic_inference/ProntoQA_dev_glm-4-flash-250414_backup-random.json'
OUTPUT_DIR = './outputs/Compare'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='比较 LLM 与求解器答案，并选择正确的结果输出。',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--llm_file', type=str, default=DEFAULT_LLM_FILE,
                        help='LLM 生成的推理结果 JSON 文件路径')
    parser.add_argument('--solver_file', type=str, default=DEFAULT_SOLVER_FILE,
                        help='求解器生成的推理结果 JSON 文件路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='合并结果输出目录')
    parser.add_argument('--force', action='store_true',
                        help='若输出文件已存在则覆盖')
    return parser.parse_args()


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"File {path} does not contain a list of entries.")
    return data


def normalize_choice(value: Optional[str]) -> Optional[str]:
    if not value or not isinstance(value, str):
        return None
    value = value.strip().upper()
    if len(value) == 1 and value in 'ABCDE':
        return value
    if value.startswith('(') and value.endswith(')') and len(value) == 3:
        candidate = value[1]
        if candidate in 'ABCDE':
            return candidate
    return None


def is_prediction_correct(entry: Dict[str, Any]) -> bool:
    answer = normalize_choice(entry.get('answer'))
    prediction = normalize_choice(entry.get('predicted_answer'))
    return bool(answer and prediction and answer == prediction)


def build_output_filename(llm_file: str, solver_file: str) -> str:
    llm_name = os.path.splitext(os.path.basename(llm_file))[0]
    solver_name = os.path.splitext(os.path.basename(solver_file))[0]
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f'{llm_name}__vs__{solver_name}_{timestamp}.json'


def merge_results(
    llm_entries: List[Dict[str, Any]],
    solver_entries: List[Dict[str, Any]],
    solver_file: str
) -> List[Dict[str, Any]]:
    solver_lookup = {entry.get('id'): entry for entry in solver_entries if entry.get('id') is not None}
    merged: List[Dict[str, Any]] = []

    for llm_entry in llm_entries:
        entry_id = llm_entry.get('id')
        base_record = deepcopy(llm_entry)
        final_source = 'llm'
        final_note = 'llm_prediction_correct'

        if not is_prediction_correct(llm_entry):
            solver_entry = solver_lookup.get(entry_id)
            if solver_entry and is_prediction_correct(solver_entry):
                base_record = deepcopy(solver_entry)
                final_source = 'solver'
                final_note = 'llm_wrong_solver_correct'
                if 'source_logic_program_file' not in base_record:
                    base_record['source_logic_program_file'] = os.path.basename(solver_file)
            else:
                final_source = 'llm_incorrect'
                final_note = 'no_correct_prediction_found'

        base_record['compare_source'] = final_source
        base_record['compare_note'] = final_note
        merged.append(base_record)

    return merged


def save_results(results: List[Dict[str, Any]], output_dir: str, filename: str, force: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    if os.path.exists(output_path) and not force:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --force to overwrite.")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return output_path


def main():
    args = parse_args()
    llm_entries = load_json(args.llm_file)
    solver_entries = load_json(args.solver_file)

    merged_results = merge_results(llm_entries, solver_entries, args.solver_file)
    filename = build_output_filename(args.llm_file, args.solver_file)
    output_path = save_results(merged_results, args.output_dir, filename, args.force)

    total = len(merged_results)
    llm_correct = sum(1 for entry in merged_results if entry.get('compare_source') == 'llm')
    solver_used = sum(1 for entry in merged_results if entry.get('compare_source') == 'solver')

    print("Comparison finished.")
    print(f"  Total entries            : {total}")
    print(f"  LLM predictions kept     : {llm_correct}")
    print(f"  Solver predictions reused: {solver_used}")
    print(f"  Output file              : {output_path}")


if __name__ == '__main__':
    main()

