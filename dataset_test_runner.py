#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用测试脚本：支持交互式或命令行方式选择数据集，并按需运行
1. 逻辑程序生成
2. 符号推理
3. 结果评估

用法示例：
python scripts/dataset_test_runner.py --dataset ProntoQA --stage logic_program logic_inference evaluation \
    --model_name glm-4-flash-250414 --api_provider zhipuai --split dev
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config_loader import load_api_key, load_api_provider
from models.logic_program import LogicProgramGenerator
from models.logic_inference import LogicInferenceEngine
from models.evaluation import full_evaluation

#AVAILABLE_DATASETS = ["ProntoQA", "ProofWriter", "FOLIO", "LogicalDeduction", "AR-LSAT"]
#STAGES = ["logic_program", "logic_inference", "evaluation"]
AVAILABLE_DATASETS = ["ProntoQA"]
STAGES = ["logic_program", "logic_inference", "evaluation"]


def prompt_choice(options: List[str], prompt: str) -> str:
    print(prompt)
    for idx, value in enumerate(options, start=1):
        print(f"[{idx}] {value}")
    while True:
        choice = input("请输入编号：").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        print("输入无效，请重新选择。")


def prompt_stage_selection() -> List[str]:
    print("请选择要执行的步骤，可输入多个编号，用逗号分隔：")
    for idx, stage in enumerate(STAGES, start=1):
        print(f"[{idx}] {stage}")
    while True:
        raw = input("请输入：").strip()
        selected = []
        try:
            for item in raw.split(","):
                item = item.strip()
                if not item:
                    continue
                idx = int(item) - 1
                if 0 <= idx < len(STAGES):
                    selected.append(STAGES[idx])
            selected = list(dict.fromkeys(selected))
        except ValueError:
            selected = []
        if selected:
            return selected
        print("输入无效，请重新选择。")


def ensure_api_settings(args: argparse.Namespace) -> None:
    if not args.api_provider:
        args.api_provider = load_api_provider()
        print(f"[INFO] 使用配置中的 API 提供商：{args.api_provider}")
    if not args.api_key:
        args.api_key = load_api_key(args.api_provider)
        if args.api_key:
            print(f"[INFO] 已从配置或环境变量读取 {args.api_provider} 的 API Key")
        else:
            print("[WARN] 未提供 API Key，若运行逻辑程序生成将会失败。")


def build_logic_programs(args: argparse.Namespace, dataset: str) -> None:
    lp_args = SimpleNamespace(
        data_path=args.data_path,
        dataset_name=dataset,
        split=args.split,
        save_path=args.logic_program_save,
        api_provider=args.api_provider,
        api_key=args.api_key,
        model_name=args.model_name,
        stop_words=args.stop_words,
        max_new_tokens=args.max_new_tokens,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
    )
    generator = LogicProgramGenerator(lp_args)
    generator.batch_logic_program_generation(
        batch_size=lp_args.batch_size,
        max_concurrent=lp_args.max_concurrent,
    )


def run_logic_inference(args: argparse.Namespace, dataset: str) -> None:
    expected_program_file = (
        Path(args.logic_program_save)
        / f"{dataset}_{args.split}_{args.model_name}.json"
    )
    if not expected_program_file.exists():
        raise FileNotFoundError(
            f"未找到逻辑程序文件：{expected_program_file}，请先运行 logic_program 步骤。"
        )

    infer_args = SimpleNamespace(
        dataset_name=dataset,
        split=args.split,
        save_path=args.logic_inference_save,
        backup_strategy=args.backup_strategy,
        backup_LLM_result_path=args.backup_llm_result_path,
        model_name=args.model_name,
        timeout=args.solver_timeout,
    )
    engine = LogicInferenceEngine(infer_args)
    engine.inference_on_dataset()


def run_evaluation(args: argparse.Namespace, dataset: str) -> None:
    result_file = (
        Path(args.logic_inference_save)
        / f"{dataset}_{args.split}_{args.model_name}_backup-{args.backup_strategy}.json"
    )
    if not result_file.exists():
        raise FileNotFoundError(
            f"未找到推理结果文件：{result_file}，请先运行 logic_inference 步骤。"
        )
    print(f"[INFO] 正在评估 {result_file}")
    full_evaluation(str(result_file))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一的数据集测试脚本，可交互选择或通过参数指定。"
    )
    parser.add_argument("--dataset", choices=AVAILABLE_DATASETS, help="待测试数据集")
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=STAGES,
        help="要运行的步骤，可指定多个，例如 --stage logic_program logic_inference",
    )
    parser.add_argument("--split", default="dev", help="数据集划分，默认 dev")
    parser.add_argument("--model_name", default="glm-4-flash-250414", help="模型名称")
    parser.add_argument("--data_path", default="./data", help="数据目录")
    parser.add_argument(
        "--logic_program_save",
        default="./outputs/logic_programs",
        help="逻辑程序输出目录",
    )
    parser.add_argument(
        "--logic_inference_save",
        default="./outputs/logic_inference",
        help="推理结果输出目录",
    )
    parser.add_argument(
        "--backup_strategy",
        default="random",
        choices=["random", "LLM"],
        help="推理备份策略",
    )
    parser.add_argument(
        "--backup_llm_result_path",
        default="./baselines/results",
        help="当使用 LLM 备份时需要提供的 CoT 结果路径",
    )
    parser.add_argument("--api_provider", choices=["openai", "zhipuai"], help="API 提供商")
    parser.add_argument("--api_key", help="LLM API Key")
    parser.add_argument("--stop_words", default="------", help="生成停止标志")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="生成 token 上限")
    parser.add_argument("--max_concurrent", type=int, default=20, help="并发请求上限")
    parser.add_argument("--batch_size", type=int, default=10, help="批处理大小")
    parser.add_argument("--solver_timeout", type=int, default=60, help="符号求解器超时时间")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="强制进入交互模式，即使已经提供参数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interactive:
        if not args.dataset:
            args.dataset = prompt_choice(AVAILABLE_DATASETS, "请选择要测试的数据集：")
        if not args.stage:
            args.stage = prompt_stage_selection()
    else:
        if not args.dataset:
            args.dataset = AVAILABLE_DATASETS[0]
            print(f"[INFO] 未指定数据集，默认使用：{args.dataset}")
        if not args.stage:
            args.stage = STAGES.copy()
            print(f"[INFO] 未指定 stage，默认执行：{', '.join(args.stage)}")

    ensure_api_settings(args)
    print(f"[INFO] 将在 {args.dataset} ({args.split}) 上运行：{', '.join(args.stage)}")

    if "logic_program" in args.stage:
        print("[STEP] 开始生成逻辑程序...")
        
        build_logic_programs(args, args.dataset)

    if "logic_inference" in args.stage:
        print("[STEP] 开始执行符号推理...")
        run_logic_inference(args, args.dataset)

    if "evaluation" in args.stage:
        print("[STEP] 开始评估推理结果...")
        run_evaluation(args, args.dataset)

    print("[DONE] 全部步骤已完成。")


if __name__ == "__main__":
    main()

