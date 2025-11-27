# input: logic program file
# output: logic program file after one round of self-refinement

import json
import os
from tqdm import tqdm
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from symbolic_solvers.csp_solver.csp_solver import CSP_Program
import argparse
import random
import sys
from backup_answer_generation import Backup_Answer_Generator
# 兼容相对导入和绝对导入
try:
    from .utils import OpenAIModel, ZhipuAIModel
except ImportError:
    from models.utils import OpenAIModel, ZhipuAIModel

# 添加项目根目录到路径，以便导入config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config_loader import load_api_key, load_api_provider
except ImportError:
    def load_api_key(provider="zhipuai"):
        return None
    def load_api_provider():
        return "zhipuai"

class SelfRefinementEngine:
    def __init__(self, args, current_round):
        self.args = args
        self.split = args.split
        self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.backup_strategy = args.backup_strategy
        self.api_provider = getattr(args, 'api_provider', 'zhipuai')
        # 预留并发配置点（目前 single_round_self_refinement 仍为顺序执行）
        self.max_concurrent = getattr(args, 'max_concurrent', 1)
        
        # 根据API提供商选择使用OpenAI或智谱AI
        # 注意：self-refinement 默认使用 gpt-4，但如果是智谱AI，使用指定的模型
        refinement_model = 'gpt-4' if self.api_provider == 'openai' else args.model_name
        if self.api_provider == 'zhipuai':
            self.api = ZhipuAIModel(args.api_key, refinement_model, args.stop_words, args.max_new_tokens)
        else:
            self.api = OpenAIModel(args.api_key, refinement_model, args.stop_words, args.max_new_tokens)
        
        # 为了向后兼容，保留 openai_api 属性
        self.openai_api = self.api
        
        self.current_round = current_round

        self.logic_programs = self.load_logic_programs()
        # self.reasoning_results = self.load_inference_results()

        program_executor_map = {'AR-LSAT': LSAT_Z3_Program,
                                'FOLIO': FOL_Prover9_Program,
                                'ProntoQA': Pyke_Program,
                                'ProofWriter': Pyke_Program,
                                'LogicalDeduction': CSP_Program}
        self.program_executor = program_executor_map[self.dataset_name]
        self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.args.backup_LLM_result_path)

    def load_logic_programs(self):
        prefix = ""
        if self.current_round > 1:
            prefix = f'self-refine-{self.current_round-1}_'
        with open(os.path.join('./outputs/logic_programs', f'{prefix}{self.dataset_name}_{self.split}_{self.model_name}.json')) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def load_prompt(self, program, error_message, context=None, question=None):
        program = program.strip()
        error_message = error_message.strip()
        prompt_file = f'./models/prompts/self-correct-{self.dataset_name}.txt'
        
        # 如果不存在特定数据集的 prompt，尝试使用通用格式
        if not os.path.exists(prompt_file):
            # 对于没有特定 prompt 的数据集，使用通用格式
            full_prompt = f"""Task: Given the wrong logic program and the error message, output the correct logic program.

>>> Initial Program:
{program}
>>> Error Message:
{error_message}
>>> Corrected Program:
"""
            return full_prompt
        
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()
        full_prompt = prompt_template.replace('[[PROGRAM]]', program).replace('[[ERROR MESSAGE]]', error_message)
        return full_prompt

    def safe_execute_program(self, id, logic_program, debug = False):
        program = self.program_executor(logic_program, self.dataset_name)
        # cannot parse the program
        if program.flag == False:
            answer = self.backup_generator.get_backup_answer(id)
            # 尝试获取更详细的错误信息
            error_message = 'Parsing Error: Failed to parse the logic program'
            try:
                # 某些 solver 可能会在初始化时设置错误信息
                if hasattr(program, 'error_message') and program.error_message:
                    error_message = f'Parsing Error: {program.error_message}'
            except:
                pass
            return answer, 'parsing error', error_message
        # execuate the program
        answer, error_message = program.execute_program(debug=debug)

        # 确保错误信息是字符串，避免后续调用 .strip() 或写入文件时报错
        if not isinstance(error_message, str):
            try:
                error_message = str(error_message)
            except Exception:
                error_message = 'Execution Error: Unknown error object returned from execute_program'
        # not executable
        if answer is None:
            answer = self.backup_generator.get_backup_answer(id)
            
            # 确保错误消息不为空
            if not error_message or (isinstance(error_message, str) and error_message.strip() == ''):
                error_message = 'Execution Error: Program execution failed with no output'

            ## output debug info
            if debug == True:
                if not os.path.exists('./debug'):
                    os.makedirs('./debug')
                with open(f'./debug/{id}.py', 'w') as f:
                    if hasattr(program, 'standard_code'):
                        f.write(program.standard_code)
                with open(f'./debug/{id}.program.txt', 'w') as f:
                    f.write(logic_program)
                    f.write('\n')
                    f.write(error_message)

            return answer, 'execution error', error_message
        # successfully executed
        answer = program.answer_mapping(answer)
        return answer, 'success', ''
    
    def single_round_self_refinement(self):
        outputs = []
        fixed_count = 0
        for example in tqdm(self.logic_programs):
            # 如果之前轮次已经成功（refinement_status == 'success'），则直接跳过后续自我修正
            # 只把该样本原样写入本轮输出，避免重复调用求解器和大模型
            prev_status = example.get('refinement_status')
            if prev_status == 'success':
                outputs.append(example)
                continue

            logic_program = example['raw_logic_programs'][0].strip()
            answer, status, error_message = self.safe_execute_program(example['id'], logic_program)

            if status == 'execution error':
                if not error_message == 'No Output': # this is not execution error, but parsing error
                    # perform self-correction based on the error message
                    full_prompt = self.load_prompt(logic_program, error_message, 
                                                  example.get('context'), example.get('question'))
                    revised_program = self.openai_api.generate(full_prompt).strip()
                    
                    # 尝试重新执行修复后的程序
                    revised_answer, revised_status, revised_error = self.safe_execute_program(
                        example['id'], revised_program)
                    
                    # 如果修复成功，使用修复后的程序；否则保留原程序
                    if revised_status == 'success':
                        programs = [revised_program]
                        refinement_status = 'success'
                        fixed_count += 1
                        print(f"Fixed example {example['id']} after {status}")
                    else:
                        # 修复失败，保留原程序，标记为仍需后续轮次尝试
                        programs = [logic_program]
                        refinement_status = 'failed'
                        print(f"Failed to fix example {example['id']}, keeping original program")
                    
                    output = {'id': example['id'], 
                            'context': example['context'],
                            'question': example['question'], 
                            'answer': example['answer'],
                            'options': example.get('options', []),
                            'raw_logic_programs': programs,
                            'refinement_status': refinement_status}
                    outputs.append(output)
                else:
                    outputs.append(example)
            elif status == 'parsing error':
                # perform self-correction based on the error message
                error_msg = error_message if error_message else 'Parsing Error'
                full_prompt = self.load_prompt(logic_program, error_msg,
                                              example.get('context'), example.get('question'))
                revised_program = self.openai_api.generate(full_prompt).strip()
                
                # 尝试重新执行修复后的程序
                revised_answer, revised_status, revised_error = self.safe_execute_program(
                    example['id'], revised_program)
                
                # 如果修复成功，使用修复后的程序；否则保留原程序
                if revised_status == 'success':
                    programs = [revised_program]
                    refinement_status = 'success'
                    fixed_count += 1
                    print(f"Fixed example {example['id']} after {status}")
                else:
                    # 修复失败，保留原程序，标记为仍需后续轮次尝试
                    programs = [logic_program]
                    refinement_status = 'failed'
                    print(f"Failed to fix example {example['id']}, keeping original program")
                
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        'options': example.get('options', []),
                        'raw_logic_programs': programs,
                        'refinement_status': refinement_status}
                outputs.append(output)
            else:
                # 本轮无需自我修正，但将其标记为 success，便于后续轮次直接跳过
                example.setdefault('refinement_status', 'success')
                outputs.append(example)
        
        print(f"Fixed {fixed_count} examples in this round.")

        # save results
        if not os.path.exists('./outputs/logic_programs'):
            os.makedirs('./outputs/logic_programs')

        # save outputs
        save_path = f'./outputs/logic_programs/self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.model_name}.json'
        with open(save_path, 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maximum_rounds', type=int, default=3)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='../baselines/results')
    parser.add_argument('--api_provider', type=str, choices=['openai', 'zhipuai'], 
                       help='API提供商: openai 或 zhipuai (如果未提供，将从config.py或环境变量读取)')
    parser.add_argument('--model_name', type=str, default='glm-4-flash-250414', 
                       help='模型名称 (OpenAI: text-davinci-003, gpt-4等; 智谱AI: glm-4-flash-250414, glm-4等)')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--api_key', type=str, help='API Key (如果未提供，将从config.py或环境变量读取)')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    
    # 如果命令行没有提供api_provider，尝试从配置文件或环境变量读取
    if not args.api_provider:
        args.api_provider = load_api_provider()
        print(f"使用API提供商: {args.api_provider}")
    
    # 如果命令行没有提供api_key，尝试从配置文件或环境变量读取
    if not args.api_key:
        args.api_key = load_api_key(args.api_provider)
        if args.api_key:
            print(f"已从配置文件或环境变量读取{args.api_provider.upper()} API Key")
        else:
            print(f"警告: 未找到{args.api_provider.upper()} API Key，请通过以下方式之一设置:")
            if args.api_provider == 'zhipuai':
                print("  1. 在config.py中设置ZHIPUAI_API_KEY")
                print("  2. 设置环境变量ZHIPUAI_API_KEY")
            else:
                print("  1. 在config.py中设置OPENAI_API_KEY")
                print("  2. 设置环境变量OPENAI_API_KEY")
            print("  3. 使用命令行参数 --api_key")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    for round in range(1, args.maximum_rounds+1):
        print(f"Round {round} self-refinement")
        engine = SelfRefinementEngine(args, round)
        engine.single_round_self_refinement()