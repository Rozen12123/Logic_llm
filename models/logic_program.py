# generate facts and rules based on the problem description

import json
import os
import sys
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import OpenAIModel, ZhipuAIModel
import argparse

# 添加项目根目录到路径，以便导入config_loader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config_loader import load_api_key, load_api_provider
except ImportError:
    def load_api_key(provider="zhipuai"):
        return None
    def load_api_provider():
        return "zhipuai"

class LogicProgramGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.api_provider = getattr(args, 'api_provider', 'zhipuai')
        self.max_retries = getattr(args, 'max_retries', 3)

        # 根据API提供商选择使用OpenAI或智谱AI
        if self.api_provider == 'zhipuai':
            self.api = ZhipuAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        else:
            self.api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        
        # 为了向后兼容，保留 openai_api 属性
        self.openai_api = self.api
        
        self.prompt_creator = {'FOLIO': self.prompt_folio,
                               'ProntoQA': self.prompt_prontoqa,
                               'ProofWriter': self.prompt_proofwriter,
                               'LogicalDeduction': self.prompt_logicaldeduction, 
                               'AR-LSAT': self.prompt_arlsat}
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}.txt'
        if self.dataset_name == 'AR-LSAT' and self.model_name == 'gpt-4':
            prompt_file = f'./models/prompts/{self.dataset_name}-long.txt'
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def prompt_folio(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def prompt_arlsat(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt
    
    def prompt_prontoqa(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_proofwriter(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_logicaldeduction(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt

    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def validate_logic_program(self, program: str) -> bool:
        if not isinstance(program, str):
            return False
        program = program.strip()
        if not program:
            return False
        required_sections = ['Predicates:', 'Facts:', 'Rules:', 'Query:']
        return all(section in program for section in required_sections)

    def generate_program_with_retry(self, prompt: str, sample_id: str):
        last_output = ''
        for attempt in range(1, self.max_retries + 1):
            try:
                output = self.openai_api.generate(prompt)
            except Exception as e:
                output = ''
                print(f'Error generating logic program for {sample_id} (attempt {attempt}): {e}')
            last_output = output if isinstance(output, str) else ''
            if self.validate_logic_program(last_output):
                return last_output.strip()
            print(f'Invalid logic program for {sample_id} on attempt {attempt}, retrying...')
        print(f'Failed to obtain valid logic program for {sample_id} after {self.max_retries} attempts.')
        return last_output.strip()

    def logic_program_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        for example in tqdm(raw_dataset):
            # create prompt
            try:
                full_prompt = self.prompt_creator[self.dataset_name](example)
                program = self.generate_program_with_retry(full_prompt, example['id'])
                programs = [program]

                # create output
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        'options': example['options'],
                        'raw_logic_programs': programs}
                outputs.append(output)
            except:
                print('Error in generating logic programs for example: ', example['id'])

        # save outputs        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    '''
    Updated version of logic_program_generation; speed up the generation process by batching
    '''
    def batch_logic_program_generation(self, batch_size = 10, max_concurrent = None):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        
        if max_concurrent:
            print(f"使用并发数: {max_concurrent}")

        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [self.prompt_creator[self.dataset_name](example) for example in chunk]
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts, max_concurrent=max_concurrent)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    program = output if isinstance(output, str) else ''
                    if not self.validate_logic_program(program):
                        program = self.generate_program_with_retry(
                            self.prompt_creator[self.dataset_name](sample),
                            sample['id']
                        )
                    programs = [program]
                    output = {'id': sample['id'], 
                            'context': sample['context'],
                            'question': sample['question'], 
                            'answer': sample['answer'],
                            'options': sample['options'],
                            'raw_logic_programs': programs}
                    outputs.append(output)
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    try:
                        program = self.generate_program_with_retry(full_prompt, sample['id'])
                        programs = [program]
                        output = {'id': sample['id'], 
                                'context': sample['context'],
                                'question': sample['question'], 
                                'answer': sample['answer'],
                                'options': sample['options'],
                                'raw_logic_programs': programs}
                        outputs.append(output)
                    except:
                        print('Error in generating logic programs for example: ', sample['id'])

        # remove examples with duplicate ids from the result
        outputs = list({output['id']: output for output in outputs}.values())
        print(f"Generated {len(outputs)} examples.")
        
        # save outputs
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--api_provider', type=str, choices=['openai', 'zhipuai'], 
                       help='API提供商: openai 或 zhipuai (如果未提供，将从config.py或环境变量读取)')
    parser.add_argument('--api_key', type=str, help='API Key (如果未提供，将从config.py或环境变量读取)')
    parser.add_argument('--model_name', type=str, default='glm-4-flash-250414', 
                       help='模型名称 (OpenAI: text-davinci-003, gpt-4等; 智谱AI: glm-4-flash-250414, glm-4等)')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_concurrent', type=int, default=20,
                       help='最大并发数，用于控制同时进行的API请求数量。默认为20')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='批处理大小，每个批次处理的样本数量。默认为10')
    parser.add_argument('--max_retries', type=int, default=3,
                       help='单个样本在输出缺失关键段落时的最大重试次数')
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

if __name__ == '__main__':
    args = parse_args()
    logic_program_generator = LogicProgramGenerator(args)
    logic_program_generator.batch_logic_program_generation(
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent
    )