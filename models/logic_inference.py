import json
import os
import sys
import shutil
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from symbolic_solvers.csp_solver.csp_solver import CSP_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
import argparse
import random
from backup_answer_generation import Backup_Answer_Generator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(CURRENT_DIR, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

from logic_program_cleaner import clean_logic_program

class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.backup_strategy = args.backup_strategy

        self.dataset = self.load_logic_programs()
        program_executor_map = {'FOLIO': FOL_Prover9_Program, 
                                'ProntoQA': Pyke_Program, 
                                'ProofWriter': Pyke_Program,
                                'LogicalDeduction': CSP_Program,
                                'AR-LSAT': LSAT_Z3_Program}
        self.program_executor = program_executor_map[self.dataset_name]
        self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.args.backup_LLM_result_path)

    def load_logic_programs(self):
        with open(os.path.join('./outputs/logic_programs', f'{self.dataset_name}_{self.split}_{self.model_name}.json'), 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def save_results(self, outputs):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}_backup-{self.backup_strategy}.json'), 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, id, logic_program):
        program = self.program_executor(logic_program, self.dataset_name)
        # cannot parse the program
        if program.flag == False:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'parsing error', ''
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'execution error', error_message
        # successfully executed
        answer = program.answer_mapping(answer)
        return answer, 'success', ''

    def inference_on_dataset(self):
        outputs = []
        error_count = 0
        
        for example in tqdm(self.dataset):
            # execute the logic program
            normalized_program = self.preprocess_logic_program(example['raw_logic_programs'][0])
            answer, flag, error_message = self.safe_execute_program(example['id'], normalized_program)
            if not flag == 'success':
                error_count += 1

            # create output
            output = {'id': example['id'], 
                    'context': example['context'],
                    'question': example['question'], 
                    'answer': example['answer'],
                    'flag': flag,
                    'predicted_answer': answer}
            # 如果有错误信息，添加到输出中
            if error_message and error_message.strip():
                output['error_message'] = error_message
            outputs.append(output)
        
        print(f"Error count: {error_count}")
        self.save_results(outputs)
        self.cleanup()

    def cleanup(self):
        candidates = [
            os.path.abspath(os.path.join(CURRENT_DIR, '..', 'compiled_krb')),
            os.path.abspath(os.path.join(CURRENT_DIR, 'compiled_krb'))
        ]
        removed = False
        for compiled_krb_dir in candidates:
            if os.path.exists(compiled_krb_dir):
                if not removed:
                    print('removing compiled_krb')
                    removed = True
                shutil.rmtree(compiled_krb_dir, ignore_errors=True)

    def preprocess_logic_program(self, raw_program: str) -> str:
        if not isinstance(raw_program, str):
            return raw_program
        return clean_logic_program(raw_program.strip(), self.dataset_name)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_inference')
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='../baselines/results')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--timeout', type=int, default=60)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()