# generate facts and rules based on the problem description
# 支持保存思考过程的版本

import json
import os
import sys
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
# 兼容相对导入和绝对导入
try:
    from .utils import OpenAIModel, ZhipuAIModel
except ImportError:
    from models.utils import OpenAIModel, ZhipuAIModel
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

class LogicProgramGeneratorWithThinking:
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
            # 获取 base_url（用于 iflow 等需要自定义 base_url 的 API）
            base_url = getattr(args, 'base_url', None)
            self.api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens, base_url=base_url)
        
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
        
        # 不同数据集使用不同的格式
        if self.dataset_name == 'FOLIO':
            # FOLIO 使用 Predicates:, Premises:, Conclusion:
            required_sections = ['Predicates:', 'Premises:', 'Conclusion:']
        elif self.dataset_name == 'AR-LSAT':
            # AR-LSAT 使用 # Declarations / # Constraints / # Options 三段格式
            required_sections = [
                ('# Declarations', '### Declarations'),
                ('# Constraints', '### Constraints'),
                ('# Options', '### Options')
            ]
        else:
            # ProntoQA, ProofWriter, LogicalDeduction 使用 Predicates:, Facts:, Rules:, Query:
            required_sections = ['Predicates:', 'Facts:', 'Rules:', 'Query:']
        
        def has_section(program_text, section):
            if isinstance(section, (list, tuple)):
                return any(marker in program_text for marker in section)
            return section in program_text
        
        return all(has_section(program, section) for section in required_sections)

    def _extract_thinking_from_response(self, response) -> Optional[str]:
        """从API响应中提取思考过程（reasoning_content/reasoning/thinking等）"""
        if not response:
            return None
        
        try:
            choices = getattr(response, 'choices', None)
            if choices is None and isinstance(response, dict):
                choices = response.get('choices')
            if not choices:
                return None
            
            choice = choices[0]
            message = getattr(choice, 'message', choice)
            
            # 定义可能包含思考过程的字段名（按优先级排序）
            thinking_fields = [
                'reasoning_content',  # 智谱AI
                'reasoning',          # 某些模型
                'thinking',           # 某些模型
                'chain_of_thought',   # CoT格式
                'thought',            # 某些模型
            ]
            
            # 首先尝试从 message 对象中获取
            thinking_content = None
            for field in thinking_fields:
                thinking_content = getattr(message, field, None)
                if thinking_content is None and isinstance(message, dict):
                    thinking_content = message.get(field)
                if thinking_content:
                    break
            
            # 如果 message 中没有，尝试从 choice 中获取（但要排除 finish_reason）
            if not thinking_content:
                for field in thinking_fields:
                    thinking_content = getattr(choice, field, None)
                    if thinking_content is None and isinstance(choice, dict):
                        thinking_content = choice.get(field)
                    # 排除 finish_reason（它可能包含 "stop" 等值）
                    if thinking_content and field != 'finish_reason':
                        # 确保不是简单的状态字符串
                        if isinstance(thinking_content, str) and thinking_content.lower() not in ['stop', 'length', 'content_filter', 'function_call', 'tool_calls']:
                            break
                        elif not isinstance(thinking_content, str):
                            break
                    else:
                        thinking_content = None
            
            # 如果 choice 中也没有，尝试从 response 中获取
            if not thinking_content:
                for field in thinking_fields:
                    thinking_content = getattr(response, field, None)
                    if thinking_content is None and isinstance(response, dict):
                        thinking_content = response.get(field)
                    if thinking_content:
                        break
            
            if thinking_content:
                # 处理不同格式的思考内容
                if isinstance(thinking_content, str):
                    # 直接返回字符串，确保完整（不截断）
                    result = thinking_content.strip()
                    return result if result else None
                elif isinstance(thinking_content, list):
                    # 如果是列表，尝试提取所有文本内容（确保完整）
                    parts = []
                    for item in thinking_content:
                        if isinstance(item, dict):
                            # 尝试多种可能的字段
                            text = (item.get('text') or 
                                   item.get('content') or 
                                   item.get('value') or
                                   item.get('reasoning_text') or
                                   "")
                            if text:
                                parts.append(str(text))
                        elif isinstance(item, str):
                            parts.append(item)
                        else:
                            parts.append(str(item))
                    result = ''.join(parts).strip()
                    return result if result else None
                elif isinstance(thinking_content, dict):
                    # 如果是字典，尝试提取文本字段（确保完整）
                    text_fields = ['text', 'content', 'thinking', 'reasoning', 'value', 'reasoning_text']
                    for field in text_fields:
                        if field in thinking_content:
                            value = thinking_content[field]
                            if value:
                                if isinstance(value, str):
                                    return value.strip()
                                elif isinstance(value, list):
                                    # 如果是列表，递归提取
                                    parts = []
                                    for item in value:
                                        if isinstance(item, dict):
                                            parts.append(str(item.get('text') or item.get('content') or item))
                                        else:
                                            parts.append(str(item))
                                    result = ''.join(parts).strip()
                                    return result if result else None
                                else:
                                    return str(value).strip()
                    # 如果没有找到文本字段，将整个字典转为字符串
                    return str(thinking_content).strip()
                else:
                    return str(thinking_content).strip() if thinking_content else None
            
            # OpenAI 兼容接口可能将思考过程放在 content 的列表结构中
            # 检查 content 是否是列表格式（OpenAI 的 reasoning 格式使用 content 列表）
            content = getattr(message, 'content', None)
            if content is None and isinstance(message, dict):
                content = message.get('content')
            
            if isinstance(content, list):
                # 如果 content 是列表，查找类型为 reasoning/thinking 的部分
                thinking_parts = []
                reasoning_text_parts = []
                
                for item in content:
                    if isinstance(item, dict):
                        item_type = str(item.get('type', '')).lower()
                        # OpenAI 兼容接口可能使用 'type': 'reasoning' 或 'type': 'reasoning_text'
                        if item_type in ['reasoning', 'thinking', 'reasoning_content', 'chain_of_thought', 'reasoning_text']:
                            # 尝试多种可能的字段名
                            text = (item.get('text') or 
                                   item.get('content') or 
                                   item.get('value') or
                                   item.get('reasoning_text') or
                                   "")
                            if text:
                                if item_type == 'reasoning_text':
                                    reasoning_text_parts.append(str(text))
                                else:
                                    thinking_parts.append(str(text))
                        # 某些格式可能在 text 类型的项中包含思考内容
                        elif item_type == 'text':
                            text = item.get('text') or item.get('content') or ""
                            # 如果文本很长且包含推理关键词，可能是思考过程
                            if text and len(text) > 50:
                                # 检查是否包含推理相关的关键词
                                reasoning_keywords = ['think', 'reason', 'analyze', 'consider', 'step', 'conclusion']
                                if any(keyword in text.lower() for keyword in reasoning_keywords):
                                    thinking_parts.append(str(text))
                
                # 优先返回 reasoning_text 类型的内容
                if reasoning_text_parts:
                    return '\n'.join(reasoning_text_parts).strip()
                if thinking_parts:
                    return '\n'.join(thinking_parts).strip()
            
            # 如果 content 是字符串，检查是否包含思考过程（某些模型可能将思考过程和答案混在一起）
            elif isinstance(content, str) and len(content) > 100:
                # 检查是否有明显的思考过程标记
                thinking_markers = [
                    'Let me think',
                    'First,',
                    'To solve this',
                    'I need to',
                    'Let me analyze',
                    '思考',
                    '分析',
                    '首先',
                ]
                # 如果内容开头包含思考标记，可能是思考过程
                content_lower = content.lower()
                for marker in thinking_markers:
                    if marker.lower() in content_lower[:200]:  # 检查前200个字符
                        # 尝试提取思考部分（可能在某个分隔符之前）
                        # 这里我们返回整个内容，让调用者决定如何处理
                        # 但通常逻辑程序会在思考过程之后
                        return content.strip()
            
            # 检查是否有 reasoning 相关的字段在 response 的其他位置
            # OpenAI 兼容接口可能将 reasoning 放在 response 的顶层或 choices 中
            if hasattr(response, 'reasoning'):
                reasoning = response.reasoning
                if reasoning:
                    # reasoning 可能是对象、列表或字符串
                    if isinstance(reasoning, (list, dict)):
                        import json
                        try:
                            return json.dumps(reasoning, default=str, ensure_ascii=False).strip()
                        except:
                            return str(reasoning).strip()
                    return str(reasoning).strip()
            elif isinstance(response, dict) and 'reasoning' in response:
                reasoning = response['reasoning']
                if reasoning:
                    if isinstance(reasoning, (list, dict)):
                        import json
                        try:
                            return json.dumps(reasoning, default=str, ensure_ascii=False).strip()
                        except:
                            return str(reasoning).strip()
                    return str(reasoning).strip()
            
            # 检查 choice 中是否有 reasoning 字段
            if hasattr(choice, 'reasoning'):
                reasoning = choice.reasoning
                if reasoning:
                    if isinstance(reasoning, (list, dict)):
                        import json
                        try:
                            return json.dumps(reasoning, default=str, ensure_ascii=False).strip()
                        except:
                            return str(reasoning).strip()
                    return str(reasoning).strip()
            elif isinstance(choice, dict) and 'reasoning' in choice:
                reasoning = choice.get('reasoning')
                if reasoning:
                    if isinstance(reasoning, (list, dict)):
                        import json
                        try:
                            return json.dumps(reasoning, default=str, ensure_ascii=False).strip()
                        except:
                            return str(reasoning).strip()
                    return str(reasoning).strip()
            
            # 检查 choice 中是否有其他字段（排除 finish_reason）
            if hasattr(choice, '__dict__'):
                for attr_name in choice.__dict__.keys():
                    # 明确排除 finish_reason
                    if attr_name.lower() == 'finish_reason':
                        continue
                    if 'reason' in attr_name.lower() or 'think' in attr_name.lower():
                        attr_value = getattr(choice, attr_name, None)
                        if attr_value:
                            # 确保不是简单的状态字符串
                            if isinstance(attr_value, str) and attr_value.lower() not in ['stop', 'length', 'content_filter']:
                                if len(attr_value) > 20:  # 只返回较长的字符串
                                    return attr_value.strip()
                            elif not isinstance(attr_value, str):
                                return str(attr_value).strip()
            
            # 检查 message 中是否有其他字段（包括 annotations）
            if hasattr(message, '__dict__'):
                # 首先检查 annotations（某些 API 可能将思考过程放在这里）
                if hasattr(message, 'annotations'):
                    annotations = message.annotations
                    if annotations:
                        # annotations 可能是列表或字典
                        if isinstance(annotations, list):
                            for ann in annotations:
                                if isinstance(ann, dict):
                                    # 查找包含思考过程的注解
                                    for key in ['content', 'text', 'reasoning', 'thinking']:
                                        if key in ann:
                                            val = ann[key]
                                            if val:
                                                return str(val).strip()
                                elif isinstance(ann, str):
                                    return ann.strip()
                        elif isinstance(annotations, dict):
                            for key in ['content', 'text', 'reasoning', 'thinking', 'reasoning_content']:
                                if key in annotations:
                                    val = annotations[key]
                                    if val:
                                        return str(val).strip()
                
                # 检查其他可能包含思考过程的字段（排除 finish_reason 等不相关字段）
                exclude_fields = ['finish_reason', 'refusal', 'role', 'function_call', 'tool_calls', 'audio']
                for attr_name in message.__dict__.keys():
                    if attr_name.lower() in [e.lower() for e in exclude_fields]:
                        continue
                    if 'reason' in attr_name.lower() or 'think' in attr_name.lower() or 'annot' in attr_name.lower():
                        attr_value = getattr(message, attr_name, None)
                        if attr_value and attr_name != 'content':  # 排除 content 本身
                            # 跳过简单字符串值（可能是 finish_reason 等）
                            if isinstance(attr_value, str) and len(attr_value) < 20 and attr_value.lower() in ['stop', 'length', 'content_filter']:
                                continue
                            # 尝试从复杂对象中提取
                            if isinstance(attr_value, (list, dict)):
                                # 递归查找
                                try:
                                    import json
                                    json_str = json.dumps(attr_value, default=str)
                                    if json_str and len(json_str) > 10:
                                        return json_str.strip()
                                except:
                                    return str(attr_value).strip()
                            elif isinstance(attr_value, str) and len(attr_value) > 20:  # 只返回较长的字符串（可能是思考过程）
                                return attr_value.strip()
                            elif not isinstance(attr_value, str):
                                return str(attr_value).strip()
            elif isinstance(message, dict):
                # 检查 annotations
                if 'annotations' in message:
                    annotations = message['annotations']
                    if annotations:
                        if isinstance(annotations, list):
                            for ann in annotations:
                                if isinstance(ann, dict):
                                    for key in ['content', 'text', 'reasoning', 'thinking']:
                                        if key in ann:
                                            val = ann[key]
                                            if val:
                                                return str(val).strip()
                        elif isinstance(annotations, dict):
                            for key in ['content', 'text', 'reasoning', 'thinking', 'reasoning_content']:
                                if key in annotations:
                                    val = annotations[key]
                                    if val:
                                        return str(val).strip()
                
                # 检查其他字段（排除 finish_reason 等不相关字段）
                exclude_fields = ['finish_reason', 'refusal', 'role', 'function_call', 'tool_calls', 'audio']
                for key in message.keys():
                    if key.lower() in [e.lower() for e in exclude_fields]:
                        continue
                    if ('reason' in key.lower() or 'think' in key.lower() or 'annot' in key.lower()) and key != 'content':
                        value = message[key]
                        if value:
                            # 跳过简单字符串值（可能是 finish_reason 等）
                            if isinstance(value, str) and len(value) < 20 and value.lower() in ['stop', 'length', 'content_filter']:
                                continue
                            if isinstance(value, str) and len(value) > 20:  # 只返回较长的字符串
                                return value.strip()
                            elif not isinstance(value, str):
                                return str(value).strip()
            
        except Exception as e:
            # 静默处理错误，不打印（避免干扰正常输出）
            pass
        
        return None

    def generate_program_with_retry(self, prompt: str, sample_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        生成逻辑程序，返回 (program, thinking) 元组
        """
        last_output = ''
        last_thinking = None
        last_response = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                # 尝试获取完整响应以提取思考过程
                response = None
                output = ''
                thinking = None
                
                # 对于智谱AI
                if self.api_provider == 'zhipuai' and hasattr(self.api, 'client'):
                    # 直接调用API获取完整响应
                    response = self.api.client.chat.completions.create(
                        model=self.api.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=self.api.max_new_tokens,
                        stop=self.api.stop_words if self.api.stop_words else None,
                        **self.api._thinking_kwargs()
                    )
                    last_response = response
                    # 提取内容
                    output = self.api._extract_message_content(response)
                    # 提取思考过程
                    thinking = self._extract_thinking_from_response(response)
                    last_thinking = thinking
                    
                    # 调试：检查 reasoning_content 是否完整（仅第一个样本）
                    if sample_id and ('_1' in sample_id or sample_id.endswith('1')):
                        try:
                            if hasattr(response, 'choices') and response.choices:
                                msg = response.choices[0].message
                                reasoning_raw = getattr(msg, 'reasoning_content', None)
                                if reasoning_raw:
                                    import json
                                    if isinstance(reasoning_raw, str):
                                        print(f"\n[调试] reasoning_content 类型: str, 长度: {len(reasoning_raw)}")
                                        print(f"[调试] reasoning_content 最后100字符: {reasoning_raw[-100:]}")
                                    elif isinstance(reasoning_raw, list):
                                        print(f"\n[调试] reasoning_content 类型: list, 长度: {len(reasoning_raw)}")
                                        print(f"[调试] reasoning_content 结构: {json.dumps(reasoning_raw[:2], default=str, ensure_ascii=False)[:500]}")
                                    elif isinstance(reasoning_raw, dict):
                                        print(f"\n[调试] reasoning_content 类型: dict, 键: {list(reasoning_raw.keys())}")
                                        print(f"[调试] reasoning_content 内容预览: {json.dumps(reasoning_raw, default=str, ensure_ascii=False)[:500]}")
                                    
                                    if thinking:
                                        print(f"[调试] 提取的思考过程长度: {len(thinking)}")
                                        print(f"[调试] 提取的思考过程最后50字符: {thinking[-50:]}")
                                    else:
                                        print(f"[调试] 未提取到思考过程")
                        except Exception as e:
                            print(f"[调试] 检查 reasoning_content 时出错: {e}")
                
                # 对于OpenAI/iflow等其他API，也尝试获取完整响应
                elif hasattr(self.openai_api, 'client') or hasattr(self.openai_api, 'model_name'):
                    # 尝试直接调用API获取完整响应
                    try:
                        from models.utils import chat_completions_with_backoff, OPENAI_VERSION
                        
                        if OPENAI_VERSION >= 2 and hasattr(self.openai_api, 'client') and self.openai_api.client:
                            # 新版本 OpenAI API
                            # 尝试添加参数以获取思考过程（如果 API 支持）
                            api_kwargs = {
                                'model': self.openai_api.model_name,
                                'messages': [{"role": "user", "content": prompt}],
                                'temperature': 0.0,
                                'max_tokens': self.openai_api.max_new_tokens,
                                'top_p': 1.0,
                            }
                            if self.openai_api.stop_words:
                                api_kwargs['stop'] = self.openai_api.stop_words
                            
                            # 某些 OpenAI 兼容接口可能需要 include 参数来返回 reasoning
                            # 尝试添加 include 参数（如果 API 支持）
                            try:
                                # 先尝试不包含 reasoning 的调用
                                response = self.openai_api.client.chat.completions.create(**api_kwargs)
                            except Exception as e:
                                # 如果失败，尝试添加 include 参数
                                try:
                                    api_kwargs['include'] = ['reasoning']
                                    response = self.openai_api.client.chat.completions.create(**api_kwargs)
                                except:
                                    # 如果还是失败，使用原始参数
                                    api_kwargs.pop('include', None)
                                    response = self.openai_api.client.chat.completions.create(**api_kwargs)
                        else:
                            # 使用工具函数
                            response = chat_completions_with_backoff(
                                self.openai_api.client if OPENAI_VERSION >= 2 and hasattr(self.openai_api, 'client') else None,
                                model=self.openai_api.model_name,
                                messages=[{"role": "user", "content": prompt}],
                                max_tokens=self.openai_api.max_new_tokens,
                                temperature=0.0,
                                top_p=1.0,
                                stop=self.openai_api.stop_words if self.openai_api.stop_words else None
                            )
                        
                        last_response = response
                        
                        # 调试：打印响应结构（仅第一个样本）
                        if sample_id and ('_1' in sample_id or sample_id.endswith('1')):
                            import json
                            try:
                                if hasattr(response, '__dict__'):
                                    response_dict = {k: str(v)[:200] for k, v in response.__dict__.items()}
                                elif isinstance(response, dict):
                                    response_dict = {k: str(v)[:200] if not isinstance(v, (dict, list)) else json.dumps(v)[:200] for k, v in list(response.items())[:10]}
                                else:
                                    response_dict = str(response)[:500]
                                print(f"\n[调试] API响应结构示例 (sample_id={sample_id}):")
                                print(json.dumps(response_dict, indent=2, ensure_ascii=False, default=str)[:1000])
                                
                                # 尝试打印 message 的详细结构
                                if OPENAI_VERSION >= 2:
                                    if isinstance(response, dict):
                                        msg = response.get('choices', [{}])[0].get('message', {})
                                    else:
                                        msg = response.choices[0].message if hasattr(response, 'choices') else {}
                                else:
                                    msg = response.get('choices', [{}])[0].get('message', {})
                                
                                if isinstance(msg, dict):
                                    print(f"\n[调试] Message字段: {list(msg.keys())}")
                                elif hasattr(msg, '__dict__'):
                                    print(f"\n[调试] Message属性: {list(msg.__dict__.keys())}")
                            except Exception as debug_e:
                                print(f"[调试] 无法打印响应结构: {debug_e}")
                        
                        # 提取内容
                        if OPENAI_VERSION >= 2:
                            if isinstance(response, dict):
                                message = response['choices'][0]['message']
                            else:
                                message = response.choices[0].message
                            content = getattr(message, 'content', None)
                            if content is None and isinstance(message, dict):
                                content = message.get('content')
                        else:
                            message = response['choices'][0]['message']
                            content = message.get('content')
                        
                        output = content.strip() if content else ""
                        
                        # 尝试提取思考过程（某些API可能支持）
                        thinking = self._extract_thinking_from_response(response)
                        last_thinking = thinking
                        
                        # 调试：打印 annotations 和其他字段的详细信息
                        if sample_id and ('_1' in sample_id or sample_id.endswith('1')):
                            print(f"\n[调试] 检查响应字段...")
                            try:
                                if hasattr(response, 'choices'):
                                    choice = response.choices[0] if hasattr(response, 'choices') else None
                                    if choice:
                                        if hasattr(choice, 'message'):
                                            msg = choice.message
                                            if hasattr(msg, '__dict__'):
                                                print(f"[调试] Message的所有属性: {list(msg.__dict__.keys())}")
                                                
                                                # 特别检查 content 的类型和结构
                                                if hasattr(msg, 'content'):
                                                    content_val = msg.content
                                                    import json
                                                    if isinstance(content_val, list):
                                                        print(f"[调试] content 是列表，包含 {len(content_val)} 项")
                                                        for i, item in enumerate(content_val[:5]):  # 显示前5项
                                                            if isinstance(item, dict):
                                                                print(f"[调试] content[{i}] 类型: {item.get('type', 'unknown')}, 键: {list(item.keys())}")
                                                                # 如果是 reasoning 类型，打印内容预览
                                                                if item.get('type', '').lower() in ['reasoning', 'reasoning_text']:
                                                                    text = item.get('text') or item.get('content') or ""
                                                                    print(f"[调试] content[{i}] reasoning 内容预览: {str(text)[:300]}")
                                                            else:
                                                                print(f"[调试] content[{i}]: {str(item)[:100]}")
                                                    elif isinstance(content_val, str):
                                                        print(f"[调试] content 是字符串，长度: {len(content_val)}")
                                                        # 检查 content 开头是否包含思考过程标记
                                                        preview = content_val[:500]
                                                        print(f"[调试] content 预览: {preview}")
                                                        # 检查是否有明显的思考过程
                                                        thinking_indicators = ['think', 'reason', 'analyze', 'step', 'first', 'let me']
                                                        found_indicators = [ind for ind in thinking_indicators if ind in preview.lower()]
                                                        if found_indicators:
                                                            print(f"[调试] content 中可能包含思考过程（发现关键词: {found_indicators}）")
                                                
                                                # 特别检查 annotations
                                                if hasattr(msg, 'annotations'):
                                                    ann = msg.annotations
                                                    import json
                                                    if ann is None:
                                                        print(f"[调试] annotations 为 None")
                                                    elif isinstance(ann, (list, dict)):
                                                        print(f"[调试] annotations 内容: {json.dumps(ann, default=str, ensure_ascii=False)[:1000]}")
                                                    else:
                                                        print(f"[调试] annotations: {str(ann)[:500]}")
                                                
                                                # 检查 response 和 choice 是否有 reasoning 字段
                                                if hasattr(response, 'reasoning'):
                                                    print(f"[调试] response.reasoning 存在: {str(response.reasoning)[:200]}")
                                                if hasattr(choice, 'reasoning'):
                                                    print(f"[调试] choice.reasoning 存在: {str(choice.reasoning)[:200]}")
                                                
                                                # 打印所有非 content 字段
                                                for attr in msg.__dict__.keys():
                                                    val = getattr(msg, attr, None)
                                                    if val and attr != 'content' and attr != 'annotations':
                                                        print(f"[调试] {attr}: {str(val)[:200]}")
                            except Exception as debug_e2:
                                print(f"[调试] 检查响应字段时出错: {debug_e2}")
                            
                            if thinking:
                                print(f"[调试] 成功提取思考过程，长度: {len(thinking)} 字符")
                                print(f"[调试] 思考过程预览: {thinking[:200]}...")
                            else:
                                print(f"[调试] 未找到思考过程")
                        
                    except Exception as e:
                        # 如果直接调用失败，回退到使用 generate 方法
                        print(f'直接调用API失败，使用generate方法: {e}')
                        output = self.openai_api.generate(prompt)
                        thinking = None
                        last_thinking = None
                else:
                    # 回退到使用 generate 方法
                    output = self.openai_api.generate(prompt)
                    thinking = None
                    last_thinking = None
                
            except Exception as e:
                output = ''
                thinking = None
                print(f'Error generating logic program for {sample_id} (attempt {attempt}): {e}')
            
            last_output = output if isinstance(output, str) else ''
            if self.validate_logic_program(last_output):
                return last_output.strip(), last_thinking
            
            # 打印调试信息
            if last_output:
                preview = last_output[:200] + "..." if len(last_output) > 200 else last_output
                print(f'Invalid logic program for {sample_id} on attempt {attempt}. Output preview: {preview}')
            else:
                print(f'Empty output for {sample_id} on attempt {attempt}, retrying...')
        
        print(f'Failed to obtain valid logic program for {sample_id} after {self.max_retries} attempts.')
        if last_output:
            preview = last_output[:500] + "..." if len(last_output) > 500 else last_output
            print(f'Last output was: {preview}')
        
        # 如果所有重试都失败，返回 None 而不是空字符串
        if not last_output or not last_output.strip():
            return None, last_thinking
        return last_output.strip(), last_thinking

    def logic_program_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        outputs = []
        for example in tqdm(raw_dataset):
            # create prompt
            try:
                full_prompt = self.prompt_creator[self.dataset_name](example)
                program, thinking = self.generate_program_with_retry(full_prompt, example['id'])
                # 如果生成失败（返回 None）或仍然是空字符串，跳过该样本
                if program is None or not program or not program.strip():
                    print(f'Skipping example {example["id"]} due to generation failure')
                    continue
                programs = [program]

                # create output
                output = {'id': example['id'], 
                        'context': example['context'],
                        'question': example['question'], 
                        'answer': example['answer'],
                        'options': example['options'],
                        'raw_logic_programs': programs}
                
                # 添加思考过程
                if thinking:
                    output['thinking'] = thinking
                
                outputs.append(output)
            except Exception as e:
                print(f'Error in generating logic programs for example {example["id"]}: {e}')

        # save outputs        
        with open(os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    '''
    Updated version of logic_program_generation; speed up the generation process by batching
    注意：为了确保能够提取思考过程，对于支持思考过程的API（如智谱AI），
    即使设置了批量大小，也会使用逐个生成的方式以确保能够获取完整的响应。
    '''
    def batch_logic_program_generation(self, batch_size = 10, max_concurrent = None):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        
        if max_concurrent:
            print(f"使用并发数: {max_concurrent}")

        # 为了确保能够提取思考过程，对于所有API都使用逐个生成
        # 因为批量生成可能无法获取完整的响应对象来提取思考过程
        # 如果 batch_size > 1，仍然会分批处理，但每个样本单独调用API
        use_individual_generation = True  # 始终使用逐个生成以确保能提取思考过程
        
        if use_individual_generation:
            print("注意: 使用逐个生成模式以确保能够提取思考过程")
            outputs = []
            thinking_count = 0
            for example in tqdm(raw_dataset):
                try:
                    full_prompt = self.prompt_creator[self.dataset_name](example)
                    program, thinking = self.generate_program_with_retry(full_prompt, example['id'])
                    # 如果生成失败（返回 None）或仍然是空字符串，跳过该样本
                    if program is None or not program or not program.strip():
                        print(f'Skipping example {example["id"]} due to generation failure')
                        continue
                    programs = [program]
                    output = {'id': example['id'], 
                            'context': example['context'],
                            'question': example['question'], 
                            'answer': example['answer'],
                            'options': example['options'],
                            'raw_logic_programs': programs}
                    
                    # 添加思考过程
                    if thinking:
                        output['thinking'] = thinking
                        thinking_count += 1
                    
                    outputs.append(output)
                except Exception as e:
                    print(f'Error in generating logic programs for example {example["id"]}: {e}')
            
            print(f"\n思考过程提取统计: {thinking_count}/{len(outputs)} 个样本包含思考过程")
        else:
            # 对于不支持思考过程或批量生成可以获取完整响应的API，使用批量生成
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
                        thinking = None
                        
                        if not self.validate_logic_program(program):
                            program, thinking = self.generate_program_with_retry(
                                self.prompt_creator[self.dataset_name](sample),
                                sample['id']
                            )
                        
                        # 如果生成失败（返回 None）或仍然是空字符串，跳过该样本
                        if program is None or not program or not program.strip():
                            print(f'Skipping example {sample["id"]} due to generation failure')
                            continue
                        programs = [program]
                        output = {'id': sample['id'], 
                                'context': sample['context'],
                                'question': sample['question'], 
                                'answer': sample['answer'],
                                'options': sample['options'],
                                'raw_logic_programs': programs}
                        
                        # 添加思考过程（批量生成时可能无法获取）
                        if thinking:
                            output['thinking'] = thinking
                        
                        outputs.append(output)
                except:
                    # generate one by one if batch generation fails
                    for sample, full_prompt in zip(chunk, full_prompts):
                        try:
                            program, thinking = self.generate_program_with_retry(full_prompt, sample['id'])
                            # 如果生成失败（返回 None）或仍然是空字符串，跳过该样本
                            if program is None or not program or not program.strip():
                                print(f'Skipping example {sample["id"]} due to generation failure')
                                continue
                            programs = [program]
                            output = {'id': sample['id'], 
                                    'context': sample['context'],
                                    'question': sample['question'], 
                                    'answer': sample['answer'],
                                    'options': sample['options'],
                                    'raw_logic_programs': programs}
                            
                            # 添加思考过程
                            if thinking:
                                output['thinking'] = thinking
                            
                            outputs.append(output)
                        except Exception as e:
                            print(f'Error in generating logic programs for example {sample["id"]}: {e}')

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
    logic_program_generator = LogicProgramGeneratorWithThinking(args)
    logic_program_generator.batch_logic_program_generation(
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent
    )

