import backoff  # for exponential backoff
import openai
import os
import asyncio
import json
from typing import Any

# 尝试导入智谱AI库
try:
    from zhipuai import ZhipuAI
    ZHIPU_AI_AVAILABLE = True
except ImportError:
    try:
        # 尝试旧版本的 zai 库
        from zai import ZhipuAiClient
        ZhipuAI = ZhipuAiClient  # 兼容性别名
        ZHIPU_AI_AVAILABLE = True
    except ImportError:
        ZHIPU_AI_AVAILABLE = False
        ZhipuAI = None

# 检测 openai 库版本并设置兼容性
try:
    # 新版本 openai (2.x+)
    if hasattr(openai, 'OpenAI'):
        OPENAI_VERSION = 2
        # 新版本的异常类
        try:
            RateLimitError = openai.RateLimitError
        except AttributeError:
            RateLimitError = openai.APIError
    else:
        # 旧版本 openai (0.x)
        OPENAI_VERSION = 0
        RateLimitError = openai.error.RateLimitError
except:
    OPENAI_VERSION = 0
    try:
        RateLimitError = openai.error.RateLimitError
    except:
        RateLimitError = Exception

@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff(client_or_none, **kwargs):
    if OPENAI_VERSION >= 2 and client_or_none:
        return client_or_none.completions.create(**kwargs)
    else:
        return openai.Completion.create(**kwargs)

@backoff.on_exception(backoff.expo, RateLimitError)
def chat_completions_with_backoff(client_or_none, **kwargs):
    if OPENAI_VERSION >= 2 and client_or_none:
        return client_or_none.chat.completions.create(**kwargs)
    else:
        return openai.ChatCompletion.create(**kwargs)

async def dispatch_openai_chat_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str],
    client=None
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
        client: OpenAI client (for v2+)
    Returns:
        List of responses from OpenAI API.
    """
    if OPENAI_VERSION >= 2 and client:
        async_responses = [
            client.chat.completions.create(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop = stop_words
            )
            for x in messages_list
        ]
    else:
        async_responses = [
            openai.ChatCompletion.acreate(
                model=model,
                messages=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop = stop_words
            )
            for x in messages_list
        ]
    return await asyncio.gather(*async_responses)

async def dispatch_openai_prompt_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str],
    client=None
) -> list[str]:
    if OPENAI_VERSION >= 2 and client:
        async_responses = [
            client.completions.create(
                model=model,
                prompt=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty = 0.0,
                presence_penalty = 0.0,
                stop = stop_words
            )
            for x in messages_list
        ]
    else:
        async_responses = [
            openai.Completion.acreate(
                model=model,
                prompt=x,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty = 0.0,
                presence_penalty = 0.0,
                stop = stop_words
            )
            for x in messages_list
        ]
    return await asyncio.gather(*async_responses)

class OpenAIModel:
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        self.API_KEY = API_KEY
        if OPENAI_VERSION >= 2:
            self.client = openai.OpenAI(api_key=API_KEY)
        else:
            openai.api_key = API_KEY
            self.client = None
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.stop_words = stop_words

    # used for chat-gpt and gpt-4
    def chat_generate(self, input_string, temperature = 0.0):
        response = chat_completions_with_backoff(
                self.client if OPENAI_VERSION >= 2 else None,
                model = self.model_name,
                messages=[
                        {"role": "user", "content": input_string}
                    ],
                max_tokens = self.max_new_tokens,
                temperature = temperature,
                top_p = 1.0,
                stop = self.stop_words
        )
        if OPENAI_VERSION >= 2:
            generated_text = response.choices[0].message.content.strip()
        else:
            generated_text = response['choices'][0]['message']['content'].strip()
        return generated_text
    
    # used for text/code-davinci
    def prompt_generate(self, input_string, temperature = 0.0):
        response = completions_with_backoff(
            self.client if OPENAI_VERSION >= 2 else None,
            model = self.model_name,
            prompt = input_string,
            max_tokens = self.max_new_tokens,
            temperature = temperature,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0,
            stop = self.stop_words
        )
        if OPENAI_VERSION >= 2:
            generated_text = response.choices[0].text.strip()
        else:
            generated_text = response['choices'][0]['text'].strip()
        return generated_text

    def generate(self, input_string, temperature = 0.0):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.prompt_generate(input_string, temperature)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.chat_generate(input_string, temperature)
        else:
            raise Exception("Model name not recognized")
    
    def batch_chat_generate(self, messages_list, temperature = 0.0, max_concurrent=None):
        open_ai_messages_list = []
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "user", "content": message}]
            )
        
        # 如果指定了并发数，使用信号量控制
        if max_concurrent and max_concurrent > 0:
            async def run_with_semaphore():
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def dispatch_with_semaphore(messages):
                    async with semaphore:
                        if OPENAI_VERSION >= 2 and self.client:
                            return await self.client.chat.completions.create(
                                model=self.model_name,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=self.max_new_tokens,
                                top_p=1.0,
                                stop=self.stop_words
                            )
                        else:
                            return await openai.ChatCompletion.acreate(
                                model=self.model_name,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=self.max_new_tokens,
                                top_p=1.0,
                                stop=self.stop_words
                            )
                
                tasks = [dispatch_with_semaphore(msgs) for msgs in open_ai_messages_list]
                return await asyncio.gather(*tasks)
            
            predictions = asyncio.run(run_with_semaphore())
        else:
            predictions = asyncio.run(
                dispatch_openai_chat_requests(
                    open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words, self.client
                )
            )
        
        if OPENAI_VERSION >= 2:
            return [x.choices[0].message.content.strip() for x in predictions]
        else:
            return [x['choices'][0]['message']['content'].strip() for x in predictions]
    
    def batch_prompt_generate(self, prompt_list, temperature = 0.0, max_concurrent=None):
        # 如果指定了并发数，使用信号量控制
        if max_concurrent and max_concurrent > 0:
            async def run_with_semaphore():
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def dispatch_with_semaphore(prompt):
                    async with semaphore:
                        if OPENAI_VERSION >= 2 and self.client:
                            return await self.client.completions.create(
                                model=self.model_name,
                                prompt=prompt,
                                temperature=temperature,
                                max_tokens=self.max_new_tokens,
                                top_p=1.0,
                                frequency_penalty=0.0,
                                presence_penalty=0.0,
                                stop=self.stop_words
                            )
                        else:
                            return await openai.Completion.acreate(
                                model=self.model_name,
                                prompt=prompt,
                                temperature=temperature,
                                max_tokens=self.max_new_tokens,
                                top_p=1.0,
                                frequency_penalty=0.0,
                                presence_penalty=0.0,
                                stop=self.stop_words
                            )
                
                tasks = [dispatch_with_semaphore(prompt) for prompt in prompt_list]
                return await asyncio.gather(*tasks)
            
            predictions = asyncio.run(run_with_semaphore())
        else:
            predictions = asyncio.run(
                dispatch_openai_prompt_requests(
                    prompt_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words, self.client
                )
            )
        
        if OPENAI_VERSION >= 2:
            return [x.choices[0].text.strip() for x in predictions]
        else:
            return [x['choices'][0]['text'].strip() for x in predictions]

    def batch_generate(self, messages_list, temperature = 0.0, max_concurrent=None):
        if self.model_name in ['text-davinci-002', 'code-davinci-002', 'text-davinci-003']:
            return self.batch_prompt_generate(messages_list, temperature, max_concurrent)
        elif self.model_name in ['gpt-4', 'gpt-3.5-turbo']:
            return self.batch_chat_generate(messages_list, temperature, max_concurrent)
        else:
            raise Exception("Model name not recognized")

    def generate_insertion(self, input_string, suffix, temperature = 0.0):
        response = completions_with_backoff(
            self.client if OPENAI_VERSION >= 2 else None,
            model = self.model_name,
            prompt = input_string,
            suffix= suffix,
            temperature = temperature,
            max_tokens = self.max_new_tokens,
            top_p = 1.0,
            frequency_penalty = 0.0,
            presence_penalty = 0.0
        )
        if OPENAI_VERSION >= 2:
            generated_text = response.choices[0].text.strip()
        else:
            generated_text = response['choices'][0]['text'].strip()
        return generated_text


class ZhipuAIModel:
    """智谱AI模型类，实现与OpenAIModel相同的接口"""
    def __init__(self, API_KEY, model_name, stop_words, max_new_tokens) -> None:
        if not ZHIPU_AI_AVAILABLE:
            raise ImportError("zhipuai库未安装，请运行: pip install zhipuai")
        
        self.client = ZhipuAI(api_key=API_KEY)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        # 处理stop_words：如果是字符串，转换为列表；如果是列表，直接使用；如果是None或空，使用空列表
        if isinstance(stop_words, str) and stop_words:
            self.stop_words = [stop_words]
        elif isinstance(stop_words, list):
            self.stop_words = stop_words
        else:
            self.stop_words = []
        self.thinking_config = self._resolve_thinking_config(model_name)

    def _resolve_thinking_config(self, model_name):
        """
        glm-4.6 等推理模型默认携带 reasoning_content，这里默认关闭思考模式，
        避免只返回中间推理步骤。
        """
        env_config = os.getenv("ZHIPUAI_THINKING_CONFIG")
        if env_config:
            parsed = self._parse_thinking_config(env_config)
            if parsed is not None:
                return parsed
        
        env_mode = os.getenv("ZHIPUAI_THINKING_MODE")
        if env_mode:
            mapped = self._map_thinking_mode(env_mode)
            if mapped is not None:
                return mapped
        
        reasoning_models = ("glm-4.6", "glm-4-long", "glm-4-air", "glm-4-plus")
        if any(tag in model_name for tag in reasoning_models):
            return {"type": "disabled"}
        return None

    def _parse_thinking_config(self, value):
        value = value.strip()
        if not value:
            return None
        if value.startswith("{") and value.endswith("}"):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return self._map_thinking_mode(value)

    def _map_thinking_mode(self, mode):
        normalized = mode.strip().lower()
        if normalized in ("disabled", "off", "false", "none", "0"):
            return {"type": "disabled"}
        if normalized in ("enabled", "on", "true", "default", "auto", "standard"):
            return None
        return None

    def _thinking_kwargs(self):
        return {"thinking": self.thinking_config} if self.thinking_config else {}

    def _extract_message_content(self, response):
        """兼容不同返回格式，优先使用content，退回reasoning_content"""
        if not response:
            return ""
        
        choices = getattr(response, 'choices', None)
        if choices is None and isinstance(response, dict):
            choices = response.get('choices')
        if not choices:
            return ""
        
        choice = choices[0]
        message = getattr(choice, 'message', choice)
        
        def _normalize(value):
            if value is None:
                return ""
            if isinstance(value, list):
                parts = []
                for item in value:
                    if isinstance(item, dict):
                        parts.append(item.get('text') or item.get('content') or "")
                    else:
                        parts.append(str(item))
                return ''.join(parts)
            return str(value)
        
        content = _normalize(getattr(message, 'content', None) or (message.get('content') if isinstance(message, dict) else None)).strip()
        if content:
            return content
        
        reasoning_content = _normalize(getattr(message, 'reasoning_content', None) or (message.get('reasoning_content') if isinstance(message, dict) else None)).strip()
        return reasoning_content

    def chat_generate(self, input_string, temperature=0.0):
        """使用聊天模式生成文本"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": input_string}
                ],
                temperature=temperature,
                max_tokens=self.max_new_tokens,
                stop=self.stop_words if self.stop_words else None,
                **self._thinking_kwargs()
            )
            return self._extract_message_content(response)
        except Exception as e:
            raise Exception(f"ZhipuAI API调用失败: {str(e)}")
    
    def prompt_generate(self, input_string, temperature=0.0):
        """智谱AI主要支持聊天模式，prompt模式也使用chat_generate"""
        return self.chat_generate(input_string, temperature)

    def generate(self, input_string, temperature=0.0):
        """统一生成接口，智谱AI使用聊天模式"""
        return self.chat_generate(input_string, temperature)
    
    async def _async_chat_generate(self, message, temperature=0.0):
        """异步生成单个消息"""
        try:
            # 使用 run_in_executor 将同步调用转换为异步
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            def sync_call():
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": message}
                    ],
                    temperature=temperature,
                    max_tokens=self.max_new_tokens,
                    stop=self.stop_words if self.stop_words else None,
                    **self._thinking_kwargs()
                )
            
            response = await loop.run_in_executor(None, sync_call)
            return self._extract_message_content(response)
        except Exception as e:
            print(f"异步生成时出错: {str(e)}")
            return ""
    
    def batch_chat_generate(self, messages_list, temperature=0.0, max_concurrent=None):
        """批量生成（支持并发）"""
        if max_concurrent is None or max_concurrent <= 1:
            # 顺序处理
            results = []
            for message in messages_list:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": message}
                        ],
                        temperature=temperature,
                        max_tokens=self.max_new_tokens,
                        stop=self.stop_words if self.stop_words else None,
                        **self._thinking_kwargs()
                    )
                    results.append(self._extract_message_content(response))
                except Exception as e:
                    print(f"批量生成时出错: {str(e)}")
                    results.append("")
            return results
        else:
            # 并发处理
            async def run_concurrent():
                semaphore = asyncio.Semaphore(max_concurrent)
                
                async def generate_with_semaphore(message):
                    async with semaphore:
                        return await self._async_chat_generate(message, temperature)
                
                tasks = [generate_with_semaphore(msg) for msg in messages_list]
                return await asyncio.gather(*tasks)
            
            return asyncio.run(run_concurrent())
    
    def batch_prompt_generate(self, prompt_list, temperature=0.0, max_concurrent=None):
        """批量prompt生成（使用chat模式）"""
        return self.batch_chat_generate(prompt_list, temperature, max_concurrent)

    def batch_generate(self, messages_list, temperature=0.0, max_concurrent=None):
        """统一批量生成接口"""
        return self.batch_chat_generate(messages_list, temperature, max_concurrent)

    def generate_insertion(self, input_string, suffix, temperature=0.0):
        """插入生成（智谱AI可能不支持，使用chat模式）"""
        # 智谱AI可能不支持插入模式，使用拼接方式
        full_prompt = input_string + suffix
        return self.chat_generate(full_prompt, temperature)