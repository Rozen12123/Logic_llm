#!/usr/bin/env python3
"""
vLLM 置信度获取脚本

该脚本演示如何从 vLLM 部署的模型中获取输出的置信度（logprobs）
"""

import requests
import json
from typing import Dict, List, Any, Optional


class VLLMConfidenceClient:
    """vLLM 置信度客户端"""
    
    def __init__(self, api_url: str = "http://localhost:8668/v1", api_key: str = "sk-12345678"):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def generate_with_confidence(
        self,
        prompt: str,
        model: str = "Qwen3-8B-1203",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_logprobs: int = 5
    ) -> Dict[str, Any]:
        """
        生成文本并获取置信度
        
        Args:
            prompt: 输入提示
            model: 模型名称
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_logprobs: 返回每个位置的top-k个token的logprobs
        
        Returns:
            包含生成文本和置信度信息的字典
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": top_logprobs,
            "echo": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            return self._parse_response(result)
        
        except requests.exceptions.RequestException as e:
            return {"error": f"请求失败: {str(e)}"}
    
    def chat_with_confidence(
        self,
        messages: List[Dict[str, str]],
        model: str = "Qwen3-8B-5",
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_logprobs: int = 5
    ) -> Dict[str, Any]:
        """
        使用对话格式生成文本并获取置信度
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            max_tokens: 最大生成token数
            temperature: 温度参数
            top_logprobs: 返回每个位置的top-k个token的logprobs
        
        Returns:
            包含生成文本和置信度信息的字典
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": top_logprobs
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            return self._parse_chat_response(result)
        
        except requests.exceptions.RequestException as e:
            return {"error": f"请求失败: {str(e)}"}
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析 completions API 响应"""
        if "error" in response:
            return {"error": response["error"]}
        
        choice = response["choices"][0]
        text = choice["text"]
        logprobs_data = choice.get("logprobs", {})
        
        result = {
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "tokens": logprobs_data.get("tokens", []),
            "token_logprobs": logprobs_data.get("token_logprobs", []),
            "top_logprobs": logprobs_data.get("top_logprobs", []),
            "average_confidence": self._calculate_average_confidence(
                logprobs_data.get("token_logprobs", [])
            )
        }
        
        return result
    
    def _parse_chat_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """解析 chat/completions API 响应"""
        if "error" in response:
            return {"error": response["error"]}
        
        choice = response["choices"][0]
        message = choice["message"]
        text = message["content"]
        logprobs_data = choice.get("logprobs", {})
        
        result = {
            "text": text,
            "finish_reason": choice.get("finish_reason"),
            "content": logprobs_data.get("content", []),
            "average_confidence": self._calculate_average_confidence_from_content(
                logprobs_data.get("content", [])
            )
        }
        
        return result
    
    def _calculate_average_confidence(self, token_logprobs: List[float]) -> Optional[float]:
        """计算平均置信度（从logprobs转换为概率）"""
        if not token_logprobs:
            return None
        
        import math
        probabilities = [math.exp(logprob) for logprob in token_logprobs if logprob is not None]
        
        if not probabilities:
            return None
        
        return sum(probabilities) / len(probabilities)
    
    def _calculate_average_confidence_from_content(self, content: List[Dict[str, Any]]) -> Optional[float]:
        """从 content 格式计算平均置信度"""
        if not content:
            return None
        
        import math
        logprobs = [item.get("logprob") for item in content if item.get("logprob") is not None]
        
        if not logprobs:
            return None
        
        probabilities = [math.exp(logprob) for logprob in logprobs]
        return sum(probabilities) / len(probabilities)


def main():
    """示例用法"""
    client = VLLMConfidenceClient()
    
    print("=" * 60)
    print("vLLM 置信度获取测试")
    print("=" * 60)
    
    print("\n1. 测试 Completions API:")
    print("-" * 60)
    prompt = "请解释什么是人工智能："
    result = client.generate_with_confidence(prompt, max_tokens=100)
    
    if "error" in result:
        print(f"错误: {result['error']}")
    else:
        print(f"生成文本: {result['text']}")
        print(f"平均置信度: {result['average_confidence']:.4f}")
        print(f"结束原因: {result['finish_reason']}")
        
        if result['tokens']:
            print(f"\n前5个token的详细信息:")
            for i, (token, logprob) in enumerate(zip(result['tokens'][:5], result['token_logprobs'][:5])):
                import math
                prob = math.exp(logprob) if logprob is not None else 0
                print(f"  Token {i+1}: '{token}' | logprob: {logprob:.4f} | prob: {prob:.4f}")
    
    # 注意：当前 vLLM 部署使用 --reasoning-parser 参数，可能只支持 Completions API
    # 如需使用 Chat Completions API，请在启动 vLLM 时移除 --reasoning-parser 参数
    
    # print("\n" + "=" * 60)
    # print("2. 测试 Chat Completions API:")
    # print("-" * 60)
    # messages = [
    #     {"role": "system", "content": "你是一个有帮助的助手。"},
    #     {"role": "user", "content": "什么是逻辑推理？"}
    # ]
    # result = client.chat_with_confidence(messages, max_tokens=100)
    
    # if "error" in result:
    #     print(f"错误: {result['error']}")
    # else:
    #     print(f"生成文本: {result['text']}")
    #     print(f"平均置信度: {result['average_confidence']:.4f}")
    #     print(f"结束原因: {result['finish_reason']}")
        
    #     if result['content']:
    #         print(f"\n前5个token的详细信息:")
    #         for i, item in enumerate(result['content'][:5]):
    #             import math
    #             logprob = item.get("logprob", 0)
    #             prob = math.exp(logprob) if logprob is not None else 0
    #             token = item.get("token", "")
    #             print(f"  Token {i+1}: '{token}' | logprob: {logprob:.4f} | prob: {prob:.4f}")
    
    print("\n" + "=" * 60)
    print("提示：当前使用 Completions API 成功获取置信度")
    print("如需 Chat API，请修改 vLLM 启动参数")
    print("=" * 60)


if __name__ == "__main__":
    main()
