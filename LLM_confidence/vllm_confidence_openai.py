#!/usr/bin/env python3
"""
使用 OpenAI 库调用 vLLM 并获取置信度

该脚本使用 openai 库（兼容 vLLM）来获取模型输出的置信度
"""

import openai
import math
from typing import Dict, List, Any, Optional


class VLLMConfidenceClient:
    """使用 OpenAI 库的 vLLM 置信度客户端"""
    
    def __init__(self, api_url: str = "http://localhost:8668/v1", api_key: str = "sk-12345678"):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=api_url
        )
    
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
        try:
            response = self.client.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=top_logprobs,
                echo=False
            )
            
            return self._parse_response(response)
        
        except Exception as e:
            return {"error": f"请求失败: {str(e)}"}
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """解析响应"""
        choice = response.choices[0]
        text = choice.text
        logprobs_data = choice.logprobs
        
        tokens = logprobs_data.tokens if logprobs_data else []
        token_logprobs = logprobs_data.token_logprobs if logprobs_data else []
        top_logprobs = logprobs_data.top_logprobs if logprobs_data else []
        
        result = {
            "text": text,
            "finish_reason": choice.finish_reason,
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "average_confidence": self._calculate_average_confidence(token_logprobs),
            "token_details": self._get_token_details(tokens, token_logprobs, top_logprobs)
        }
        
        return result
    
    def _calculate_average_confidence(self, token_logprobs: List[float]) -> Optional[float]:
        """计算平均置信度（从logprobs转换为概率）"""
        if not token_logprobs:
            return None
        
        probabilities = [math.exp(logprob) for logprob in token_logprobs if logprob is not None]
        
        if not probabilities:
            return None
        
        return sum(probabilities) / len(probabilities)
    
    def _get_token_details(
        self, 
        tokens: List[str], 
        token_logprobs: List[float],
        top_logprobs: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """获取每个token的详细信息"""
        details = []
        
        for i, (token, logprob) in enumerate(zip(tokens, token_logprobs)):
            prob = math.exp(logprob) if logprob is not None else 0
            
            detail = {
                "position": i + 1,
                "token": token,
                "logprob": logprob,
                "probability": prob
            }
            
            if top_logprobs and i < len(top_logprobs) and top_logprobs[i]:
                top_alternatives = []
                for alt_token, alt_logprob in top_logprobs[i].items():
                    alt_prob = math.exp(alt_logprob)
                    top_alternatives.append({
                        "token": alt_token,
                        "logprob": alt_logprob,
                        "probability": alt_prob
                    })
                
                top_alternatives.sort(key=lambda x: x["probability"], reverse=True)
                detail["top_alternatives"] = top_alternatives
            
            details.append(detail)
        
        return details


def print_token_details(details: List[Dict[str, Any]], max_tokens: int = 10):
    """打印token详细信息"""
    print(f"\n前 {min(max_tokens, len(details))} 个token的详细信息:")
    print("-" * 80)
    
    for detail in details[:max_tokens]:
        pos = detail["position"]
        token = detail["token"]
        logprob = detail["logprob"]
        prob = detail["probability"]
        
        print(f"\nToken {pos}: '{token}'")
        print(f"  Logprob: {logprob:.4f} | 概率: {prob:.4f} ({prob*100:.2f}%)")
        
        if "top_alternatives" in detail and detail["top_alternatives"]:
            print(f"  Top-5 候选:")
            for i, alt in enumerate(detail["top_alternatives"][:5], 1):
                alt_token = alt["token"]
                alt_prob = alt["probability"]
                print(f"    {i}. '{alt_token}' - {alt_prob:.4f} ({alt_prob*100:.2f}%)")


def main():
    """示例用法"""
    client = VLLMConfidenceClient()
    
    print("=" * 80)
    print("使用 OpenAI 库调用 vLLM 并获取置信度")
    print("=" * 80)
    
    print("\n测试示例 1: 解释概念")
    print("-" * 80)
    prompt = "请解释什么是人工智能："
    result = client.generate_with_confidence(prompt, max_tokens=150, top_logprobs=5)
    
    if "error" in result:
        print(f"错误: {result['error']}")
    else:
        print(f"\n生成文本:\n{result['text']}")
        print(f"\n统计信息:")
        print(f"  总token数: {len(result['tokens'])}")
        print(f"  平均置信度: {result['average_confidence']:.4f} ({result['average_confidence']*100:.2f}%)")
        print(f"  结束原因: {result['finish_reason']}")
        
        print_token_details(result['token_details'], max_tokens=5)
    
    print("\n" + "=" * 80)
    print("\n测试示例 2: 逻辑推理")
    print("-" * 80)
    prompt = "如果所有的猫都是动物，而所有的动物都需要食物，那么可以推断出："
    result = client.generate_with_confidence(prompt, max_tokens=100, temperature=0.3, top_logprobs=3)
    
    if "error" in result:
        print(f"错误: {result['error']}")
    else:
        print(f"\n生成文本:\n{result['text']}")
        print(f"\n统计信息:")
        print(f"  总token数: {len(result['tokens'])}")
        print(f"  平均置信度: {result['average_confidence']:.4f} ({result['average_confidence']*100:.2f}%)")
        print(f"  结束原因: {result['finish_reason']}")
        
        print_token_details(result['token_details'], max_tokens=8)
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
