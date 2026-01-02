import asyncio
import aiohttp
import time
import json

# 配置
API_URL = "http://localhost:8668/v1/chat/completions"
API_KEY = "sk-12345678"
MODEL = "Qwen3-8B-5"
CONCURRENT_REQUESTS = 200  # 同时发送 50 个请求

async def send_request(session, request_id):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a logic parser for the **ProofWriter** dataset.\nYour task is to translate natural language into a structured rule-based representation.\n\nOutput Requirements (STRICT):\n- Use exactly four section headers: `Predicates:`, `Facts:`, `Rules:`, `Query:`.\n- Each line format: `Predicate(args) ::: natural language explanation`.\n- Represent negation via boolean arguments (e.g., `Red($x, False)`).\n- Use `>>>` for implication and `&&` for conjunction.\n- Ensure every variable in the head of a rule appears in the body.\n\nRespond in plain text only. Do not add any conversational filler."},
            {"role": "user", "content": "Bob is cold. Bob is quiet. Bob is red. Bob is smart. Charlie is kind. Charlie is quiet. Charlie is red. Charlie is rough. Dave is cold. Dave is kind. Dave is smart. Fiona is quiet. If something is quiet and cold then it is smart. Red, cold things are round. If something is kind and rough then it is red. All quiet things are rough. Cold, smart things are red. If something is rough then it is cold. All red things are rough. If Dave is smart and Dave is kind then Dave is quiet.\n\nBased on the above information, is the following statement true, false, or unknown? Charlie is kind.\n\n\nOptions:\n\nA) True\n\nB) False\n\nC) Unknown  /no_think"}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    start_time = time.time()
    try:
        async with session.post(API_URL, json=payload, headers=headers) as response:
            result = await response.json()
            latency = time.time() - start_time
            print(f"请求 {request_id} 完成，耗时: {latency:.2f}s, 状态码: {response.status}")
            return latency
    except Exception as e:
        print(f"请求 {request_id} 失败: {e}")
        return 0

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        print(f"开始发送 {CONCURRENT_REQUESTS} 个并发请求...")
        
        # 瞬间创建所有任务
        for i in range(CONCURRENT_REQUESTS):
            tasks.append(send_request(session, i))
        
        # 并发执行
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    # 安装依赖: pip install aiohttp
    asyncio.run(main())