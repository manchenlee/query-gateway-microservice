import asyncio
import pandas as pd
import httpx
import time
import random
from config import settings

async def send_request(client, text, user_id):
    payload = {"text": text}
    start = time.perf_counter()
    try:
        resp = await client.post(f"http://{settings.SERVER_HOST}:{str(settings.SERVER_PORT)}/v1/query-classify", json=payload, timeout=20.0)
        latency = (time.perf_counter() - start) * 1000
        router_latency = resp.headers.get("x-router-latency", "10")
        
        print(f"[User {user_id:03d}] Result: {resp.json()} | Total: {latency:.1f}ms | Router: {router_latency}ms")
        return latency
    except Exception as e:
        print(f"[User {user_id:03d}] Error: {e}")
        return None

async def run_stress_test(num_requests=5000, concurrency=20):
    df = pd.read_csv('data/data_test.csv')
    test_texts = df['instruction'].tolist()
    sem = asyncio.Semaphore(concurrency)

    async def wrapped_request(client, text, i):
        async with sem:
            return await send_request(client, text, i)
    
    async with httpx.AsyncClient() as client:
        start_time = time.perf_counter()
        
        tasks = [
            asyncio.create_task(wrapped_request(client, random.choice(test_texts), i))
            for i in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)
        
        total_duration = time.perf_counter() - start_time

        valid_results = [r for r in results if r is not None]
        
        print("-" * 30)
        print(f"Test Completed.")
        print(f"Total Requests: {len(valid_results)}")
        print(f"Total Duration: {total_duration:.2f}s")
        print(f"TPS: {len(valid_results)/total_duration:.2f}")
        if valid_results:
            print(f"Mean Latentcy: {sum(valid_results)/len(valid_results):.1f}ms")

if __name__ == "__main__":
    asyncio.run(run_stress_test(num_requests=1000, concurrency=100))