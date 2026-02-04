import os
import asyncio
import time
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from config import settings
from cachetools import TTLCache
from utils import *

os.environ["HUGGING_FACE_HUB_TOKEN"] = settings.HF_TOKEN

class QueryBatchEngine:
    def __init__(self, model_path, batch_size=settings.MAX_BATCH_SIZE, window_ms=settings.BATCH_WINDOW_MS):
        self.encoder = SentenceTransformer(settings.ENCODER_NAME)
        self.router = joblib.load(model_path)
        self.queue = asyncio.Queue()
        self.batch_size = batch_size
        self.window_seconds = window_ms / 1000.0
        self.cache = TTLCache(maxsize=1000, ttl=3600)
        self.normalize = normalize_text
        self.max_chars = 1200 
        self.overlap = 200

    async def classify(self, text: str):
        if self.queue.qsize() > 200:
            return {"error": "Server busy", "status": 503}
        text = self.normalize(text)

        # cache
        if text in self.cache:
            return self.cache[text]

        # data chunking
        chunks = []
        if len(text) <= self.max_chars:
            chunks = [text]
        else:
            step = self.max_chars - self.overlap
            for i in range(0, len(text), step):
                chunks.append(text[i : i + self.max_chars])
                if i + self.max_chars >= len(text): break

        futures = [asyncio.get_event_loop().create_future() for _ in chunks]
        tasks = [self.queue.put((chunk, f)) for chunk, f in zip(chunks, futures)]
        await asyncio.gather(*tasks)

        # get results
        chunk_results = await asyncio.gather(*futures)

        # voting
        all_label_1_probs = [res["probs"][1] for res in chunk_results]
        avg_label_1_prob = np.mean(all_label_1_probs)
        
        if settings.THRESHOLD:
            final_label = 1 if avg_label_1_prob >= settings.THRESHOLD else 0
        else:
            final_label = 1 if avg_label_1_prob >= 0.5 else 0

        final_confidence = avg_label_1_prob if final_label == 1 else (1.0 - avg_label_1_prob)

        res = {
            "label": int(final_label),
            "confidence": float(final_confidence),
        }
        
        self.cache[text] = res
        return res

    async def batch_worker(self):
        print('Start dynamic batch worker...')
        while True:
            text, future = await self.queue.get()
            batch = [(text, future)]
            await asyncio.sleep(0)
            start_t = time.perf_counter()

            # wait for the end of batch window time or full of batch 
            while len(batch) < self.batch_size:
                elapsed = (time.perf_counter() - start_t) * 1000
                if elapsed >= settings.BATCH_WINDOW_MS:
                    break
                try:
                    t, f = self.queue.get_nowait()
                    batch.append((t, f))
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)
            
            with open("batch_monitor.log", "a") as f:
                f.write(f"Time: {time.time()} | BatchSize: {len(batch)} / {settings.MAX_BATCH_SIZE} | QSize: {self.queue.qsize()}\n")

            # feed the batch of samples into encoder
            texts = [b[0] for b in batch]
            embs = self.encoder.encode(texts)
            
            probs = self.router.predict_proba(embs)
            labels = np.argmax(probs, axis=1)
            
            for i, (_, f) in enumerate(batch):
                if not f.done():
                    f.set_result({
                        "label": int(labels[i]),
                        "probs": probs[i].tolist(),
                        "confidence": float(probs[i][labels[i]]) 
                    })