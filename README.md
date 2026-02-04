# Query Gateway Microservice
This repository is for the homework "Intelligence Query Gateway Microservices".


## Environments
python 3.11.9

## Setup

Configure your environment variables by copying `.env.example` to `.env` and filling in the required settings.

### Dataset

This project utilizes a subset of the **databricks-dolly-15k** dataset, specifically extracting four categories: `classification`, `summarization`, `creative_writing`, and `general_qa`. The data is split into an 80/20 ratio and saved as `data_train.csv` and `data_test.csv`.

```bash
python data_split.py
```

### Generate Semantic Router
You can run the below command to generate model pipeline and the conrresponding report:  
```bash
python pipeline_gen.py --use_pca --use_threshold
```

### Docker Build & Run

Run the following commands to build and deploy the container:

```bash
docker build -t router-eval .
docker run -d -p 8000:8000 -p 5665:5665 --name k6-test router-eval
docker logs -f k6-test
```

After the stress test completes, export the report using:

```bash
mkdir -p ./reports/k6_test/
docker cp k6-test:/app/summary_report.html ./reports/k6_test/<output_name>.html
```

## Methodology & Evaluation

### Semantic Router

Given that this query gateway must handle high-concurrency demands, an external LLM-based approach is unsuitable due to latency and cost constraints. Conversely, while traditional NLP methods and regular expressions offer high efficiency, they fail to grasp semantic context and synonyms. As a result, we implemented a semantic router using an embedding-based classifier to bridge the gap between performance and linguistic understanding.  

The **Semantic Router** is implemented using the below pipeline:

* **Embedding Model:** `intfloat/e5-small-v2`  

* **Text Chunking & Normalization:** Since the embedding model has a maximum sequence length of 512 tokens, long inputs are partitioned into segments (chunking) before being processed. The final label is determined via a **voting mechanism** across all chunks. Text normalization is applied as a preprocessing step prior to embedding generation.  

* **Dimensionality Reduction:** To optimize inference efficiency for production environments, **PCA (Principal Component Analysis)** is applied to reduce vector dimensions from 384 down to 256. We have verified that the reduced data retains approximately 90% of the original variance, ensuring that the dimensionality reduction does not compromise classification performance.  

* **Logistic Regression:** Given the binary classification nature of the task and the requirement for high inference throughput, **Logistic Regression** was selected as the core classifier. The model utilizes predicted probabilities as a **confidence score** for decision routing.  

* **Reports:** Evaluation results are stored in `reports/router_test`, and the fully trained model pipeline is exported to the `model` directory for deployment.  

#### 1. Embedding Model Comparison  
The following table compares different 384-dimensional embedding models (`e5-small-v2`, `all-MiniLM-L6-v2` and `bge-small-en-v1.5`). To ensure a fair comparison, all models were tested using the same **Logistic Regression** classifier, **PCA reduction (256D)**, and a **Confidence Threshold of 0.6**.

| Model | Class. (Acc.) | Summ. (Acc.) | Creat. (Acc.) | Gen. QA (Acc.) | Overall Acc. | Precision | Recall | F1 Score |  
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |  
| **e5-small-v2** | 96.25% | **78.15%** | **92.03%** | **84.16%** | **0.88** | **0.88** | **0.88** | **0.88** |  
| all-MiniLM-L6-v2 | **96.96%** | 76.05% | 89.86% | 80.09% | 0.86 | 0.86 | 0.86 | 0.86 |  
| bge-small-en-v1.5 | 95.08% | 77.31% | 89.13% | 79.64% | 0.86 | 0.86 | 0.85 | 0.86 |  
  
As shown in the table above, `e5-small-v2` outperformed the other models, particularly in the General QA and Creative Writing categories. While `all-MiniLM-L6-v2` showed a slight lead in simple classification tasks, `e5-small-v2`'s superior semantic representation leads to a more balanced and higher overall F1 score, making it the most robust choice for a Semantic Router.  

#### 2. Ablation Study - Feature Enginearing   

Using `e5-small-v2` as the base model, we conducted an ablation study to justify the use of PCA for dimensionality reduction and the confidence threshold for decision filtering.

| Configuration | Class. (Acc.) | Summ. (Acc.) | Creat. (Acc.) | Gen. QA (Acc.) | Overall Acc. | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PCA + Threshold** | 96.25% | 78.15% | 92.03% | 84.16% | **0.88** | **0.88** | 0.88 | **0.88** |
| **PCA Only** | 95.08% | 71.43% | **94.20%** | 87.78% | **0.88** | **0.88** | 0.88 | **0.88** |
| **Threshold Only** | **98.13%** | **80.67%** | 86.96% | 81.00% | 0.87 | **0.88** | 0.87 | 0.87 |
| **Baseline (None)** | 96.02% | 69.33% | 92.75% | **88.01%** | **0.88** | **0.88** | **0.88** | **0.88** |

As shown in the ablation study, reducing dimensions to **256D via PCA** maintains a consistent **0.88 F1-score** relative to the 384D baseline while successfully lowering memory footprint and compute latency. The **Threshold mechanism** proves indispensable for handling "noisy" categories; specifically, it elevates **Summarization** accuracy from 69% to 78%, ensuring that high-risk or ambiguous queries are reliably routed to the fallback path.

Our primary objective is to maintain high response quality by minimizing misrouting. To achieve this, we prioritize maximizing precision for **Creative Writing** and **General QA** while ensuring no single category suffers from disproportionately low accuracy. 

A significant challenge identified during testing is the **feature overlap** between the `summarization` and `general_qa` categories. This is evident when examining misclassified samples from the summarization test set:

* `40: Tell me about the Mughal Empire?`
* `57: The paper "Attention is all you need" proposed...`
* `76: Is Gloomhaven on Steam worth buying?`
* `99: Tell me about the Battle of Rabb`
* `172: How large is the Tennessee River?`

With the exception of index 57, these samples are contextually indistinguishable from standard **General QA** prompts. This result suggests that even a 384-dimensional embedding model struggles to discern the subtle boundaries between these classes due to "noisy" ground truth labels in the dataset.

Balancing per-category precision, overall system stability, and inference efficiency, we have selected the **PCA (256D) + Threshold (0.6)** configuration as the standard for our stress testing and deployment.   


### Query Batch Engine

To meet the requirements of high concurrency and low-latency routing, the Gateway implements a custom **Asynchronous Dynamic Batching Engine**. This architecture ensures that the system remains responsive under heavy load while maximizing hardware utilization.

#### 1. High Concurrency & Asynchronous Architecture

The system is built on **FastAPI** and utilizes **Asyncio** to ensure non-blocking I/O operations.

* **Non-blocking Main Thread:** The HTTP server receives requests and immediately hands them off to a background worker, preventing the event loop from being blocked by heavy CPU-bound inference tasks.

* **Asynchronous Task Coordination:** Each request creates an internal `asyncio.Future`. The request waits asynchronously until the worker completes the batch and populates the result.

#### 2. Dynamic Batching Mechanism

Instead of performing inference on every single HTTP request, the system aggregates requests into batches.

| Parameter | Value | Description |
| --- | --- | --- |
| **Batch Window** | `10ms` | The maximum time the engine waits to collect requests before triggering inference. |
| **Max Batch Size** | `32` | The maximum number of query chunks processed in a single model forward pass. |

**How it works:**

1. **Request Aggregation:** Requests are placed into an `asyncio.Queue`.

2. **Adaptive Wait:** The `batch_worker` triggers as soon as the first request arrives. It continues to pull from the queue until either the **Batch Window** expires or the **Max Batch Size** is reached.

3. **Vectorized Inference:** The entire batch is encoded and classified in a single vectorized operation, significantly reducing the overhead compared to sequential processing.

#### 3. Chunking & Voting Strategy

To handle inputs exceeding the embedding model's token limit (512 tokens), the engine employs a sliding window approach:

* **Text Chunking:** Long texts are split into chunks of `1200` characters with a `200` character overlap to preserve local context.  

* **Soft Voting:** The engine first calculates the mean probability for the "Slow Path" (Label 1) across all chunks. The **final confidence** is then derived based on the winning class: if the final decision is Label 1, the confidence is the average label 1 probability $\bar{P}_{\text{label}=1}$; otherwise, it is the complement ($\bar{P}_{\text{label}=0} = 1.0 - \bar{P}_{\text{label}=1}$). This ensures the confidence score always represents the average probability of the selected label.

#### 4. Confidence-aware Routing

The system implements a safety-first routing logic based on the model's output probability (Softmax).

* **Confidence Calculation:** The confidence is defined as the probability of the predicted class. For binary classification, we focus on the probability of the "Slow Path" (Label 1).

* **Threshold Selection:** We set a **threshold = 0.6**.
    * If `label == 1` and `confidence < threshold`, the query is downgraded to the **Fast Path (0)**, even if the raw prediction was Label 1.


* **Rationale:** This threshold acts as a "confidence gate," ensuring that only queries with high semantic certainty trigger the resource-intensive Slow Path.

#### 5. Caching Mechanism

To further reduce latency for redundant queries:

* **TTL Cache:** An in-memory **LRU (Least Recently Used)** cache with a **Time-To-Live (TTL)** of 3600 seconds.
* **Deduplication:** Repeated queries bypass the Batch Engine entirely, returning results in sub-millisecond time.

### Stress Test

Performance and stress testing are conducted using **K6**. The testing strategy includes:

1. **Baseline Test:** A 3-minute stability test at a constant rate of **200 RPS**.
2. **Scaling Stress Test:** A 3-minute ramping test with stages at **100, 200, and 400 RPS**.
3. **Cooldown:** A 30-second window at 0 RPS to ensure the system successfully processes all remaining requests in the queue.

#### 1. Global Performance Metrics
Based on the overall HTTP report, the system maintains high availability and stable throughput even under concurrent stress.

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Total Requests** | 73,977 | Total successful HTTP transactions during the test. |
| **Throughput** | 189.53 req/s | Average requests processed per second. |
| **Success Rate (HTTP 200)** | 100.00% | Zero failed requests or connection errors detected. |
| **Classification Accuracy** | 88.03% | Real-time accuracy maintained during the stress test. |

#### 2. Router Latency Analysis (Trends & Times)
The latency metrics below include the end-to-end processing time (Embedding + PCA + Inference + Routing logic).

| Metric | P50 (Median) | P95 | P99 | P99.9 |
| :--- | :---: | :---: | :---: | :---: |
| **http_req_duration** | **76.90 ms** | **1072.09 ms** | **1415.87 ms** | 3118.80 ms |
| **http_req_waiting** | 76.78 ms | 1071.96 ms | 1415.80 ms | 3118.71 ms |

**Key Observations:**
* **Median Stability:** The **P50 latency of 76.90 ms** demonstrates that for the majority of requests, the **Query Batch Engine** successfully processes queries within a very responsive timeframe, including the 10ms batching window.
* **Tail Latency (P95/P99):** The jump to **1072ms (P95)** and **1415ms (P99)** indicates the simulated overhead of the **Slow Path** and potential queue congestion during peak bursts when the `MAX_BATCH_SIZE` is saturated.
* **Non-Blocking Efficiency:** Despite the tail latencies, the `http_req_connecting` and `http_req_blocked` times remain near **0.01 ms**, proving that our asynchronous `FastAPI` + `asyncio.Queue` architecture effectively prevents thread blocking.


#### 3. Batching & Resource Efficiency
* **Batch Utilization:** During the peak stage (up to 312 Virtual Users), the system maintained a consistent iteration rate of **189.65/s**, closely matching the request rate. This indicates that the batching mechanism successfully consolidated individual requests into optimized matrix operations.
* **Data Throughput:** The system handled a total of **29.06 MB** of bidirectional data transfer with a steady rate of **0.07 MB/s**, showing low network overhead relative to the complexity of the classification task.

The stress test confirms that the Router can handle sustained traffic of **~190 RPS** with a **100% success rate**. While the Slow Path and batching contention increase tail latency (P99), the median response time remains under **80ms**, making it highly suitable for real-time semantic routing.

Detailed performance metrics, including comprehensive visualizations, are available in the [summary](https://manchenlee.github.io/query_gateway_microservice/reports/summary_report) and [dashboard](https://manchenlee.github.io/query_gateway_microservice/reports/dashboard) HTML reports.  

## Conclusion

This study demonstrates the efficacy of a high-performance Semantic Router optimized for real-time query classification and routing. By utilizing the `intfloat/e5-small-v2` embedding model combined with PCA dimensionality reduction and a confidence-aware threshold mechanism, the system achieves a robust overall F1-score of 0.88 while significantly reducing computational overhead.  

The integration of an asynchronous dynamic batching engine proves critical for scalability, enabling the gateway to maintain a 100% success rate under a sustained load of approximately 190 RPS. Although feature overlap between complex categories presents inherent classification challenges, the implemented thresholding strategy effectively mitigates misrouting risks. Stress test results, characterized by a median latency of 76.90 ms, confirm that the proposed architecture successfully balances semantic precision with operational efficiency, providing a reliable foundation for high-concurrency production environments.