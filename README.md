# Query Gateway Microservice
This repository is for the homework "Intelligence Query Gateway Microservices".


## Environment
python 3.11.9

## Setup

```bash
pip install -r requirements.txt
```

Configure your environment variables by copying `.env.example` to `.env` and filling in the required settings.

```bash
cp .env.example .env
```

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
  
As shown in the comparison, `e5-small-v2` outperformed the other models, particularly in the General QA and Creative Writing categories. While `all-MiniLM-L6-v2` showed a slight lead in simple classification tasks, `e5-small-v2`'s superior semantic representation leads to a more balanced and higher overall F1 score, making it the most robust choice for a Semantic Router.  

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
| **Max Batch Size** | `8` | The maximum number of query chunks processed in a single model forward pass. |

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

Based on the updated HTTP report with **Batch Size = 8**, the system maintained high availability and a stable throughput of approximately **191 RPS** throughout the test duration.

| Metric | Value | Description |
| --- | --- | --- |
| **Total Requests** | 74,575 | Total successful HTTP transactions processed. |
| **Throughput** | 190.92 req/s | Average requests handled per second. |
| **Success Rate (HTTP 200)** | 100.00% | No failed requests or connection errors detected. |
| **Classification Accuracy** | 88.06% | Real-time accuracy remained consistent with smaller batching. |

#### 2. Router Latency Analysis (Trends & Times)

The latency metrics below reflect the system's performance at a peak of **258 Virtual Users** using the reduced batch size configuration.

| Metric | P50 (Median) | P95 | P99 | P99.9 |
| --- | --- | --- | --- | --- |
| **http_req_duration** | **74.67 ms** | **1061.11 ms** | **1195.01 ms** | 2209.67 ms |
| **http_req_waiting** | 74.54 ms | 1061.00 ms | 1194.91 ms | 2209.56 ms |

**Key Observations:**

* **Latency Trade-off:** With **Batch Size = 8**, the **P50 latency slightly increased to 74.67 ms** compared to the previous run (67.36 ms). This suggests that smaller batches lead to more frequent but smaller inference calls, slightly shifting the median overhead.
* **Improved Tail Stability:** Interestingly, the **P99 latency decreased significantly to 1195.01 ms** (down from 1291.66 ms). This indicates that reducing the batch size prevented long queue waiting times during peak bursts, providing a more "predictable" tail latency.
* **Asynchronous Integrity:** `http_req_connecting` and `http_req_blocked` remained at an average of **0.00 ms** and **0.01 ms** respectively, confirming that the smaller batch size did not introduce I/O blocking.

#### 3. Batching & Resource Efficiency

* **Throughput Consistency:** The system maintained an iteration rate of **191.07/s**, successfully keeping pace with the incoming request rate of **190.92/s**.
* **Resource Footprint:** Total data transfer reached **29.43 MB** (Sent: 17.50 MB / Received: 11.93 MB), showing that the change in batch size had a negligible impact on bandwidth consumption.

The **Batch Size = 8** configuration proved to be highly effective for stabilizing tail latency (P99), reducing it by nearly **100ms** compared to larger batch settings. While the median response time (P50) saw a minor increase, the overall system reliability remained excellent with a **100.00% success rate** and robust classification accuracy. This setting is recommended for environments prioritizing consistent tail-end performance.

Detailed performance metrics, including comprehensive visualizations, are available in the [summary](https://manchenlee.github.io/query_gateway_microservice/reports/summary_report_200_400_8_10) and [dashboard](https://manchenlee.github.io/query_gateway_microservice/reports/dashboard_200_400_8_10) HTML reports.

* batch_size_16: [summary](https://manchenlee.github.io/query_gateway_microservice/reports/summary_report_200_400_16_10)  
* batch_size_32: [summary](https://manchenlee.github.io/query_gateway_microservice/reports/summary_report) / [dashboard](https://manchenlee.github.io/query_gateway_microservice/reports/dashboard)
## Conclusion

This study demonstrates the efficacy of a high-performance Semantic Router optimized for real-time query classification and routing. By utilizing the `intfloat/e5-small-v2` embedding model combined with PCA dimensionality reduction and a confidence-aware threshold mechanism, the system achieves a robust overall F1-score of 0.88 while significantly reducing computational overhead.  

The integration of an asynchronous dynamic batching engine proves critical for scalability, enabling the gateway to maintain a 100% success rate under a sustained load of approximately 190 RPS. Although feature overlap between complex categories presents inherent classification challenges, the implemented thresholding strategy effectively mitigates misrouting risks. Stress test results, characterized by a median latency of 76.90 ms, confirm that the proposed architecture successfully balances semantic precision with operational efficiency, providing a reliable foundation for high-concurrency production environments.