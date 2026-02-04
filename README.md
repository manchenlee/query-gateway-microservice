# Query Gateway Microservice

## Setup

Configure your environment variables by copying `.env.example` to `.env` and filling in the required settings.

### Dataset

This project utilizes a subset of the **databricks-dolly-15k** dataset, specifically extracting four categories: `classification`, `summarization`, `creative_writing`, and `general_qa`. The data is split into an 80/20 ratio and saved as `data_train.csv` and `data_test.csv`.

```bash
python data_split.py
```

### Model
The **Semantic Router** is implemented using an embedding-based classifier pipeline:

* **Embedding Model:** `intfloat/e5-small-v2`  
* **Text Chunking & Normalization:** Since the embedding model has a maximum sequence length of 512 tokens, long inputs are partitioned into segments (chunking) before being processed. The final label is determined via a **voting mechanism** across all chunks. Text normalization is applied as a preprocessing step prior to embedding generation.  
* **Dimensionality Reduction:** To optimize inference efficiency for production environments, **PCA (Principal Component Analysis)** is applied to reduce vector dimensions from 384 down to 256.  
* **Logistic Regression:** Given the binary classification nature of the task and the requirement for high inference throughput, **Logistic Regression** was selected as the core classifier. The model utilizes predicted probabilities as a **confidence score** for decision routing.  
* **Reports:** Evaluation results are stored in `reports/router_test`, and the fully trained model pipeline is exported to the `model` directory for deployment.  

You can run the below command to generate pipeline and the conrresponding report:  
```bash
python pipeline_gen.py --use_pca --use_threshold
```

### Docker Build & Run

Performance and stress testing are conducted using **K6**. The testing strategy includes:

1. **Baseline Test:** A 3-minute stability test at a constant rate of **200 RPS**.
2. **Scaling Stress Test:** A 3-minute ramping test with stages at **100, 200, and 400 RPS**.
3. **Cooldown:** A 30-second window at 0 RPS to ensure the system successfully processes all remaining requests in the queue.

Run the following commands to build and deploy the container:

```bash
docker build -t router-eval .
docker run -d -p 8000:8000 -p 5665:5665 --name k6-test router-eval
docker logs -f k6-test
```

After the stress test completes, export the report using:

```bash
docker cp k6-test:/app/summary_report.html ./reports/k6_test/<output_name>.html
```

## Results

### Classification
#### Embedding Model Comparison  
The following table compares different 384-dimensional embedding models (`e5-small-v2`, `all-MiniLM-L6-v2` and `bge-small-en-v1.5`). To ensure a fair comparison, all models were tested using the same **Logistic Regression** classifier, **PCA reduction (256D)**, and a **Confidence Threshold of 0.6**.

| Model | Class. (Acc.) | Summ. (Acc.) | Creat. (Acc.) | Gen. QA (Acc.) | Overall Acc. | Precision | Recall | F1 Score |  
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |  
| **e5-small-v2** | 96.25% | **78.15%** | **92.03%** | **84.16%** | **0.88** | **0.88** | **0.88** | **0.88** |  
| all-MiniLM-L6-v2 | **96.96%** | 76.05% | 89.86% | 80.09% | 0.86 | 0.86 | 0.86 | 0.86 |  
| bge-small-en-v1.5 | 95.08% | 77.31% | 89.13% | 79.64% | 0.86 | 0.86 | 0.85 | 0.86 |  
  
As shown in the table above, `e5-small-v2` outperformed the other models, particularly in the General QA and Creative Writing categories. While `all-MiniLM-L6-v2` showed a slight lead in simple classification tasks, `e5-small-v2`'s superior semantic representation leads to a more balanced and higher overall F1 score, making it the most robust choice for a Semantic Router.  

#### Feature Engineering & Optimization Study    

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

### Stress Test

empty
