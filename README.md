# Query Gateway Microservice

## Setup

Configure your environment variables by copying `.env.example` to `.env` and filling in the required settings.

### Dataset

This project utilizes a subset of the **databricks-dolly-15k** dataset, specifically extracting four categories: `classification`, `summarization`, `creative_writing`, and `general_qa`. The data is split into an 80/20 ratio and saved as `data_train.csv` and `data_test.csv`.

```bash
python data_split.py

```

### Model

The **Semantic Router** is implemented using an embedding-based classifier.

* **Embedding Model:** `intfloat/e5-small-v2`
* **Dimensionality Reduction:** To optimize inference efficiency for production, **PCA** is applied to reduce the vector dimensions from 384 down to 256.
* **Artifacts:** Evaluation results are saved in `reports/router_test`, and the trained model pipeline is exported to the `model` directory.

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

empty

### Stress Test

empty
