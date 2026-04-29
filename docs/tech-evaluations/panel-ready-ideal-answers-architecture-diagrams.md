# 20260429 - 2
Below is a **panel‑ready text‑only “ideal answers + architecture diagrams”** guide for your **60‑question open‑source bank**, using **Airflow / MLflow / FastAPI / Kafka / Milvus / Postgres / Redis / Kubernetes** (no Azure/AWS). You can paste each block into your interviewer sheet.

***

### 1. Batch ETL pipeline (Airflow / Postgres / Python)

**Question**  
Design a batch ETL pipeline using Python + Airflow on‑prem for daily ingestion from multiple sources.

**Ideal answer**  
- **Orchestration**: Use **Airflow** with DAGs for:
  - Extract from DBs/APIs → load into **Postgres / data lake** → transform features → push to feature store.
- **Idempotency**:  
  - Each task uses `ds` / `execution_date` as partition; writes overwrite same partition if needed.  
  - Use `upserts` or `INSERT ... ON CONFLICT` in Postgres.
- **Data quality**:  
  - Run **Great Expectations / custom checks** between extract and transform.  
  - If checks fail, fail the task and send alert.
- **Monitoring**:  
  - Airflow logs + **Prometheus + Grafana** for DAG health, duration, failures.

**Architecture sketch**

```
[Sources] → [Airflow DAG: Extract] → [Postgres / MinIO]  
                                   ↓  
                [Airflow DAG: Validate + Transform] → [Feature DB / Parquet]
                                   ↓
                   [Airflow DAG: Feature Load] → [Feature Store (Postgres / Redis)]
```

***

### 2. Churn prediction system (MLflow + FastAPI + Postgres)

**Question**  
Design a churn prediction system for 10M users using scikit‑learn / XGBoost with MLflow + FastAPI.

**Ideal answer**  
- **Data & features**:  
  - Raw data in **Postgres / Parquet**; features built via **Airflow‑driven Python jobs**, stored in **Postgres feature store + Redis** for fast lookup.  
- **Training**:  
  - Training script logs to **MLflow**: params, metrics, artifacts, model.  
  - Model versioned in **MLflow Model Registry**.  
- **Inference**:  
  - Batch: **Airflow** runs daily scoring over all users, writes to **Postgres**.  
  - Real‑time: **FastAPI** service on **Kubernetes**, pulls features from **Postgres + Redis**, loads model from **MLflow / local disk**, returns prediction.  
- **Monitoring**:  
  - Track latency, drift, performance with **Prometheus + Grafana** and **batch drift checks**.

**Architecture sketch**

```
[Postgres (raw)] → [Airflow (Feature Engineering)] → [Postgres feature store + Redis]  
                                                                    ↓  
            [Training Script (MLflow)] → [MLflow Tracking + Registry]  
                                                                    ↓
[Batch Scoring (Airflow)] → [Postgres churn scores] ← [FastAPI (K8s)] → [Clients]
```

***

### 3. Near‑real‑time pipeline (Kafka / Postgres / MLflow)

**Question**  
Design a near‑real‑time pipeline for processing millions of distributor transactions per day.

**Ideal answer**  
- **Ingestion**:  
  - Applications send events to **Kafka / Redpanda** topics.  
- **Processing**:  
  - **Kafka consumers** or **Flink / Spark Streaming** compute aggregations / features, write to **Postgres / ClickHouse**.  
- **Feature pipelines**:  
  - Batch features for training run via **Airflow**; streaming features used for real‑time scoring.  
- **Training & serving**:  
  - Model trained on batch features, logged in **MLflow**, served via **FastAPI** with **Redis**‑cached features.

**Architecture sketch**

```
[Apps] → [Kafka] → [Stream Processing (Flink / Spark / Python)] → [Postgres / ClickHouse]  
                                   ↓  
                        [Airflow (Batch Features)] → [MLflow Training]  
                                                                  ↓
                                           [MLflow Registry] → [FastAPI (K8s)]
```

***

### 4. Sudden model accuracy drop (no‑cloud)

**Question**  
Model accuracy dropped from 82% to 65% in production. How do you debug?

**Ideal answer**  
1. **Check logs & metrics**  
   - Use **Prometheus + Grafana** to see latency, error rate, volume.  
2. **Check data drift**  
   - Compare **current feature distributions** with training (via **custom stats** or **MLflow register**).  
3. **Check feature pipeline**  
   - Validate that feature values in serving DB match expectations.  
4. **Check model version**  
   - Confirm **FastAPI** is pointing at correct model (from **MLflow**).  
5. **Rollback procedure**  
   - Use **MLflow model registry** to demote bad model and promote previous version.  
6. **Mitigation**  
   - Add **thresholds** on confidence scores; route low‑confidence cases to human.

**Architecture angle**  
Candidate should separate: **data path → feature path → model version → infra** and use **MLflow** for model versioning.

***

### 5. FastAPI latency spike (Airflow + MLflow + K8s)

**Question**  
Your FastAPI ML service latency increased from 100ms to 800ms. How do you debug?

**Ideal answer**  
- Instrument:  
  - **OpenTelemetry** or **structured logging** to break down latency: deserialization → feature fetch → model prediction → serialization.  
- Bottlenecks:  
  - If feature lookup is slow → add **Redis cache** for frequent features.  
  - If model is slow → consider **ONNX, quantization, smaller model**.  
- Scaling:  
  - Use **Kubernetes HPA** scaling on CPU/memory + **multiple replicas** of the FastAPI pod.  
- Observability:  
  - **Prometheus + Grafana** per‑pod metrics; correlate with **Kafka** or **Airflow** job volume.

**Architecture sketch**

```
[Clients] → [Nginx / API Gateway] → [FastAPI pods (K8s)]  
                                  ↓  
                  [Redis (features)] ← [Postgres feature store]  
                                  ↓
                          [Model (on disk / MLflow artifact)]
```

***

### 6. RAG with open‑source stack (Milvus / Chroma / FastAPI)

**Question**  
Design a RAG pipeline for internal documents using open‑source tools and self‑hosted models.

**Ideal answer**  
- **Ingestion**:  
  - Use **Python scripts / Airflow** to pull PDFs → **MinIO / local storage** → text extraction → **chunking** → embeddings via **sentence‑transformers / BAAI/bge‑small**.  
- **Vector DB**:  
  - Store embeddings in **Milvus / Chroma / Qdrant** (depending on scalability needs).  
- **Query path**:  
  - User question → embed → retrieve top‑k chunks → inject into **self‑hosted LLM / Hugging Face model** prompt → return answer.  
- **Evaluation**:  
  - Build a **QA test set**; compute **accuracy + hallucination rate** manually or via an LLM‑based checker.

**Architecture sketch**

```
[PDFs] → [MinIO / NFS] → [Airflow / Python: Extract → Chunk → Embed] → [Vector DB]
                                                                            ↓
       [User Question] → [FastAPI / LangChain] → [Vector DB → Retrieve] → [LLM / HF API] → Answer
```

***

### 7. Reduce hallucinations in LLM outputs (RAG + prompt constraints)

**Question**  
How do you reduce hallucinations in LLM outputs?

**Ideal answer**  
- **Grounding**:  
  - Use **RAG** to inject retrieved context into the prompt.  
- **Prompt design**:  
  - Explicitly constrain: “If answer not in context, say you don’t know.”  
- **Post‑filtering**:  
  - Check if answer sentences **mention** or **paraphrase** retrieved chunks (citation‑based fact‑checking).  
- **System design**:  
  - Add **fallback**: return “Ask human / check internal docs” when confidence is low.  
- **Evaluation**:  
  - Track **hallucination rate** via **human‑eval** plus automated checks (e.g., using another LLM).

**Panel check**  
Strong answer balances **prompt design + RAG grounding + system‑level fallback**.

***

### 8. CI/CD for ML models (Git + Airflow + MLflow)

**Question**  
How do you implement CI/CD for ML models using open‑source tools?

**Ideal answer**  
- **Code repo**: `git` + **GitHub / GitLab**.  
- **CI**:  
  - On PR/push, run **unit tests, integration tests, data validation** (e.g., **Great Expectations**).  
- **CD pipeline**:  
  - **Airflow / GitHub Actions** triggers:  
    1. Data validation →  
    2. Training →  
    3. Evaluation →  
    4. If metrics pass, register model in **MLflow Model Registry**.  
- **Deployment**:  
  - **FastAPI** service on **Kubernetes** pulls model from **MLflow** (or shared storage).  
  - Use **canary / blue‑green rollout** via Kubernetes manifests.  
- **Rollback**:  
  - **MLflow** allows model version rollback; change the **K8s config** to point to older version.

**Architecture sketch**

```
[Git] → [CI: Tests + Validation] → [Airflow (Train + Eval)] → [MLflow Registry]  
                                                                   ↓
                                   [K8s: FastAPI pods ← MLflow artifact]
```

***

### 9. Data drift & monitoring on‑prem (Prometheus + MLflow)

**Question**  
How do you monitor data drift in on‑prem ML systems?

**Ideal answer**  
- **Offline**:  
  - After each batch run, compute **statistical profiles** (mean, std, PSI, KS) of features vs. training set; store in **Postgres**.  
  - If drift exceeds threshold → **alert** via **Prometheus + AlertManager**.  
- **Online**:  
  - Sample incoming requests; log feature distributions regularly.  
- **Visualization**:  
  - **Grafana** dashboards for drift metrics per model.  
- **Automated response**:  
  - Trigger **retraining** via **Airflow** when drift is detected.

**Panel check**  
Candidate should talk about **specific metrics (PSI, KS)** and **automated triggers**.

***

### 10. Full ML platform on‑prem (no‑cloud)

**Question**  
Design a full ML platform on‑prem (data → feature pipelines → training → serving → monitoring).

**Ideal answer**  
- **Data**:  
  - **Postgres / MySQL / ClickHouse** for structured; **MinIO / HDFS** for semi‑structured.  
- **ETL / Feature pipelines**:  
  - **Airflow / Prefect / Dagster** for orchestration; **Python + pandas / Spark** for transformations.  
- **Experimentation**:  
  - **MLflow** for tracking, parameter sweeps, model versioning.  
- **Serving**:  
  - **FastAPI** on **Kubernetes** for real‑time inference; **batch scoring** via **Airflow**.  
- **Monitoring**:  
  - **Prometheus + Grafana** for infra; **custom drift + metric tables** in Postgres for ML.  

**Architecture sketch**

```
[Sources] → [Airflow (ETL)] → [Postgres / MinIO]  
                                   ↓  
                [Airflow (Feature Pipeline)] → [Feature DB / Redis]  
                                   ↓  
            [Training (MLflow)] → [MLflow Tracking + Registry]  
                                   ↓  
             [FastAPI (K8s)] ← [MLflow artifacts]  
                                   ↓  
                   [Prometheus + Grafana] + [Postgres / ELK]
```

***

### Remaining questions (text‑only pattern)

For the remaining 50 questions, you can use **this reusable pattern** when evaluating:

***

#### For Data Engineering / SQL questions (e.g., dedup, schema changes)

**Ideal answer pattern**  
- Use **SQL window functions / CTEs** for dedup (`ROW_NUMBER()` over partition).  
- Schema changes handled via **idempotent pipelines** (Airflow / Prefect) with **soft schema checks**.  
- **Data quality** via **custom checks / Great Expectations**, alerts on failures.

**Architecture sketch pattern**

```
[Sources] → [Airflow DAG: Extract] → [Postgres / MinIO]  
                                   ↓  
              [Airflow DAG: Validate + Transform] → [Feature DB]
```

***

#### For ML System Design / MLOps (e.g., retraining, monitoring, drift)

**Ideal answer pattern**  
- **Training**: Python + scikit‑learn / XGBoost + **MLflow**.  
- **Features**: stored in **Postgres / Redis**; same logic for training and serving.  
- **Serving**: **FastAPI** on **Kubernetes**.  
- **Monitoring**:  
  - **Prometheus + Grafana** for infra.  
  - **batch drift checks** stored in **Postgres**, with Airflow‑triggered alerts.

**Architecture sketch pattern**

```
[Postgres (raw)] → [Airflow (Features)] → [MLflow Training] → [MLflow Registry]  
                                                                    ↓
                                         [FastAPI (K8s)] ← [MLflow]
```

***

#### For GenAI / RAG questions (vector DB + chunks + evaluation)

**Ideal answer pattern**  
- **Ingestion**: Airflow + Python → extract → chunk → embed → load into **Milvus / Chroma / Qdrant**.  
- **Query**: User → FastAPI/LangChain → embed → vector DB → retrieve → LLM prompt → answer.  
- **Evaluation**:  
  - **human‑eval** + **automated metrics** on hallucination rate / answer correctness.  
- **Cost‑per‑use**:  
  - Cache frequent queries; throttle heavy usage; monitor GPU usage.

**Architecture sketch pattern**

```
[Docs] → [MinIO] → [Airflow: Text → Chunks + Embeds] → [Vector DB]  
                                                        ↓
         [User] → [FastAPI / LangChain] → [Retrieve] → [LLM] → Answer
```

***

### How to Assemble This in a Sheet

For your 60 questions, create a table like:

| Question | Difficulty | Ideal answer bullets | Architecture sketch (text‑only) |
|---------|-----------|----------------------|----------------------------------|
| 1. Batch ETL… | 🟡 | - Uses Airflow for idempotent DAGs… | `[Sources] → [Airflow (ETL)] → [Postgres...]` |
| 2. Churn prediction… | 🟡 | - Features via Airflow… | `[Postgres raw] → [Airflow Features] → [MLflow] → [FastAPI]` |

If you want, I can next:
- Turn this into an **actual Google Sheet / CSV structure** with all 60 rows pre‑filled in this format, ready for your team.

# 20260429 - 1 

Great — below is a **recrafted version of your 60‑question‑style bank** that replaces all Azure‑specific components with **open‑source or on‑prem friendly tools** (e.g., **Airflow, MLflow, Postgres, Redis, Milvus / Chroma, FastAPI, Kubernetes**). You can plug this directly into your interviewer playbook.

***

## Section 1: Data Engineering + SQL (Open‑Source Stack)

1. You receive duplicate transaction records in a pipeline — how do you deduplicate using SQL (window functions)?  
   *Tool‑agnostic: Postgres / MySQL / SQLite.*

2. Design a batch ETL pipeline using **Python + Airflow / Prefect / Dagster** on‑prem for daily ingestion from multiple sources.

3. How would you handle schema changes in production pipelines without breaking downstream jobs?

4. Write a SQL query to get the latest record per user using window functions (CTE + `ROW_NUMBER()`).

5. How do you design **data quality checks** before feeding into ML (e.g., **Great Expectations** or custom validations)?

6. What indexing strategy would you use for large fact tables in a **Postgres / MySQL** setup?

7. How do you optimize a slow query with multiple joins (explain plan, partitioning, indexing)?

8. How would you design an **idempotent** pipeline using Airflow so retries don’t corrupt data?

9. How do you handle late‑arriving data in a batch‑first pipeline (e.g., watermarking via timestamps)?

10. How do you store and version large datasets on‑prem (e.g., **HDFS / NFS / S3‑compatible object store**)?

***

## Section 2: ML System Design (On‑Prem MLOps)

11. Design a **churn prediction system** for 10M users using **scikit‑learn / XGBoost** with **Python + MLflow** for tracking and versioning.

12. How do you build an **offline feature store** on‑prem (e.g., **Postgres / Parquet files + Redis** for fast lookup)?

13. How would you design **batch + real‑time inference** using **FastAPI** on **Kubernetes / Docker**?

14. What metrics would you track in production (latency, p95, error rate, drift, business metrics)?

15. How do you ensure **training‑serving skew** does not happen when using different code paths for training vs inference?

16. How do you design **feature pipelines** so the same features are used in training and serving?

17. How do you handle **model versioning** and **model registry** when you’re using **MLflow**?

18. How would you design a **retraining workflow** that triggers only when drift or performance drops below threshold?

19. How do you **package** a model so it can be deployed consistently across environments (e.g., MLflow + Docker)?

20. How do you design **canary deployment** for ML models using **Kubernetes** (e.g., Istio / Ambassador + FastAPI)?

***

## Section 3: MLOps Using Open‑Source Tools

21. How do you track **experiments** using **MLflow** (parameters, metrics, artifacts, models)?

22. How do you ensure **reproducibility** of ML experiments (code versioning, data versioning, environment)?

23. How do you implement **CI/CD for ML** using **Git + GitHub Actions / GitLab CI + MLflow**?

24. How do you **automate retraining** and **model promotion** from staging → prod when metrics pass?

25. How do you monitor **data drift** on‑prem (e.g., custom stats + dashboards / Prometheus + Grafana)?

26. How do you monitor **model performance drift** in production (e.g., periodic batch scoring + diff with training) ?

27. How do you design **logging** and **tracing** for ML pipelines (e.g., structured logging + ELK / Loki)?

28. How do you **rollback** a bad model version when using **MLflow model registry**?

29. How do you handle **secret management** on‑prem (e.g., **Vault / Kubernetes secrets**) for database credentials, ML models, APIs?

30. How do you design **observability** for ML pipelines (monitoring, alerting, dashboards)?

***

## Section 4: API, Deployment & Scaling (FastAPI + On‑Prem)

31. How do you expose a trained model as an API using **FastAPI**?

32. How do you handle concurrent requests in **FastAPI** (threading, async, batching, gunicorn + uvicorn)?

33. How would you **scale** an ML inference API to thousands of requests per second using **Kubernetes**?

34. How do you implement **rate limiting** and **circuit breaking** on‑prem (e.g., via middleware or reverse proxy)?

35. How do you **batch** inference requests to improve latency and throughput (e.g., BentoML‑style micro‑batching in your own service)?

36. How do you reduce **latency** for large models (ONNX, quantization, model pruning, caching, feature pre‑compute)?

37. How do you design **health checks** and **readiness probes** for a FastAPI ML service on Kubernetes?

38. How do you **secure** ML APIs on‑prem (auth, rate limiting, API gateway, TLS termination)?

39. How do you **monitor** latency, error rate, and CPU/GPU usage for ML services (Prometheus + Grafana)?

40. How do you perform **A/B testing** or **shadow deployment** without external cloud services?

***

## Section 5: GenAI / RAG (Open‑Source Stack)

41. How would you design a **RAG pipeline** for internal documents using **open‑source LLMs or private‑hosted models** (e.g., Hugging Face / custom hosted LLMs)?

42. How do you choose **embedding models** when using open‑source (e.g., `sentence‑transformers`, `BAAI/bge‑small`, etc.)?

43. How do you store and index embeddings using **open‑source vector databases** (e.g., **Milvus, Chroma, Qdrant, Weaviate**)?

44. How do you design **chunking and metadata** for PDFs / docs so retrieval is efficient and relevant?

45. How do you implement **hybrid search** on‑prem (keyword + vector) using **Elasticsearch / OpenSearch** + a vector DB?

46. How do you **reduce hallucinations** in LLM outputs when using your own models (RAG, constrained prompts, fallback to human / docs)?

47. How do you evaluate LLM‑based answers without external cloud APIs (e.g., human‑eval + automated metrics)?

48. How do you design **cost‑aware GenAI usage** when you are hosting your own models (throughput, latency, GPU memory)?

49. How do you design **LLM orchestration** using **LangChain / LlamaIndex** on on‑prem infrastructure?

50. How do you **monitor** and **log** LLM usage patterns, latency, and error types?

***

## Section 6: Deep‑Dive System Design (No‑Cloud)

51. Design a **full ML platform** on‑prem for a B2B company (data → feature pipelines → training → serving → monitoring).

52. Design a **real‑time fraud detection** system using **streaming** (Kafka / Redpanda) and **ML** (FastAPI + MLflow).

53. Design a **recommendation system** for an e‑commerce‑like product on‑prem (collaborative + content‑based, offline + online features).

54. How would you debug a **pipeline** that succeeded in Airflow but the model output is wrong in production?

55. How do you design **lineage** tracking across data → features → model → endpoint (e.g., custom metadata + MLflow + logging)?

56. How do you design **multi‑tenant support** for ML models without cloud services (e.g., separate namespaces in Kubernetes + DB schemas)?

57. How do you secure **model weights** and **config** on‑prem (access control, encryption at rest)?

58. How do you design **backup & disaster recovery** for ML models and data on‑prem?

59. How do you design **feature store** on‑prem that supports both batch and low‑latency online lookups (e.g., **Postgres + Redis**)?

60. When would you **avoid deep learning** and stick with classical ML when running on‑prem hardware?

***

## Mapping to Open‑Source Tool Stack

Where Azure was mentioned before, replace with:

- **Azure ML** → **MLflow + your own training scripts**
- **Azure Data Factory / ADF** → **Airflow / Prefect / Dagster**
- **Azure Blob Storage / Data Lake** → **HDFS / NFS / S3‑compatible object store (MinIO / Ceph)**
- **Azure SQL / Synapse** → **Postgres / MySQL / ClickHouse**
- **Azure Monitor / Application Insights** → **Prometheus + Grafana + ELK / Loki**
- **Azure AI Search / Azure OpenAI** → **RAG with on‑prem vector DB (Milvus / Chroma / Qdrant) + open‑source / self‑hosted LLMs**

***

If you want, I can next:

- Rewrite the earlier **scoring sheet + interviewer feedback form** with this **no‑cloud stack context** baked in.  
- Or generate **ideal answers + architecture diagrams** (text‑only) for this open‑source version of the 60 questions, exactly like the Azure‑based one.

Which one should I do next for you?

# 20260429
Here’s a **panel‑ready “ideal answers + architecture diagrams” guide** for your top scenario questions, tailored to your stack: **Azure + FastAPI + MLflow + RAG**. You can paste this directly into a shared doc / Notion sheet.

***

### 1. Near‑real‑time pipeline (Azure)

**Question**  
Design a near‑real‑time pipeline for processing distributor transactions (millions/day) on Azure.

**Expected answer (ideal)**  
- Ingestion: **Azure Event Hubs / Kafka** for streaming events.  
- Processing: **Azure Stream Analytics / Databricks Structured Streaming** for idempotent, windowed processing.  
- Storage: **Azure Data Lake Gen2** for raw + processed zones; **Azure SQL / Synapse** for aggregated tables.  
- Orchestration: **Azure Data Factory** or **Azure Pipelines** to coordinate batch + streaming.  
- Data quality: **PySpark / Pandas UDFs** for validation, write to **DLQ** when invalid.  
- Monitoring: **Azure Monitor + Log Analytics** for pipeline health.

**Key signals from candidate**  
- Mentions **idempotency, watermarking, DLQ**.  
- Separates raw vs processed zones.  
- Uses **Azure-native tools** in a consistent flow.

**Architecture sketch (you can draw this)**  

```
[Event Hubs] → [Stream Analytics / Databricks] → [Data Lake (raw)]  
                                           ↓
                                   [Data Lake (processed)]
                                           ↓
                               [Azure SQL / Synapse] ← ML
                                           ↓
                                   [MLflow / Azure ML]
```

***

### 2. Churn prediction system (batch + real‑time)

**Question**  
Design a churn prediction system for 10M users with both batch and real‑time inference.

**Ideal answer**  
- Offline training:  
  - Features stored in **Azure Data Lake** and served via **Azure Feature Store** (or custom DB).  
  - Training job in **Azure ML** using **PyTorch / scikit‑learn**; experiments logged in **MLflow**.  
  - Model versioned in **Azure ML Model Registry / MLflow Model Registry**.  
- Batch:  
  - Scheduled **Azure ML pipeline** (daily) scoring all users, results written to **Azure SQL / Cosmos**.  
- Real‑time:  
  - **FastAPI** wrapper around model; **AKS** or **App Services** for serving.  
  - Features fetched from **Azure Cache for Redis / feature store DB**.  
  - Latency SLA enforced via **load testing** and **caching**.  
- Monitoring:  
  - **Azure Monitor + custom dashboards** for drift, latency, error rate.  
  - Automated **retrain trigger** when drift exceeds threshold.

**Architecture sketch**

```
[Azure Data Lake] → [Feature Extraction (Azure ML / Databricks)] → [Feature Store DB]
                                                                       ↓
                    [Training (Azure ML + MLflow)] → [Model Registry] → |
                                                                     ↓
[Frontend] → [FastAPI (AKS)] ← [Redis / Feature Store] ← [Feature Store DB]
                   ↓
           [Azure SQL / Cosmos] → CRM + Dashboard
```

***

### 3. Sudden drop in model accuracy

**Question**  
Model accuracy dropped from 82% to 65% in production. Walk through debugging.

**Ideal answer**  
1. **Check metrics immediately**  
   - Errors per endpoint, latency, input volume.  
   - Use **Azure Application Insights / custom logging**.  
2. **Check data drift**  
   - Compare **current feature distributions** vs training.  
   - Use **Azure ML / custom drift detection** (KS‑test, PSI).  
3. **Check concept drift**  
   - If labels shift (e.g., business rules changed), correlation with prediction confidence drops.  
4. **Check pipeline**  
   - Did feature pipeline break or schema change?  
   - Validate **feature values** at serving time.  
5. **Check model**  
   - Is the **wrong version** deployed?  
   - **Rollback** to previous model using **Azure ML / MLflow**.  
6. **Mitigation**  
   - Put **temporary threshold** on predictions.  
   - Add **canary** for new model.  

**Architecture angle to watch**  
Candidate should separate **data side → model side → infra side** and use **Azure ML / MLflow** for versioning and rollback.

***

### 4. FastAPI latency spike

**Question**  
Your FastAPI ML service latency increased from 100ms to 800ms. How do you debug?

**Ideal answer**  
- Instrument:  
  - Use **OpenTelemetry / Azure Monitor** to break down latency: deserialization, feature lookup, model prediction, serialization.  
- Bottlenecks:  
  - If feature lookup is slow: add **Redis / cache** for frequent features.  
  - If model is slow: **batching, quantization, ONNX** or lighter model.  
- Scaling:  
  - **Horizontal scaling** on **AKS** with proper HPA.  
  - Use **gunicorn + async** workers if IO‑bound.  
- Observability:  
  - Track **p95 latency, error rate, CPU/memory** per pod.  
  - Correlate with **traffic spikes**.

**Architecture note**  
Good candidate will distinguish between **model latency** and **IO latency** and propose **caching + batching**.

***

### 5. Document Q&A with RAG (1M PDFs)

**Question**  
Design a RAG system over 1M PDFs for internal teams.

**Ideal answer**  
- Ingestion:  
  - Use **Azure Functions / Data Factory** to pull PDFs → **Azure Blob Storage**.  
  - Pipeline: **PDF → text extraction → chunking (semantic + metadata)** → embeddings → vector DB.  
- Vector DB:  
  - **Azure AI Search** (as vector index) or **Pinecone / Weaviate** if external.  
- Embeddings:  
  - Use **Azure OpenAI embeddings** or **open‑source** (e.g., `sentence‑transformers`).  
- Query:  
  - User question → **embed** → retrieve top‑k chunks → inject into **LLM prompt** (e.g., **Azure OpenAI**).  
- Reranking:  
  - Optional **cross‑encoder** or **Azure OpenAI reranking**.  
- Evaluation:  
  - Build **QA‑style test set** and compute **answer correctness + hallucination rate**.  
- Cost / perf:  
  - Cache common queries, batch runs, and use **smaller models** when quality is acceptable.

**Architecture sketch**

```
[PDFs] → [Blob Storage] → [Chunking + Embed Pipeline (Azure ML / ADF)]  
                                   ↓
                          [Vector Index (Azure AI Search)]
                                   ↓
[User Question] → [Embed] → [Retrieve] → [LLM Prompt] → [Azure OpenAI] → Answer
                                   ↑
                           [Azure Functions / FastAPI]
```

***

### 6. GenAI / RAG: reduce hallucinations

**Question**  
How do you reduce hallucinations in LLM outputs?

**Ideal answer**  
- **Grounding**:  
  - Always use **retrieval‑augmented context** (RAG) instead of pure generative guess.  
- **Prompt design**:  
  - Explicitly tell the model: “If you don’t know, say so.”  
  - Add constraints: “Respond only from the provided context.”  
- **Post‑filtering**:  
  - Check if answer **references retrieved text** (citation‑based fact verification).  
- **System design**:  
  - Use **Azure OpenAI with content filters** and **custom guardrails**.  
  - Build a **fallback path** (e.g., “Ask a human / check docs”) when confidence is low.  
- **Evaluation**:  
  - Track hallucination rate via **human‑eval + automatic metrics** (e.g., fact‑checking models).

**Panel check**  
Watch for **“just prompt engineering”** vs **system‑level checks + fallback**.

***

### 7. CI/CD for ML models (Azure + MLflow)

**Question**  
How would you implement CI/CD for ML models?

**Ideal answer**  
- **Code repo**: `git` on **Azure DevOps / GitHub**.  
- **CI**:  
  - On PR/push: run **unit tests, integration tests, data validation**.  
- **CD pipeline**:  
  - **Azure DevOps pipeline** triggers:  
    1. Data validation →  
    2. Training in **Azure ML** →  
    3. Evaluation →  
    4. If metrics pass, register model in **Azure ML / MLflow**.  
- **Deployment**:  
  - **Canary** or **blue‑green** rollout via **AKS** or **Azure App Service**.  
  - **Feature flag** to control rollout.  
- **Rollback**:  
  - Use **model versioning** in MLflow / Azure ML to roll back by changing the endpoint config.

**Architecture note**  
Strong candidate will show a **pipeline graph** from Git → Azure ML → Model Registry → AKS, with **gates** for metrics.

***

### 8. Trade‑offs: When not to use deep learning

**Question**  
When would you NOT use deep learning and prefer classical ML?

**Ideal answer**  
- **Small / medium data** → start with **XGBoost / Logistic Regression**.  
- **Low latency / resource‑constrained** → lighter models (e.g., **linear models, decision trees**).  
- **High interpretability** → use **tree‑based models with SHAP / LIME**.  
- **Strong feature engineering** available → classical ML often beats complex DL.  
- **Cost / infra simplicity** → simpler models cheaper to train and serve.

***

### How to Use This for Panel Alignment

For each question in your **60‑question bank**, you can add a row like:

| Question | Difficulty | Expected Answer Highlights | Architecture Diagram Outline |
|---------|-----------|----------------------------|------------------------------|
| Near‑real‑time pipeline (Q21) | 🟡 | Event Hubs → Stream Analytics → DL + SQL → MLflow | Event Hubs → Stream → DL → SQL → Azure ML / MLflow |
| Churn prediction (Q25) | 🟡 | Batch + real‑time, feature store, Azure ML, FastAPI on AKS | DL → Feature Store → Azure ML → FastAPI → SQL |

If you like, I can next:
- Generate a **Google Sheet template** with this table pre‑filled for all 60 questions, ready for your team to paste.
