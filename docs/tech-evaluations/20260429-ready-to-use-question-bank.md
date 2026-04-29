Here’s a **ready-to-use question bank (60 questions)** tailored to your stack: **Azure + FastAPI + MLflow + RAG**, grouped by difficulty and mapped to real production scenarios.

***

## 🟢 Level 1: Foundational but Practical (20 questions)

### Data Engineering + SQL
1. You receive duplicate transaction records in a pipeline — how do you deduplicate using SQL (window functions)?
2. How would you design an ETL pipeline in Azure for daily ingestion from multiple sources?
3. How do you handle schema changes in production pipelines?
4. Write a SQL query to get the latest record per user using window functions.
5. How would you validate data quality before feeding into ML?
6. What indexing strategy would you use for large fact tables?
7. How do you optimize a slow query with multiple joins?

### ML Basics (Applied)
8. You have missing values in critical features — what’s your strategy?
9. How do you choose between classification vs ranking model?
10. What metrics would you use for imbalanced datasets?
11. How do you perform feature selection in real-world datasets?
12. Explain bias-variance tradeoff with a real production example.

### FastAPI / Deployment
13. How do you expose a trained model as an API using FastAPI?
14. How do you handle concurrent requests in FastAPI?
15. What are common bottlenecks in ML inference APIs?

### Azure Basics
16. How would you deploy an ML model using Azure ML?
17. Difference between Azure Blob Storage vs Data Lake?
18. How would you schedule pipelines in Azure?

### MLflow
19. How do you track experiments in MLflow?
20. What is model registry and why is it important?

***

## 🟡 Level 2: Strong Practical Engineer (25 questions)

### Data Pipelines
21. Design a near real-time pipeline using Azure (Event Hub + Stream processing).
22. How do you handle late-arriving data in streaming pipelines?
23. How do you ensure idempotency in pipelines?
24. How would you design a feature pipeline shared across models?

### ML System Design
25. Design a recommendation system for an e-commerce platform.
26. How do you ensure training-serving skew does not happen?
27. How do you version features and datasets?
28. Batch vs real-time inference — when to use what?

### MLOps
29. How would you implement CI/CD for ML models?
30. How do you rollback a bad model deployment?
31. What metrics do you monitor in production ML systems?
32. How do you detect data drift vs concept drift?
33. How do you automate retraining pipelines?

### FastAPI + Scaling
34. Your API latency increased from 100ms to 800ms — how do you debug?
35. How would you scale inference to 1M requests/day?
36. Would you use async or sync endpoints? Why?

### MLflow Advanced
37. How do you manage multiple model versions across environments?
38. How do you ensure reproducibility using MLflow?

### Azure Advanced
39. Design ML architecture using Azure ML + AKS.
40. How do you manage secrets securely in Azure?
41. How do you optimize cost in Azure ML workloads?

### GenAI / RAG Basics
42. What is RAG and why is it better than fine-tuning in some cases?
43. How do embeddings work in simple terms?
44. How do you choose embedding models?
45. What is chunking and why is it important?

***

## 🔴 Level 3: Senior / Day-1 Ready (15 questions)

### End-to-End System Design
46. Design a full ML platform (data → training → deployment → monitoring) on Azure.
47. Build a churn prediction system with both batch and real-time inference.
48. Design a fraud detection system with low-latency requirements.

### Failure & Debugging (Critical)
49. Model accuracy dropped suddenly — walk me through debugging.
50. Predictions are inconsistent between training and API — why?
51. Pipeline succeeded but model output is wrong — how do you trace?

### Advanced MLOps
52. How do you implement canary deployment for ML models?
53. How do you monitor feature drift in real-time?
54. How do you design lineage tracking across data + models?

### RAG / GenAI Advanced
55. Design a scalable RAG system for 10M documents.
56. How do you reduce hallucination in LLM outputs?
57. How do you evaluate LLM responses quantitatively?
58. How do you optimize cost of LLM APIs in production?
59. How do you design hybrid search (keyword + vector)?

### Architecture Trade-offs
60. When would you NOT use deep learning and prefer classical ML?

***

## 💡 How to Use This Bank

- Pick:
  - 2–3 from Level 1 (warm-up)
  - 3–4 from Level 2 (core evaluation)
  - 2–3 from Level 3 (depth filter)

- Always follow with:
  - “What would fail in your design?”
  - “How will you debug this in production?”
  - “What trade-offs did you make?”

***

If you want, I can next:
- Turn this into a **scoring sheet + interviewer feedback form**
- Or create **ideal answers + expected architecture diagrams for each question** (this is extremely useful for panel alignment)
