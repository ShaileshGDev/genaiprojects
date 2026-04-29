Got it — you don’t want the questions directly, you want **high-quality reusable prompts** that will generate *strong, scenario-based, hands-on interview questions* aligned to your JD.

Here are **structured prompt templates** you can reuse across roles or tools (LLMs, internal tooling, etc.).

***

### 1. Master Prompt (Full Coverage)

Use this when you want a complete interview set.

```
You are an expert ML + Data Engineering interviewer hiring for a production-focused role.

Generate scenario-based interview questions based on the job description below.

Focus on evaluating:
- Real-world hands-on experience
- Ability to design production systems
- Trade-offs, decision-making, and debugging skills
- Readiness to deliver from Day 1

For each question:
1. Start with a realistic business scenario
2. Ask what the candidate will do
3. Ask for architecture/design approach
4. Ask what tools/frameworks they will choose and why
5. Include follow-up probing questions (failure cases, scaling, edge cases)

Ensure coverage of:
- Data pipelines (batch + streaming)
- Feature engineering and data quality
- Classical ML (not just theory, but production usage)
- Model deployment (APIs, Docker, CI/CD)
- MLOps (monitoring, retraining, drift handling)
- SQL and data modeling
- System design for ML systems
- GenAI (RAG, LLMs, embeddings, evaluation) where applicable

Questions should:
- Be scenario-driven, not theoretical
- Require architecture-level thinking
- Expose depth vs surface knowledge
- Include ambiguity (to test thinking)

Job Description:
[PASTE JD HERE]

Output format:
- Question
- What a strong candidate should cover (key points)
- Red flags to watch for
```

***

### 2. Data Engineering + Pipelines Focus

```
Generate scenario-based interview questions to evaluate a candidate’s ability to build and maintain scalable data pipelines.

Focus areas:
- Batch vs near real-time pipelines
- Data quality, validation, and schema evolution
- Handling large structured and semi-structured data
- Orchestration tools (Airflow, etc.)
- Performance tuning and failure recovery

Each question must:
- Start with a real production scenario (e.g., broken pipeline, late data, scaling issue)
- Ask candidate to design solution
- Ask how they ensure reliability and observability

Also include:
- SQL-heavy scenarios (CTEs, window functions, optimization)

Output:
- Scenario
- Questions
- Expected strong answer signals
- Weak signals
```

***

### 3. ML System Design (Production ML)

```
Create scenario-based interview questions for evaluating production ML system design skills.

Focus on:
- Feature pipelines and feature stores
- Model training and versioning
- Inference (batch vs real-time)
- Monitoring (data drift, concept drift)
- Retraining strategies

Each scenario should:
- Represent a real business problem (fraud detection, recommendation, churn, etc.)
- Ask candidate to design end-to-end system
- Include constraints (latency, scale, cost)

Force candidate to answer:
- What ML model and why
- How features are built and served
- How system is deployed
- How failures are handled

Output:
- Scenario
- Structured questions
- Ideal approach
```

***

### 4. MLOps + Reliability

```
Generate interview questions focused on MLOps and production reliability.

Test:
- MLflow / model registry usage
- CI/CD for ML
- Experiment tracking
- Reproducibility
- Monitoring and alerting
- Rollbacks and versioning

Use scenarios like:
- Model performance suddenly drops
- Data drift detected
- Deployment breaks in production

Ask:
- Diagnosis approach
- Tools used
- Automation strategies

Output:
- Scenario
- Questions
- What good answers include
```

***

### 5. GenAI / LLM / RAG (Modern Must-Have)

```
Generate scenario-based interview questions to evaluate practical GenAI skills.

Focus on:
- RAG pipelines
- Embeddings and vector databases
- Prompt engineering
- LLM orchestration frameworks
- Cost vs performance trade-offs
- Evaluation of LLM outputs

Scenarios must include:
- Building document Q&A system
- Improving hallucination issues
- Scaling LLM usage in production
- Latency/cost optimization

Ask:
- Architecture design
- Choice of models (open vs closed)
- Retrieval strategy
- Evaluation approach

Output:
- Scenario
- Questions
- Strong answer indicators
```

***

### 6. API + Deployment + Infra

```
Create interview questions to evaluate ML API deployment skills.

Focus:
- FastAPI / inference services
- Dockerization
- CI/CD pipelines
- Scaling (Kubernetes, autoscaling)
- Latency optimization

Scenario examples:
- High latency inference API
- Scaling from 100 to 1M requests/day
- Model version rollout

Ask:
- API design
- Infra decisions
- Monitoring setup

Output:
- Scenario
- Questions
- Expected answers
```

***

### 7. Candidate Depth Filter (Very Important)

Use this to **separate real engineers from resume-driven candidates**.

```
For each generated question, add deep probing follow-ups that test:

- “What broke in your last similar system?”
- “What would fail first in your design?”
- “How would you debug this in production?”
- “What metrics would you track?”
- “What trade-offs did you make and why?”

Ensure questions cannot be answered with textbook knowledge.
```

***

### Example (Mini Output Generated by These Prompts)

Scenario:
Your real-time recommendation pipeline is producing stale results during peak hours.

Questions:
- How will you debug the issue?
- What architecture changes would you propose?
- Would you use batch + streaming hybrid? Why?
- How will you ensure freshness vs cost trade-off?

Strong signals:
- Mentions feature freshness, caching layers, streaming (Kafka/Flink)
- Talks about monitoring lag and backpressure
- Suggests hybrid architecture

Weak signals:
- Only talks about retraining model
- No system-level thinking

***

If you want next step, I can:
- Convert these into a **ready-to-use interviewer playbook**
- Or generate **actual 40–50 high-quality questions using this** tailored to your stack (Azure + ML + GenAI)

Which one do you want?
