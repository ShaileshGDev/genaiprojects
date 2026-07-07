# Large Language Models:     

## 1. Fundamentals of the LLM Landscape
* **Core Concepts:** Visual perception, natural language understanding, tokenization, embeddings, and text generation mechanisms.
* **Architecture:** The canonical end-to-end architecture connecting every layer.
* **Vector Databases & Similarity Search:** Understanding why vector databases exist, how embeddings represent meaning as geometry, and how similarity search powers retrieval at scale.
* **Prompt Engineering Best Practices:** Structuring prompts for consistent, grounded outputs and identifying common failure modes.
* **Building Custom LLM Applications:** Exploring three distinct paths: training from scratch, fine-tuning foundation models, and in-context learning (with associated tradeoffs).
* **Retrieval-Augmented Generation (RAG):** Connecting retrieval to generation, identifying breakdown points, and requirements for production-ready implementations.

## 2. Challenges in Enterprise Adoption
* **Adoption Reality:** Overcoming the "PoC to Production" stall; managing pressures of cost, accuracy, latency, and infrastructure constraints.
* **Technology Challenges:** Context-window limits, dataset quality, PII handling, fine-tuning costs (PEFT approaches), and compounding inference costs.
* **Business and Risk Challenges:** Compliance exposure, legal risks, misaligned customer expectations, and organizational culture gaps in KPI alignment.
* **Human Behavior Challenges:** Addressing fear of AI, resistance to change, biased prompting, and subjectivity in feedback/evaluation.
* **Prompting Challenges:** Addressing sensitivity, fatigue, overengineering, jailbreaking/injection risks, and hallucinations from lack of grounding.
* **Best Practices:** Implementing guardrails, defensive UX design, LLM caching for cost control, structured feedback, and fairness/explainability frameworks.

## 3. Challenges in RAG Applications
* **Privacy, Security, and Compliance:** PII-safe ingestion (anonymization, differential privacy), authenticated access, and data retention logging.
* **Multimodal and Multilingual RAG:** Vision models (CLIP, BLIP-2, Gemini), audio pipelines (Whisper, noise-tolerant search), and cross-lingual retrieval (LaBSE, mBERT, XLM-R).
* **Advanced Architectures:** DSPy for prompt compilation, KG-RAG (graph-grounded answers), and self-improving retrieval via feedback loops and ranking callbacks.
* **Retrieval Layer and Index Optimization:** Comparing FAISS, Qdrant, Weaviate, and Pinecone; index types (Flat, HNSW, IVF, PQ); hybrid search (BM25 + dense) and re-rankers (BGE, ColBERT).
* **Evaluation and Metrics:** Retrieval metrics (Precision@K, Recall@K, Hit Rate@K) and generation metrics (BLEU, ROUGE, LLM-as-a-Judge, TruLens, Phoenix, EvalGen).
* **Latency, Cost, and Scalability:** Bottleneck identification, caching strategies (embedding, prompt, disk layers), and cost controls (batch queries, quantized models).

## 4. Transformer Architecture and Attention Mechanisms
* **Introduction to LLMs:** Strengths vs. weaknesses; discriminative vs. generative AI; predictive vs. generative differences.
* **Transformer Architecture:** Deep dive into tokenization, embeddings, positional encoding, and the attention mechanism.
* **Embeddings and Similarity:** Vector proximity, meaning in vector space, and its importance for retrieval/reasoning.
* **Attention Mechanism:** Keys, Queries, and Values in self-attention; focusing mechanisms during token generation.
* **Softmax and Probabilities:** From raw attention scores to probability distributions for next-token prediction.
* **Training and Fine-Tuning:** Adapting pre-trained models, understanding parameter changes, and identifying overfitting.
* **Search and Retrieval:** Building semantic search engines and connecting them to generation for grounded answers.
* **Hands-On Exercises:** Sentence Transformers, semantic search, attention scoring, and manual implementation of attention mechanisms.

## 5. Vector Databases
* **Overview and Rationale:** Differences from traditional search and the role in grounding LLM applications.
* **Search Types:** Vector, text, and hybrid search use cases.
* **Indexing Techniques:** Tradeoffs between speed, memory, and recall (PQ, LSH, HNSW).
* **Retrieval Techniques:** Cosine similarity, nearest neighbor search, and scaling challenges.
* **Advanced RAG Techniques:** Chunking, filtering, query rewriting, auto-cut, and re-ranking.
* **Embedding and Model Selection:** Domain-specific embeddings, fine-tuned models, and compression (scalar, product, binary, and matryoshka).
* **Adaptive Retrieval and Multi-Tenancy:** Multi-phase rescoring, tenant isolation, and resource allocation.
* **Production Challenges:** Scaling, reliability, and cost optimization for real query volumes.
* **Hands-On Exercises:** Vector search, hybrid search, generative search, Weaviate Query Agent, multi-tenancy, and semantic caching.

## 6. Mastering LangChain
* **Introduction:** Abstracting RAG challenges and the purpose of the framework.
* **Core Components:** LLMs, chat models, prompt templates, example selectors, document loaders, and transformers.
* **Output Parsers:** Structured data extraction, consistent formatting, and error handling.
* **Retrieval and Vector Stores:** Embedding, vectorization, metadata filtering, and parent document retrieval.
* **Chains:** Sequential logic, pre/post-LLM steps, and composable workflows.
* **Tool Use and Memory:** API integration, conversation history, state management, and context persistence for agents.
* **Callbacks and Observability:** Event hooks, monitoring, logging, and custom success/failure actions.
* **LCEL (LangChain Expression Language):** Piping runnables, parallel branches, and modular composition.
* **LangGraph and Agents:** Graph-based workflows, dynamic control flows, and non-linear reasoning.

## 7. Fine-Tuning LLMs
* **Core Concepts:** Transfer learning, full fine-tuning vs. LoRA/QLoRA, parameter-efficient tuning (PEFT), and quantization.
* **Key Considerations:** Data quality as a primary lever, overfitting risks, and RAG vs. Fine-tuning decision-making.
* **Hands-On Exercises:** Instruction fine-tuning, deploying/evaluating LLaMA2-7B 4-bit quantized models, and Azure AI Studio deployment projects.

## 8. Evaluation of LLMs
* **Need for Evaluation:** Reliability, accuracy, safety, business alignment, and ethical accountability.
* **Challenges in Evaluation:** Hallucinations, prompt sensitivity, weak context handling, and navigating accuracy vs. fluency tradeoffs.
* **Benchmarking Approaches:** MMLU (multitask), HELM (holistic), BBH, and HotpotQA (reasoning).
* **Text Quality Metrics:** BLEU, ROUGE, BERTScore, METEOR, and perplexity.
* **RAG-Specific Evaluation (RAGAs):** Faithfulness, answer relevance, context precision, and context recall.
* **Open-Ended Output Evaluation (G-Eval):** Fluency, faithfulness, and claim-level scoring.
* **Additional Benchmarks:** GLUE (NLU), TriviaQA, RealToxicityPrompts (safety), MRR, MAP, and ROSCOE (reasoning).

## 9. MCP (Model Context Protocol)
* **Origins & Motivation:** Addressing fragmented integrations with a "USB-C for AI" unified interface.
* **Protocol Structure:** Client–server handshake, resources, tools, prompts, and JSON-RPC transport.
* **Context Exposure:** Schema-driven discoverability, governance, and controlled access.
* **Agentic Integration:** Connecting MCP to reflection, planning, tool-use, and multi-agent coordination.
* **Hands-On Labs:** Streamlit MCP client setup, tool registration, workflow automation, and trace logging.

## 10. Build A Multi-agent LLM Application
* **Security Focus:**
    * Sensitive Data Protection (Identification and obfuscation).
    * Secure GenAI Agents (Defense-in-Depth pattern).
    * Red Teaming (Audit and verification).
    * Practical exercise: Hardening security in a cloud environment.
* **Project Tracks:**
    * Conversational Workflow Orchestration.
    * Knowledge-Enhanced Agent (Search/API integration).
    * Document-Aware Action Agent.
    * Orchestrated Collaboration (MCP-based).
* **Resources Provided:** Comprehensive datasets, step-by-step guides, and ready-to-use code templates.
* **Learner Options:** Virtual Assistant, Content Generation (Marketing Co-pilot), Legal & Compliance Assistant, Content Personalizer, or MCP Chatbot.
* **Outcome:** A production-ready multi-agent application demonstrating mastery of reasoning, retrieval, tool use, and protocol-driven interoperability.

# Agentic AI Course  

## 1. Introduction to Agentic AI
* **Foundations of Agentic AI:** Transitioning from next-token prediction to reasoning models; understanding the limitations of classic LLMs versus reasoning LLMs; and the core pillars of agentic systems: reasoning, context, and autonomy.
* **Understanding LLMs:** Exploration of context windows, session memory, and long-term memory (vector databases, knowledge graphs, summaries); analysis of data sources including pre-training, fine-tuning, and in-context learning.
* **Retrieval-Augmented Generation (RAG):** Analysis of naïve RAG workflows and common challenges; RAG as a context enhancement strategy; and data preparation/structuring for effective RAG pipelines.
* **Agentic AI Components:** Cognition (reasoning, planning, self-reflection), knowledge representation, and autonomy through tool use, action execution, and monitoring.
* **Agentic Design Patterns:** Planning, tool use, and reflection loops; Agentic RAG, routers, and iterative loops; and sequential, parallel, and hierarchical workflows.
* **Architectures for Agents:** Single-agent vs. multi-agent systems; human-in-the-loop strategies; and hybrid reasoning pipelines and decision graphs.
* **Advanced Context Techniques:** Session summaries, hybrid memory systems, Model Context Protocol (MCP), and scalable context management.
* **Observability, Safety & Governance:** Guardrails, explainability, monitoring and evaluation, ethical alignment, and compliance strategies.
* **Hands-On Exercises:** Practical implementations of reasoning workflows, memory systems, RAG pipelines, and safe agent design.

## 2. Transformers & Attention Mechanism
* **Introduction to LLMs:** Overview of the strengths and weaknesses of large language models.
* **Discriminative vs. Generative AI:** Contrasting predictive models with generative models.
* **Transformer Architecture:** Deep dive into tokenization, embeddings, positional encoding, and the attention mechanism.
* **Embeddings and Similarity:** Representing words as vectors and measuring spatial closeness.
* **Attention Mechanism:** Understanding Keys, Queries, and Values in self-attention.
* **Softmax and Probabilities:** Converting scores into probabilities for next-word prediction.
* **Training and Fine-Tuning:** Adapting models with curated data for specific tasks.
* **Search and Retrieval:** Building semantic search engines with embeddings.
* **Retrieval Augmented Generation (RAG):** Combining retrieval with generation for grounded answers.
* **Hands-On Exercises:** Sentence Transformers, semantic search, attention scoring, and attention mechanism implementation.

## 3. Mastering LangChain
* **Introduction to LangChain:** Purpose, scope, building LLM-powered applications, and common RAG implementation challenges.
* **Core Components:** LLMs and chat models, prompt templates, example selectors, document loaders, and transformers for preprocessing.
* **Output Parsers:** Extracting structured data, enforcing consistent formats, and handling validation failures.
* **Retrieval:** Embedding/vectorization strategies, retrievers, metadata filtering, and parent document retrieval.
* **Vector Stores:** Efficient embedding storage, scalable similarity search, and large dataset optimization.
* **Chains:** Sequential prompt logic, pre- and post-LLM processing, and end-to-end integration of retrieval and tools.
* **Tool Use:** Connecting APIs and external systems, managing tool outputs, retries, and error handling.
* **LangChain Expression Language (LCEL):** Building modular workflows using runnables, piping operations, and parallel branches.
* **Hands-On Exercises:** Constructing retrieval chains, parsing structured outputs, and building production-ready workflows.

## 4. Vector Databases and Agentic RAG
* **Vector Database Fundamentals:** Embeddings, vector storage, ANN vs. kNN search, and modern architectures/data models.
* **Hybrid Retrieval Design:** Combining dense and sparse vectors, metadata filters, payload indexing, and full-text tokenization.
* **Advanced Techniques:** Maximal Marginal Relevance (MMR) for diversity, Discovery APIs, and HNSW index health monitoring.
* **Agentic RAG Concepts:** Using AI-native vector databases as long-term memory, multi-step retrieval with reasoning loops, and hallucination mitigation.
* **Semantic Caching:** Caching semantically similar queries, TTL and invalidation policies, and cost/latency optimization.
* **Hands-On Exercises:** AI-native database exploration, hybrid search implementation, MMR re-ranking, and agentic RAG orchestration.

## 5. Context Engineering
* **Complex Agentic Workflows:** Designing system/user prompts, integrating memory and web search, and implementing critique/refinement loops.
* **Deterministic Chains and Control Flows:** Building sequential pipelines with structured task execution and predictable control logic.
* **Agent Reliability and Dynamic Decisions:** Utilizing router agents and conditional flows to balance autonomy with control.
* **LangGraph Fundamentals:** Nodes, edges, state management, and condition-based execution for auditable workflows.
* **Tool Integration:** Node-based tool calls for APIs/databases and shared state updates.
* **Agentic Design Patterns:** Reflection for self-critique, tool use for actions, and planning for task decomposition.
* **Multi-Agent Collaboration:** Parallel, sequential, loop, and router flows with error handling and human supervision.
* **Multi-Agent Architectures:** Hierarchical delegation systems, approval nodes, shared memory, and resource coordination.
* **Hands-On Experience:** Practical implementation of advanced context engineering across agentic and multi-agent systems.

## 6. Agentic Design Patterns
* **Why Agentic Patterns Matter:** Moving from single-pass prompting to iterative, goal-oriented reasoning loops.
* **Reflection Pattern:** Enabling agents to evaluate, critique, and refine outputs via structured feedback cycles.
* **Planning Pattern:** Stepwise reasoning flows to decompose complex goals and manage task dependencies.
* **Tool Use Pattern:** Connecting models to external systems to extend problem-solving capabilities beyond internal knowledge.
* **Multi-Agent Collaboration Pattern:** Coordinating specialized agents with defined roles for collaborative problem-solving.
* **Pattern Trade-Offs:** Balancing autonomy vs. control, flexibility vs. stability, and creativity vs. reliability.
* **Pattern Composition:** Integrating reflection, planning, and tool use into hybrid workflows.
* **Hands-On Labs:** Implementing and combining individual patterns into production-ready workflows for real-world tasks.

## 7. Agentic AI Protocols
* **Multi-Agent Coordination:** Addressing collaboration challenges, message routing, and task orchestration.
* **Need for Agentic Protocols:** Establishing discovery, negotiation, structured task management, and secure cooperation.
* **Model Context Protocol (MCP):** Client–server architecture for tool integration; standardized access to data, prompts, tools, and resources.
* **MCP Architecture:** Roles of hosts, clients, and servers; message exchange formats; and connecting applications/IDEs.
* **Agent-to-Agent Protocol (A2A):** Task-oriented communication, capability discovery using Agent Cards, and structured message formats.
* **Agent Communication Protocol (ACP):** Open ecosystem for cross-agent interaction, routing, and discovery.
* **MCP vs ACP vs A2A:** Comparative analysis of scope, complexity, and message types for different requirements.
* **Hands-On Exercise – MCP Client with Streamlit:** Environment setup, dependency installation, server connection, tool discovery, and workflow automation.

## 8. MCP (Deep Dive)
* **Origins & Motivation:** Solving fragmented integrations with a unified, interoperable interface (the “USB-C for AI”).
* **Protocol Structure:** Client–server handshake, JSON-RPC transport, and schema-driven messages for resources, tools, and prompts.
* **Context Exposure:** Surfacing tools, data, and metadata via consistent schemas for governance and discoverability.
* **Agentic Integration:** Connecting MCP endpoints to reflection, planning, tool-use, and multi-agent patterns.
* **Hands-On Labs:** Streamlit client setup, tool registration, workflow automation, and trace logging for monitoring.

## 9. Evaluation of Agents
* **Need for Evaluation:** Reliability, accuracy, safety, business/ethical alignment, and user trust.
* **Challenges in Evaluation:** Hallucinations, prompt sensitivity, weak context grounding, and subjectivity.
* **Benchmarking Approaches:** MMLU (multitask), HELM (robustness/fairness), BBH, and HotpotQA (reasoning).
* **Text Quality Metrics:** BLEU (precision), ROUGE (recall), and BERTScore (semantic similarity).
* **RAG Evaluation (RAGAS):** Faithfulness, answer relevance, context precision, and context recall.
* **G-Eval:** Fluency, faithfulness, and claim-level scoring of open-ended outputs.
* **Additional Benchmarks:** GLUE (NLU), TriviaQA (QA), RealToxicityPrompts (safety), and Blended Skill Talk (dialogue).
* **Other Metrics:** Perplexity (confidence), METEOR (alignment), MRR/MAP (ranking), and ROSCOE (reasoning).
* **Hands-On Exercises:** Applying RAGAS to RAG pipelines, comparing text metrics, and evaluating agent outputs using G-Eval.

## 10. Build a Multi-Agent LLM Application
* **Project Tracks:**
    * **Conversational Workflow Orchestration:** Multi-turn assistant with specialized agents.
    * **Knowledge-Enhanced Agent:** Search and API integration for grounding and real-time access.
    * **Document-Aware Action Agent:** Reasoning over documents to trigger external tools/services.
    * **Orchestrated Collaboration (MCP):** Coordinated multi-agent systems using Model Context Protocol.
* **Attendee Resources:**
    * Comprehensive Datasets
    * Step-by-Step Implementation Guides
    * Ready-to-Use Code Templates
* **Implementation Options:**
    * Virtual Assistant
    * Content Generation (Marketing Co-pilot)
    * Conversational Agent (Legal & Compliance Assistant)
    * Content Personalizer
    * MCP Chatbot (Calendar, CRM, and API integrations)
* **Outcome:** A production-ready multi-agent application demonstrating mastery of reasoning, retrieval, tool use, and protocol-driven interoperability.

Picked topics from @ datasciencedojo
