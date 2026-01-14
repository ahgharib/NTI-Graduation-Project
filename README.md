# 📘 Study Buddy — Intelligent Learning Companion

**Study Buddy** is an AI‑powered, interactive learning platform designed to help learners build structured roadmaps, deeply understand concepts, track progress, and receive personalized feedback throughout their learning journey. The system acts as a virtual AI tutor that adapts dynamically to the learner’s goals, strengths, and weaknesses.

This project was developed as part of the **NTI Graduation Project** and demonstrates the integration of modern AI techniques such as **Retrieval‑Augmented Generation (RAG)**, **multi‑agent orchestration**, **OCR**, and **multimodal content generation** within a real, usable application.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Key Features](#-key-features)
* [System Workflow](#-system-workflow)
* [Technologies Used](#-technologies-used)
* [Repository Structure](#-repository-structure)
* [Installation & Setup](#-installation--setup)
* [Usage Guide](#-usage-guide)
* [System Architecture & Workflow](#-system-architecture--workflow)
* [Learned Lessons](#-learned-lessons)
* [Media & Demo](#-media--demo)
* [License](#-license)

---

## 🎯 Project Overview

Study Buddy is designed to solve a common problem in self‑learning: **lack of structure, feedback, and personalization**. Instead of static courses or generic quizzes, the system dynamically builds a learning experience that evolves with the user.

By combining large language models, vector databases, and intelligent agents, Study Buddy provides:

* Structured learning paths
* Concept‑level assessment
* Adaptive quizzes
* Continuous progress tracking
* Personalized feedback and memory

---

## 🚀 Key Features

### Personalized Learning Roadmaps

* Generate **complete learning roadmaps** based on user preferences, goals, and background.
* Roadmaps are divided into **clear milestones** and sub‑topics.
* Each milestone is designed to ensure gradual and measurable progress.

### Dynamic Roadmap Adjustment

* Users can **modify any milestone** at any time.
* The system automatically re‑balances the roadmap to keep learning coherent.
* Supports changing timelines, topic depth, or learning focus.

### Intelligent Quiz Generation

* Generate quizzes with multiple formats:

  * **Multiple Choice Questions (MCQ)**
  * **Coding questions**
  * **Article / explanation‑based questions**
* Quizzes are generated based on:

  * Roadmap milestones
  * User performance history
  * Uploaded documents

### Advanced Feedback & Marking System

* Grading focuses on **conceptual understanding**, not memorization.
* Partial credit is awarded when reasoning shows understanding.
* Each question includes:

  * Given mark
  * Explanation of correctness
  * Reasoning when full marks are not achieved

### Strength & Weakness Tracking

* Automatically stores:

  * User strong points
  * User weak points
* This data is reused to:

  * Adjust future quizzes
  * Emphasize weak concepts
  * Recommend revision topics

### Milestone Progress Evaluation

* After each milestone:

  * A quiz is generated automatically
  * Understanding is validated before moving forward
* Prevents knowledge gaps from accumulating.

### Concept Explanation & Summarization

* Users can ask for explanations or summaries of any concept.
* The system:

  * Searches the internet
  * Retrieves relevant content
  * Generates concise and clear explanations

### 🎥 Educational Video Generation

* Automatically generates **video explanations** for concepts.
* Useful for visual learners and revision sessions.

### 📝 Handwritten Notes Generation

* Generates **handwritten‑style notes** for key topics.
* Notes can be saved and reused as quick revision material.

### Document Upload & Custom Learning

* Users can upload:

  * Text‑based documents
  * Scanned images or PDFs
* OCR is applied to extract content.
* Uploaded content is used to:

  * Generate quizzes
  * Tailor explanations
  * Build roadmap content

### 🖥️ Interactive Application Interface

* User‑friendly and intuitive interface.
* Users can explore all features from a single application.
* Real‑time feedback and visual progress tracking.

---

## 🔁 System Workflow

1. User defines learning goal or uploads documents.
2. System builds a personalized roadmap.
3. Each milestone triggers learning content and quizzes.
4. Performance is analyzed and stored.
5. Future actions adapt based on user memory and progress.

This workflow is orchestrated using **LangChain** and **LangGraph** to ensure modular and maintainable AI pipelines.

---

## 🧰 Technologies Used

### AI & NLP

* **LangChain** – LLM integration and chaining
* **LangGraph** – Agent workflow orchestration
* **RAG (Retrieval‑Augmented Generation)**
* **Mini‑RAG Memory System** for personalization
* **Diffusion Models** for visual content generation
* **Text‑to‑Speech (TTS)** models

### Data & Search

* **FAISS Vector Database** for semantic retrieval
* **OCR** for scanned and image‑based documents

### Backend & Frontend

* **FastAPI** – backend APIs
* **Streamlit** – interactive UI
* **Ngrok** – secure tunneling for deployment

---

## 📁 Repository Structure

```
NTI-Graduation-Project/
│
├── agents.py                # Core AI agents
├── app.py                   # FastAPI application
├── chat_graph.py            # LangGraph workflow
├── orchestrator.py          # Agent orchestration logic
├── quiz_agent.py            # Quiz generation and grading
├── search_agent.py          # Internet & document search
├── summarizer_agent.py      # Concept summarization
├── generatingnotes.py       # Handwritten notes generation
├── roadmap.json             # Roadmap structure
├── schemas.py               # Data schemas
├── tools.py                 # Shared tools and utilities
└── config.py                # Configuration settings
```

---

## 🛠 Installation & Setup

### Prerequisites

* Python 3.9+
* Virtual environment (recommended)
* API keys for LLM services

### Steps

```bash
# Clone repository
git clone https://github.com/ahgharib/NTI-Graduation-Project.git
cd NTI-Graduation-Project

# Install dependencies
pip install -r requirements.txt

# Run backend
uvicorn app:app --reload

# Run frontend
streamlit run main.py
```

---

## 📘 Usage Guide

* **Create Roadmap**: Enter your learning goal and preferences.
* **Follow Milestones**: Complete topics step‑by‑step.
* **Take Quizzes**: Validate understanding after each milestone.
* **Review Feedback**: Learn from detailed explanations.
* **Upload Documents**: Customize learning using your own materials.

---

## 🏗 System Architecture & Workflow

The system is built as a **Multi-Agent State Machine** using **LangGraph**. This allows for non-linear workflows where the AI can "loop back" to correct errors or ask for clarification.

### 1. The Orchestrator
The `orchestrator.py` acts as the central router. It analyzes the user's intent from the `chat_graph.py` state and determines which specialized agent to invoke. It manages the conversation history and ensures the "Global State" (user progress, memory, and roadmap) is passed correctly between nodes.

### 2. ReAct Agent Framework
We utilize the **ReAct (Reasoning and Acting)** pattern across our agentic layers. 
* **Reasoning:** Agents first generate a thought process about the task.
* **Acting:** They execute specific tools (e.g., `tavily_search`, `OCR_tool`, `vector_store_retriever`).
* This ensures the agents don't just "guess" but actually retrieve real-time data or process files before responding.

### 3. Validator & Discriminator Logic (The Graph Updater)
A core technical challenge was ensuring that when a user asks to "change my roadmap," the resulting JSON structure remains valid. In `chat_tools.py`, we implemented a **Validator/Discriminator** pattern:
* **The Generator:** Proposes a modification to the `roadmap.json`.
* **The Discriminator (Validator):** An internal logic gate that checks the proposal against a strict Pydantic schema. It ensures that the updated roadmap is logically sequenced (e.g., prerequisite topics come first) and syntactically correct.
* **Refinement Loop:** If the Discriminator rejects the change, it provides feedback to the generator to fix the error before the state is ever saved to the database.

  
![Alt text for screen readers](images/Overview_final.png "Optional title for mouseover")

#### more detailed images could be found at `/images` subdirectory
---

## 📚 Learned Lessons

* RAG significantly improves factual accuracy and reliability.
* Semantic grading is essential for meaningful assessment.
* Modular agent design simplifies debugging and scaling.
* Memory systems greatly enhance personalization.

---

## 🎥 Media & Demo

* **Demo Video**: [![Watch the Demo Video](images/video_thumbnail.png)](https://drive.google.com/file/d/1z2sGzrw8PLx2DfMXWdKWcQLWseXAxikb/view?usp=sharing)
* **Screenshots**:  


---

## 📄 License

This project is released under the **MIT License**.
