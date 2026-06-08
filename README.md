# 🩺 MediQuery AI Portal (Medical Chatbot 2.O)

MediQuery AI Portal is a Clinical Knowledge & Retrieval Assistant built using a Retrieval-Augmented Generation (RAG) architecture. It allows users to query clinical documentation (such as medical encyclopedias or textbooks) in real-time, fetching precise answers along with their verified source reference citations.

---

## 🚀 Key Features

* **Semantic Search & RAG**: Combines local document context with Large Language Models to provide fact-based, context-grounded responses.
* **Source Citations**: Displays verified document names and page numbers used to generate the responses, ensuring traceability and accountability.
* **Vibrant Streamlit Interface**: Offers an intuitive, chat-based UI designed for clean clinical use.
* **Ultra-fast Inference**: Leverages Groq Cloud API with `llama-3.1-8b-instant` for near-instant responses.
* **Local Embeddings**: Utilizes the Hugging Face `sentence-transformers/all-MiniLM-L6-v2` model for high-quality local text embedding.

---

## 🛠️ Tech Stack

* **Framework**: [Streamlit](https://streamlit.io/) (Web UI)
* **Orchestration**: [LangChain](https://www.langchain.com/) (Document loaders, chunking, memory retriever)
* **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
* **LLM API**: [Groq](https://groq.com/) (`llama-3.1-8b-instant`)
* **Environment Management**: Pipenv

---

## 📂 Project Structure

```text
├── data/                       # Directory containing raw PDFs (e.g., GALE Encyclopedia)
├── vectorstore/
│   └── db_faiss/               # Generated FAISS database and index files
├── Medichatbot.py              # Main Streamlit web application
├── create_memory_for_llm.py    # Script to load PDFs, chunk text, embed and build FAISS DB
├── connect_memory_with_llm.py  # CLI-based testing script for RAG retrieval
├── Pipfile                     # Python packages and environment specifications
├── Pipfile.lock                # Locked dependency tree
├── .gitignore                  # Git ignore rules (excludes .env and Python caches)
└── README.md                   # This project documentation
```

---

## 🏁 Getting Started

### 📋 Prerequisites

* Python **3.13** or higher
* [Pipenv](https://pipenv.pypa.io/) installed (`pip install pipenv`)
* Groq API Key (get one free at [console.groq.com](https://console.groq.com/))

### 🔧 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/santunandi95/Medical_chatbot_2.O.git
   cd Medical_chatbot_2.O
   ```

2. **Install Dependencies**:
   ```bash
   pipenv install
   ```
   *Alternatively, activate the environment shell:*
   ```bash
   pipenv shell
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your Groq API Key:
   ```env
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```

4. **Prepare Vector Database** (Optional):
   Place any PDF document(s) you wish to query in the `data/` folder and run the builder script to generate the FAISS embeddings database:
   ```bash
   python create_memory_for_llm.py
   ```

5. **Run the Application**:
   Start the Streamlit portal:
   ```bash
   streamlit run Medichatbot.py
   ```

---

## 🏥 Prompt Customization

The system operates with a strict instruction template targeting clinically accurate, context-bound answers:
> "Use the pieces of information provided in the context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Don't provide anything out of the given context."

---

## 📝 License

This project is open-source. Please see the project guidelines and terms of use for clinical assistants.
