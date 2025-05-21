# 🔍 Competency-Based RAG Chatbot

A simple **Retrieval-Augmented Generation (RAG)** chatbot powered by Hugging Face Transformers and LangChain. This chatbot answers questions using a competency knowledge base — useful for HR systems, employee upskilling platforms, or internal career recommendation tools.

---

## 🚀 Features

- ✅ Embeds and retrieves relevant competency documents
- ✅ Uses a custom prompt template to interact with a language model
- ✅ Provides context-aware answers with cited source summaries
- ✅ Built with `Gradio` for a user-friendly chat interface

---

## 📂 Project Structure
ML-project/
│
├── src/
│ ├── config_model.py # Loads and configures the pretrained model
│ ├── data_transformation.py # Cleans and transforms input data
│ ├── embedding_vector_store.py # Embeds data and creates a vector store
│ └── prompt_template.py # Defines the prompt template for the LLM
│
├── app.py # Main Gradio app file
├── requirements.txt
└── README.md


---

## 🧑‍💻 How It Works

1. **Data Loading and Transformation**  
   Loads and cleans competency and role data.

2. **Vector Store Creation**  
   Embeds documents and stores them using `FAISS` or similar (via LangChain).

3. **RAG Chain Setup**  
   Constructs a RAG pipeline using:
   - A retriever to fetch relevant documents
   - A custom prompt template for summarization or Q&A
   - A Hugging Face LLM (e.g., Mistral, BERT-based models)

4. **User Interaction (via Gradio)**  
   Users ask questions → app retrieves documents → LLM generates answers → chat displays them with context.

---

## ▶️ Running the App

### 1. Install dependencies

Make sure you're in a virtual environment (`venv` or `conda`), then run:

```bash
pip install -r requirements.txt

### 2. Launch the Gradio App
```bash
python app.py

**Notes**
The src/ modules must return the expected objects (e.g., LangChain retrievers, LLM chains).

You can extend this app to handle cold-start users or multiple role recommendations.

For deployment, consider Gradio sharing links, Hugging Face Spaces, or Docker.
