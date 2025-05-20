import gradio as gr
from src.data_transformation import DataTransformation
from src.embedding_vector_store import DataEmbedding
from src.config_model import ModelConfig
from src.prompt_template import PromptTemplateBuilder
from langchain.schema.runnable import RunnableLambda
import re


def load_components():
    transformer = DataTransformation()
    clean_competencies_df, role_df = transformer.transforamtion_data()

    embedding_vector = DataEmbedding()
    db = embedding_vector.embedding_vector_store(clean_competencies_df, role_df)

    model_config = ModelConfig()
    model, tokenizer = model_config.pretrained_model_config()

    builder = PromptTemplateBuilder()
    llm_chain = builder.build_prompt_template(model, tokenizer)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})

    def get_context_and_summary(question):
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return {
                "context": "No relevant documents.",
                "summary": "",
                "question": question
            }

        context = "\n".join([
            f"{doc.metadata.get('name', '')}\n→ {doc.page_content}"
            for doc in docs
        ])

        return {
            "context": context,
            "summary": context,
            "question": question
        }

    rag_chain = RunnableLambda(get_context_and_summary) | llm_chain
    return rag_chain


rag_chain = load_components()

def chat_with_rag(message, history):
    result = rag_chain.invoke(message)

    try:
        raw_text = result["text"]
        clean_text = re.sub(r"QUESTION:.*?\n+", "", raw_text).strip()
        answer = clean_text.split('[/INST]')[-1].strip()

        context = result.get("summary", "")
        text_ans = f"{answer}\n\n📘 *Competency Suggestions:* \n{context}"
        return text_ans
    except Exception as e:
        return f"⚠️ Error: {e}"

# Build a chat interface (with memory)
chat_ui = gr.ChatInterface(
    fn=chat_with_rag,
    title="🔍 RAG Chatbot",
    description="Ask me anything based on the competency knowledge base!",
    theme="default"
)

# Launch it
chat_ui.launch()

