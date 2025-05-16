import streamlit as st
from src.data_transformation import DataTransformation
from src.embedding_vector_store import DataEmbedding
from src.config_model import ModelConfig
from src.prompt_template import PromptTemplateBuilder
from langchain.schema.runnable import RunnableLambda
import re



# Set up page config
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🔍 RAG Chatbot")
st.markdown("Ask me anything based on the knowledge base!")

@st.cache_resource
def load_components():
    # Data ingestion and transformation
    transformer = DataTransformation()
    clean_competencies_df, role_df = transformer.transforamtion_data()

    # Embedding and vector store
    embedding_vector = DataEmbedding()
    db = embedding_vector.embedding_vector_store(clean_competencies_df, role_df)

    # Model and tokenizer
    model_config = ModelConfig()
    model, tokenizer = model_config.pretrained_model_config()

    # Prompt and chain
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

# Load the chain only once
rag_chain = load_components()

output = rag_chain.invoke("I want to learn about data")
raw_text = output["text"]
clean_text = re.sub(r"QUESTION:.*?\n+", "", raw_text).strip()
answer = clean_text.split('[/INST]')[-1].strip()
print(answer)

# Input box for the user
user_question = st.text_input("📩 Enter your question:")

if user_question:
    with st.spinner("Generating answer..."):
        output = rag_chain.invoke(user_question)
        raw_text = output["text"]
        clean_text = re.sub(r"QUESTION:.*?\n+", "", raw_text).strip()
        answer = clean_text.split('[/INST]')[-1].strip()
        st.success("✅ Answer")
        st.markdown(answer)
