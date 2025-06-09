import transformers
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

from dataclasses import dataclass
import os

@dataclass
class PromptTemplateBuilder:
    def __init__(self):
        pass
    def build_prompt_template(self,model, tokenizer):
        prompt_templates = """
                            ### [INST]
                            You are an AI Assistant that helps users find the answer from relevant documents.

                            If there are no relevant documents or the context does not match the user's question, respond with:
                            "Sorry, there are no relevant documents in the system."

                            {context}

                            ### QUESTION:
                            {question}

                            [/INST]
                            """
        text_generation_pipeline = transformers.pipeline(
                                                        model=model,
                                                        tokenizer=tokenizer,
                                                        task="text-generation",
                                                        temperature=0.2,
                                                        repetition_penalty=1.1,
                                                        return_full_text=True,
                                                        max_new_tokens=300,
                                                    )
        mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        # Create prompt from prompt template
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_templates,
        )

        # Create llm chain
        # llm_chain = LLMChain(llm=mistral_llm, prompt=prompt)
        llm_chain = prompt | mistral_llm

        return llm_chain
    
    
    
