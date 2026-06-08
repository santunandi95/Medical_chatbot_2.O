import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Setup LLM (Groq - Free API, ultra-fast)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def load_llm():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Free & fast model on Groq
        temperature=0.5,
        max_tokens=512,
        api_key=GROQ_API_KEY
    )
    return llm


# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt


# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Invoke with user query
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("\nRESULT:", response["result"])
print("\nSOURCE DOCUMENTS:", response["source_documents"])
