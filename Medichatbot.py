import streamlit as st
import os
from dotenv import load_dotenv

# Force load variables from your .env file right at startup
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

def load_llm():
    api_key_from_env = os.environ.get("GROQ_API_KEY")
    
    if not api_key_from_env:
        st.error("GROQ_API_KEY is missing from your .env file! Please check its path and contents.")
        return None

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3, # Slightly lower temperature for more precise medical answers
        max_tokens=512,
        api_key=api_key_from_env  
    )
    return llm


def main():
    # 1. Page Configuration (Sets the browser tab title and medical icon)
    st.set_page_config(
        page_title="MediQuery AI Portal", 
        page_icon="🩺", 
        layout="centered"
    )

    # 2. Classic & Professional CSS Injections
    st.markdown("""
        <style>
            /* Main Header Styling */
            .main-title {
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                color: #0F2C59; /* Deep Classic Navy */
                font-weight: 700;
                font-size: 2.4rem;
                margin-bottom: 5px;
            }
            .sub-title {
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                color: #6C7A89; /* Neutral Professional Grey */
                font-size: 1.1rem;
                margin-bottom: 25px;
            }
            /* Adjust padding for clean viewport spacing */
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)  # <-- FIXED HERE

    # Render Styled Header Elements
    st.markdown('<div class="main-title">🩺 MediQuery AI Portal</div>', unsafe_allow_html=True)  # <-- FIXED HERE TOO!
    st.markdown('<div class="sub-title">Clinical Knowledge & Retrieval Assistant</div>', unsafe_allow_html=True) # <-- FIXED HERE TOO!
    st.markdown("---")


    
    # Initialize chat history if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display old messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Enter clinical query or symptom information...")

    if prompt:
        # Display user message in chat message container
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        try:
            llm_client = load_llm()
            if llm_client is None:
                return

            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm_client,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # A professional clinical loading message
            with st.spinner("Analyzing verified vector databases..."):
                response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]
            
            # Format the output with clean spacing and Markdown rules
            result_to_show = f"{result}\n\n"
            
            # Group reference metadata cleanly into an architectural block
            sources_block = "\n**Verified Reference Material Used:**\n"
            for doc in source_documents:
                source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                page = doc.metadata.get('page', 0) + 1
                sources_block += f"- *{source_file} (Page {page})*\n"
            
            # Combine them for history log
            full_assistant_payload = result_to_show + sources_block
            
            # Render to interface
            st.chat_message('assistant').markdown(full_assistant_payload)
            st.session_state.messages.append({'role': 'assistant', 'content': full_assistant_payload})

        except Exception as e:
            st.error(f"Error executing retrieval sequence: {str(e)}")

if __name__ == "__main__":
    main()