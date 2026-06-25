import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="🤖"
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

llm = init_chat_model(
    "groq:llama-3.1-8b-instant"
)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}
    """
)

st.title("PDF Chatbot")

question = st.text_input(
    "Ask a question about the PDF"
)

if st.button("Ask"):

    with st.spinner("Thinking..."):

        docs = retriever.invoke(question)

        context = "\n".join(
            doc.page_content for doc in docs
        )

        chain = prompt | llm

        result = chain.invoke({
            "context": context,
            "question": question
        })

    st.subheader("Answer")
    st.write(result.content)
