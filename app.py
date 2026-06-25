from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from langchain_community.vectorstores import FAISS
vectorstore = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever()

from langchain.chat_models import init_chat_model
llm = init_chat_model(
    "groq:llama-3.1-8b-instant"
)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}
    """
)

question = "What is Generative AI?"

docs = retriever.invoke(question)

context = "\n".join(
    doc.page_content for doc in docs
)

chain = prompt | llm
result = chain.invoke({
    "context": context,
    "question": question
})

print(result.content)