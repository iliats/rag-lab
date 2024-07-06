import io
import os

from langchain_core.documents import Document

import config
from langchain.globals import set_debug, set_verbose
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def init():
    set_debug(False)
    set_verbose(True)

    embeddings = AzureOpenAIEmbeddings(
        api_version=config.OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        model=config.EMBEDDINGS_MODEL,
        api_key=config.OPENAI_API_KEY
    )
    llm = AzureChatOpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        openai_api_version=config.OPENAI_API_VERSION,
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        model=config.OPENAI_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return embeddings, llm


def load_file_pages(uploaded_file: io.BytesIO) -> list[Document]:
    save_path = os.path.join(os.getcwd(), "files", "user_file.pdf")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    loader = PyPDFLoader(save_path)
    return loader.load_and_split()


def get_prompt() -> PromptTemplate:
    prompt_template = """You are professional doctor. Using the following context answer 
        the user question in 1-2 sentences.

        Context: {context}

        Question: {question}
    """

    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def generate(uploaded_file: io.BytesIO, query: str) -> str:
    if uploaded_file:
        embeddings, llm = init()

        pages = load_file_pages(uploaded_file)
        chroma_db = Chroma.from_documents(documents=pages, embedding=embeddings)
        retriever = chroma_db.as_retriever()
        prompt = get_prompt()

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        res = rag_chain.invoke(query)
        return res
    return "failed to parse file"
