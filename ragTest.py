import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
import getpass
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
from langchain_community.llms import Ollama
llm = Ollama(model="mistral", temperature =0)
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader( web_path=("https://www.psaworldtour.com/rankings/"))
docs = loader.load()
docs[0].page_content
len(docs[0].page_content)
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200, add_start_index = True)
all_splits = text_splitter.split_documents(docs)
len(all_splits)
len(all_splits[1].page_content)


from langchain_community.embeddings.ollama import OllamaEmbeddings
embedding = OllamaEmbeddings( model="nomic-embed-text")

from langchain_community.vectorstores import Chroma
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding)


retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs={"k":6})
retriever.get_relevant_documents(
    "What is langGraph"
)

from langchain import hub
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain =(
    {"context": retriever | format_docs, "question": RunnablePassthrough() } 
    | prompt
    | llm
    | StrOutputParser()

)

print(rag_chain.invoke("Who are the top five squash players in the world?"))


