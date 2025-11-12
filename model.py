from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama as ChatOllama
from typing import List, Optional
from langchain_core.documents import Document
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import BM25Retriever, EnsembleRetriever
def get_or_create_faiss_index(
    index_dir: str,
    docs: Optional[List[Document]],
    embeddings: SentenceTransformerEmbeddings,
    *,
    should_save: bool = True
) -> FAISS:
    """
    If an index exists in `index_dir`, load it. Otherwise, build from `docs`
    and save it (when should_save=True).
    """
    index_path = Path(index_dir)
    exists = index_path.exists() and any(index_path.iterdir())

    if exists:
        # Load previously saved index
        return FAISS.load_local(
            index_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

    if not docs:
        raise ValueError(
            "Index not found and `docs` is None/empty. "
            "Provide documents to build the index the first time."
        )

    vectordb = FAISS.from_documents(docs, embeddings)
    if should_save:
        index_path.mkdir(parents=True, exist_ok=True)
        vectordb.save_local(index_dir)

INDEX_DIR = "faiss_index"  # folder to store/reuse index files
SOURCE_FILE = "fitness.txt"

# 1) Embeddings (define once; same model used for build & load)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2) Load documents ONLY if index doesn't exist (we’ll pass docs below)
#    You can keep this simple: load+chunk every time; the helper will ignore docs when loading existing index.
loader = TextLoader(SOURCE_FILE, encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# 3) Get the vector store (load existing or build+save once)
vectordb = get_or_create_faiss_index(
    index_dir=INDEX_DIR,
    docs=splits,
    embeddings=embeddings,
    should_save=True
)

# 4) Retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 5) Your LLM + chain
llm = ChatOllama(model="mistral", temperature=0.3)
prompt = ChatPromptTemplate.from_template("Answer from context:\n{context}\n\nQuestion: {question}")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}  # plugs retrieved docs + user question
    | prompt
    | llm
    | StrOutputParser()
)

# 6) Demo query — now it won’t rebuild the index on subsequent runs
print(chain.invoke("Give me fitness plans"))