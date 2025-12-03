from langchain_chroma import Chroma  # 从独立包导入Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer

# 1. 加载PDF并清洗
loader = PyPDFLoader("LangChain.pdf")
docs = loader.load()


def clean_document(doc: Document) -> Document:
    cleaned_content = doc.page_content.replace("\n\n", "\n").strip()
    cleaned_content = cleaned_content.replace("\u200b", "").replace("\\", "")
    cleaned_content = cleaned_content.encode("utf-8", errors="ignore").decode("utf-8")
    return Document(page_content=cleaned_content, metadata=doc.metadata)


cleaned_docs = [clean_document(doc) for doc in docs if doc.page_content.strip()]
if not cleaned_docs:
    print("清洗后无有效文档，终止拆分")
    exit()

# 2. 初始化千问官方Tokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained(
    "qwen/Qwen-7B-Chat",
    trust_remote_code=True
)


def qwen_token_length(text: str) -> int:
    return len(qwen_tokenizer.encode(text, add_special_tokens=False))


# 3. 初始化拆分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=qwen_token_length,
    separators=["\n\n", "\n", ". ", "！", "。", "？", " ", ""]
)

chunks = text_splitter.split_documents(cleaned_docs)

# 初始化开源嵌入模型（模型ID：all-MiniLM-L6-v2，轻量且效果好）
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# 初始化Chroma（无需手动persist）
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="./chroma_db"
)

'''
# 加载已有的Chroma存储
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)
'''

# 4. 基础检索
query = "LangChain 有什么模块？"
retrieved_docs = vector_store.similarity_search(
    query=query,
    k=3  # 返回的Chunk数量
)

# 查看检索结果
for i, doc in enumerate(retrieved_docs, 1):
    print(f"第{i}个Chunk：")
    print(f"内容：{doc.page_content[:300]}...")
    print(f"来源：{doc.metadata['source']}，页码：{doc.metadata.get('page', '无')}")
    print("-" * 50)