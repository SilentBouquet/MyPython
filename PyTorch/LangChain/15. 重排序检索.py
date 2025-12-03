from langchain_chroma import Chroma  # 从独立包导入Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings  # 从独立包导入嵌入模型
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer

# 1. 加载PDF并清洗
loader = PyPDFLoader("../example.pdf")
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

# 初始化嵌入模型（使用新的导入方式）
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# 加载已有的Chroma存储
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

query = "秦婦吟是谁写的？"

cross_encoder_model = HuggingFaceCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    model_kwargs={"device": "cpu"}
)

reranker = CrossEncoderReranker(
    model=cross_encoder_model,  # 轻量重排序模型
    top_n=3  # 重排序后保留Top 3
)

rerank_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vector_store.as_retriever(k=10)  # 先获取Top 10，再重排序
)

reranked_docs = rerank_retriever.invoke(query)

# 查看检索结果
for i, doc in enumerate(reranked_docs, 1):
    print(f"第{i}个Chunk：")
    print(f"内容：{doc.page_content[:300]}...")
    print(f"来源：{doc.metadata['source']}，页码：{doc.metadata.get('page', '无')}")
    print("-" * 50)
