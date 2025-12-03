from langchain_chroma import Chroma  # 从独立包导入Chroma
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings  # 从独立包导入嵌入模型
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY
from langchain_classic.retrievers import ContextualCompressionRetriever

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

# 初始化嵌入模型
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# 加载已有的Chroma存储
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

query = "薛天緯的評點"

llm = ChatTongyi(
    model="qwen-turbo",
    api_key=ALIBABA_API_KEY
)

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
    请从以下上下文中，提取与问题“{question}”最相关的内容。
    仅返回提取的内容，不要添加额外解释。如果上下文与问题无关，返回空字符串。
    上下文：{context}
    """
)

# 初始化压缩器（用LLM提取相关内容）
compressor = LLMChainExtractor.from_llm(llm, prompt=prompt)

# 包装检索器
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(k=3)  # 基础检索器
)

# 压缩检索
compressed_docs = compression_retriever.invoke(query)

# 检查基础检索器返回的原始文档
base_retriever = vector_store.as_retriever(k=3)
base_docs = base_retriever.invoke(query)
print("基础检索到的文档：")
for i, doc in enumerate(base_docs):
    print(f"第{i+1}个文档：{doc.page_content[:500]}...")  # 打印部分内容

# 查看压缩后的内容（仅保留与查询相关的部分）
print("\n压缩后的内容：", compressed_docs[0].page_content)