from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from transformers import AutoTokenizer
from Settings.Hugging_Face_API import HUGGING_FACE_API

# 加载PDF并清洗
loader = PyPDFLoader("LangChain.pdf")
docs = loader.load()


def clean_document(doc: Document) -> Document:
    cleaned_content = doc.page_content.replace("\n\n", "\n").strip()
    cleaned_content = cleaned_content.replace("\u200b", "").replace("\\", "")
    cleaned_content = cleaned_content.encode("utf-8", errors="ignore").decode("utf-8")
    return Document(page_content=cleaned_content, metadata=doc.metadata)


cleaned_docs = [clean_document(doc) for doc in docs if doc.page_content.strip()]  # 过滤空页
if not cleaned_docs:
    print("清洗后无有效文档，终止拆分")
    exit()

# 初始化千问官方Tokenizer
qwen_tokenizer = AutoTokenizer.from_pretrained(
    "qwen/Qwen-7B-Chat",
    trust_remote_code=True,  # 千问tokenizer需要此参数
    use_auth_token=HUGGING_FACE_API
)


# 定义千问token计算函数（准确计算文本的千问token数）
def qwen_token_length(text: str) -> int:
    # 千问tokenizer的encode方法返回token ID列表，长度即token数
    return len(qwen_tokenizer.encode(text, add_special_tokens=False))  # 不添加特殊token（避免额外计数）


# 3. 初始化拆分器（用千问token控制长度）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # 按千问token计算的Chunk长度（需匹配后续千问模型的context window，如qwen-turbo支持8k token）
    chunk_overlap=50,  # 重叠token数（避免拆分割裂语义，如句子被拆断）
    length_function=qwen_token_length,  # 关键：用千问token数判断长度
    separators=["\n\n", "\n", ". ", "！", "。", "？", " ", ""]  # 适配中文语义的拆分符（优先按段落、句子拆分）
)

# 4. 执行拆分并验证效果
chunks = text_splitter.split_documents(cleaned_docs)

# 打印拆分结果（验证正确性）
print(f"=== 拆分结果 ===")
print(f"原始有效文档数：{len(cleaned_docs)}")
print(f"拆分后Chunk数：{len(chunks)}")
print(f"第一个Chunk内容预览：\n{chunks[0].page_content[:150]}...")
print(f"第一个Chunk的千问token数：{qwen_token_length(chunks[0].page_content)}")
print(f"最后一个Chunk的千问token数：{qwen_token_length(chunks[-1].page_content)}")
