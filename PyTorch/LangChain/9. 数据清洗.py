from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# 加载PDF文件（支持多页，自动提取页码元数据）
loader = PyPDFLoader("LangChain.pdf")

# 加载所有页面（返回Document列表，每个Document对应一页）
docs = loader.load()


def clean_document(doc: Document) -> Document:
    # 1. 去除多余空格和换行
    cleaned_content = doc.page_content.replace("\n\n", "\n").strip()
    # 2. 去除特殊字符（如\u200b零宽空格）
    cleaned_content = cleaned_content.replace("\u200b", "").replace("\\", "")
    # 3. 统一编码（避免乱码）
    cleaned_content = cleaned_content.encode("utf-8", errors="ignore").decode("utf-8")
    # 4. 更新Document内容
    return Document(
        page_content=cleaned_content,
        metadata=doc.metadata  # 保留原元数据
    )


# 批量清洗文档
cleaned_docs = [clean_document(doc) for doc in docs]

for i in range(5):
    print(cleaned_docs[i].page_content[:100])