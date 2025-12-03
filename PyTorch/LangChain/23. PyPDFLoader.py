# 1. 安装依赖（PyPDF2用于解析PDF）
# pip install pypdf2 langchain-community
# 2. 初始化Loader（本地PDF）
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path="../example.pdf")

# 3. 加载数据（返回Document列表）
docs = loader.load()

# 4. 查看结果（验证文本和元数据）
print(f"加载的Document数量：{len(docs)}")
print(f"第1页文本前200字符：{docs[0].page_content[:200]}")
print(f"第1页元数据：{docs[0].metadata}")  # 输出：{"source": "...", "page": 1, "file_path": "..."}