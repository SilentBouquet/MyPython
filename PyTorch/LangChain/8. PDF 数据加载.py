from langchain_community.document_loaders import PyPDFLoader

# 加载PDF文件（支持多页，自动提取页码元数据）
loader = PyPDFLoader("LangChain.pdf")

# 加载所有页面（返回Document列表，每个Document对应一页）
docs = loader.load()

# 查看第一页的内容与元数据
print(len(docs))

for i, doc in enumerate(docs):
    print(f"第{i+1}页内容：", doc.page_content[:50])
    print(f"第{i+1}页元数据：", doc.metadata)