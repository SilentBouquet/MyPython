from langchain_community.document_loaders import SQLDatabaseLoader
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.documents import Document

# 1. 连接数据库
db = SQLDatabase.from_uri(
    "mysql+mysqlconnector://root:yy040806@localhost:3306/product_reviews"
)

# 2. 定义查询（获取所有需要的列）
query = """
        SELECT id, product_name, content, score, create_time
        FROM comments
        WHERE score >= 4; \
        """

# 3. 加载原始数据（默认格式）
loader = SQLDatabaseLoader(db=db, query=query)
raw_documents = loader.load()

# 4. 手动处理：提取 content 作为 page_content，其他列作为 metadata
processed_documents = []
for doc in raw_documents:
    # 解析原始 page_content（默认格式为 "列名: 值\n..."）
    content_lines = doc.page_content.split("\n")
    content_dict = {}
    for line in content_lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            content_dict[key.strip()] = value.strip()

    # 构建新的 Document
    processed_doc = Document(
        page_content=content_dict.get("content", ""),  # 核心文本
        metadata={
            "id": content_dict.get("id"),
            "product_name": content_dict.get("product_name"),
            "score": content_dict.get("score"),
            "create_time": content_dict.get("create_time"),
            "source": doc.metadata.get("source")  # 保留来源信息
        }
    )
    processed_documents.append(processed_doc)

# 5. 打印处理后的结果
print(f"共加载 {len(processed_documents)} 条数据：\n")
for i, doc in enumerate(processed_documents, 1):
    print(f"第 {i} 条：")
    print(f"page_content: {doc.page_content}")
    print(f"metadata: {doc.metadata}\n")