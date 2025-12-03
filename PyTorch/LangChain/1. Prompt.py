# 多变量字符串模板
from langchain_core.prompts import PromptTemplate

summary_template = PromptTemplate(
    input_variables=["document", "summary_length"],  # 显式声明参数（强制校验）
    template="""请将以下文档总结为{summary_length}字以内的内容：
    文档：{document}
    总结："""
)

# 渲染 Prompt
prompt = summary_template.format(
    document="LangChain是LLM应用开发框架...",
    summary_length=200
)

print(prompt)
print()

# 含元数据的对话模板
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

rag_chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是基于文档的问答助手，仅根据以下文档回答：\n来源：{doc_source}\n文档：{doc_content}"),
    ("human", "{question}"),
    ("ai", "{history}")  # 历史对话占位符（结合Memory模块）
])

# 从Document对象提取元数据
doc = Document(
    page_content="LangChain核心模块包括Model I/O...",
    metadata={"source": "LangChain官方文档v0.2", "page": 5}
)

prompt = rag_chat_template.format(
    doc_source=doc.metadata["source"],
    doc_content=doc.page_content,
    question="LangChain的Model I/O模块有什么用？",
    history=""
)

print(prompt)
print()

from langchain_core.prompts import FewShotPromptTemplate

# 1. 定义示例
examples = [
    {"text": "LangChain很强大", "sentiment": "正面"},
    {"text": "模型调用经常超时", "sentiment": "负面"},
    {"text": "文档拆分需要调参", "sentiment": "中性"}
]

# 2. 示例模板
example_template = PromptTemplate(
    input_variables=["text", "sentiment"],
    template="文本：{text}\n情感：{sentiment}"
)

# 3. Few-Shot模板
few_shot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="请判断以下文本的情感（正面/负面/中性）：",  # 指令前缀
    suffix="文本：{new_text}\n情感：",  # 待预测文本后缀
    input_variables=["new_text"]  # 仅需传入新文本
)

# 渲染
prompt = few_shot_template.format(new_text="RAG检索效果符合预期")  # 输出含3个示例的Prompt
print(prompt)