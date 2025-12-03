from langchain_classic.memory import ConversationBufferWindowMemory  # 窗口记忆类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 新增导入
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

# 1. 初始化窗口记忆（只保留最近2轮）
memory = ConversationBufferWindowMemory(
    return_messages=True,
    memory_key="chat_history",
    k=2  # 关键参数：只保留最近2轮对话
)

# 2. 用MessagesPlaceholder处理历史对话
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是聊天助手，需结合历史对话回答当前问题"),
    # 关键：用MessagesPlaceholder渲染Message列表
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

llm = ChatTongyi(model="qwen-turbo", api_key=ALIBABA_API_KEY)
chain = prompt | llm


def chat(question):
    history = memory.load_memory_variables({})["chat_history"]
    response = chain.invoke({"question": question, "chat_history": history})
    memory.save_context({"question": question}, {"output": response.content})
    return response.content


# 测试（3轮对话，验证只保留最后2轮）
print(chat("我叫小明"))  # 轮1
print(chat("我喜欢吃苹果"))  # 轮2
print(chat("我刚才说我喜欢什么？"))  # 轮3：能记住轮2（最近2轮内）
print(chat("我叫什么名字？"))  # 轮4：轮1已被丢弃（超出k=2），会忘记名字
