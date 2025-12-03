from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 新增导入
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

# 初始化Memory
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"  # 与Prompt中的占位符key一致
)

# 用MessagesPlaceholder处理历史对话
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是聊天助手，需结合历史对话回答当前问题"),
    # 用MessagesPlaceholder渲染Message列表
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

llm = ChatTongyi(model="qwen-turbo", api_key=ALIBABA_API_KEY)
chain = prompt | llm

# 多轮对话（需手动将历史对话传入）
# 第一轮：传入空历史，同时更新memory
first_input = {"question": "我叫小明", "chat_history": memory.load_memory_variables({})["chat_history"]}
first_output = chain.invoke(first_input)
print("第一轮AI回复：", first_output.content)
# 更新memory：将本轮对话存入
memory.save_context({"question": "我叫小明"}, {"output": first_output.content})

# 第二轮：传入已存储的历史对话
second_input = {"question": "我是谁？", "chat_history": memory.load_memory_variables({})["chat_history"]}
second_output = chain.invoke(second_input)
print("第二轮AI回复：", second_output.content)

third_input = {"question": "我今年12岁了", "chat_history": memory.load_memory_variables({})["chat_history"]}
third_output = chain.invoke(third_input)
print("第三轮AI回复：", third_output.content)
memory.save_context({"question": "我今年12岁了"}, {"output": third_output.content})

fourth_input = {"question": "告诉我关于我的信息", "chat_history": memory.load_memory_variables({})["chat_history"]}
fourth_output = chain.invoke(fourth_input)
print("第四轮AI回复：", fourth_output.content)