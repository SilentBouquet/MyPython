from langchain_classic.memory import ConversationEntityMemory, ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

# 初始化模型
llm = ChatTongyi(
    model="qwen-turbo",
    api_key=ALIBABA_API_KEY
)

# 实体记忆：只负责提取和存储实体（如“年糕-布偶猫”）
entity_memory = ConversationEntityMemory(
    llm=llm,
    memory_key="entities",  # 实体键名
    input_key="question",  # 输入键（用户问题）
    output_key="output"  # 输出键（AI回复）
)

# 对话历史记忆：单独存储完整对话
history_memory = ConversationBufferMemory(
    memory_key="chat_history",  # 对话历史键名
    return_messages=True
)

# 提示词模板：同时使用实体和对话历史
prompt = ChatPromptTemplate.from_messages([
    ("system", "结合已知实体信息回答：\n{entities}\n"),
    MessagesPlaceholder(variable_name="chat_history"),  # 对话历史来自 history_memory
    ("human", "{question}")
])

# 对话链
chain = prompt | llm


def chat(question):
    # 加载实体（从 entity_memory）
    entities = entity_memory.load_memory_variables({"question": question})["entities"]
    # 加载对话历史（从 history_memory，不再依赖实体记忆）
    chat_history = history_memory.load_memory_variables({})["chat_history"]

    # 生成回复
    response = chain.invoke({
        "question": question,
        "entities": entities,
        "chat_history": chat_history
    })

    # 同时更新两个记忆：实体记忆和对话历史记忆
    entity_memory.save_context({"question": question}, {"output": response.content})
    history_memory.save_context({"question": question}, {"output": response.content})

    return response.content


# 测试对话
print("用户：我家有只猫叫年糕，是只布偶猫")
print("AI：", chat("我家有只猫叫年糕，是只布偶猫"))

print("\n用户：年糕今年2岁，特别喜欢吃冻干")
print("AI：", chat("年糕今年2岁，特别喜欢吃冻干"))

print("\n用户：年糕是什么品种？")
print("AI：", chat("年糕是什么品种？"))

print("\n用户：它喜欢吃什么？")
print("AI：", chat("它喜欢吃什么？"))