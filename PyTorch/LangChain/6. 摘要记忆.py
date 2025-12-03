from langchain_classic.memory import ConversationSummaryMemory  # 摘要记忆核心类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

# 初始化大语言模型（用于生成回复和对话摘要）
llm = ChatTongyi(
    model="qwen-turbo",
    api_key=ALIBABA_API_KEY
)

# 初始化摘要记忆（核心：用LLM自动生成对话摘要）
memory = ConversationSummaryMemory(
    llm=llm,
    return_messages=True,  # 记忆返回Message对象（而非纯文本，方便Prompt渲染）
    memory_key="chat_history",  # 记忆存储的键名，需与Prompt中的占位符一致
    input_key="question",  # 输入变量名（用户问题）
    output_key="output"  # 输出变量名（AI回复）
)

# 4. 构建提示词模板（包含历史对话摘要占位符）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个聊天助手，需要结合历史对话的摘要信息回答当前问题。保持回答简洁自然。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# 构建对话链
chain = prompt | llm


# 封装对话函数（自动加载记忆、调用模型、保存对话）
def chat(question):
    # 从记忆中加载历史对话摘要
    history = memory.load_memory_variables({})["chat_history"]

    # 调用链生成回复（传入问题和历史摘要）
    response = chain.invoke({
        "question": question,
        "chat_history": history
    })

    # 将本轮对话（问题+回复）存入记忆，触发摘要更新
    memory.save_context(
        inputs={"question": question},
        outputs={"output": response.content}
    )

    # 打印当前记忆中的摘要（方便观察摘要变化）
    print("当前对话摘要：", memory.buffer)  # buffer属性存储了文本形式的摘要
    return response.content


# 7. 多轮对话测试（验证摘要记忆效果）
if __name__ == "__main__":
    print("第一轮对话")
    print("用户：我叫张三，今年30岁，在一家互联网公司做产品经理")
    print("AI：", chat("我叫张三，今年30岁，在一家互联网公司做产品经理"))

    print("\n第二轮对话")
    print("用户：我平时喜欢跑步和看电影，最近在看《奥本海默》")
    print("AI：", chat("我平时喜欢跑步和看电影，最近在看《奥本海默》"))

    print("\n第三轮对话")
    print("用户：能总结一下我刚才说的所有信息吗？")
    print("AI：", chat("能总结一下我刚才说的所有信息吗？"))