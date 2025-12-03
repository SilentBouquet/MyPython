from langchain_community.chat_models import ChatTongyi
from langchain_classic.agents import ConversationalAgent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from settings import ALIBABA_API_KEY


def init_llm(api_key: str):
    return ChatTongyi(
        api_key=api_key,
        model="qwen-turbo",
        top_p=0.9
    )


def create_conversational_agent(llm):
    """创建ConversationalAgent（带对话记忆）"""
    # 1. 初始化对话记忆（存储上下文）
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 记忆在prompt中的变量名
        return_messages=True  # 以消息对象形式返回记忆，兼容Chat模型
    )

    # 2. 自定义提示词（ConversationalAgent的核心模板）
    template = """你是一个友好的对话助手，善于根据对话历史回答用户当前问题。

    对话历史：
    {chat_history}

    用户当前问题：{input}

    请基于上述信息回答，语气自然。
    """
    prompt = PromptTemplate(
        input_variables=["chat_history", "input"],
        template=template
    )

    # 3. 创建ConversationalAgent（无需工具时，tools参数传空列表）
    agent = ConversationalAgent.from_llm_and_tools(
        llm=llm,
        tools=[],  # 不使用外部工具
        prompt=prompt,
        verbose=False  # 关闭详细日志（如需调试可设为True）
    )

    # 4. 创建Agent执行器（绑定记忆）
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=[],
        memory=memory,
        handle_parsing_errors=True  # 自动处理解析错误
    )


def run_conversational_chat():
    """运行带记忆的对话交互"""
    print("ConversationalAgent已启动，输入'退出'结束对话（支持上下文记忆）。")
    try:
        llm = init_llm(ALIBABA_API_KEY)
        agent_executor = create_conversational_agent(llm)
    except Exception as e:
        print(f"初始化失败：{str(e)}")
        return

    while True:
        user_input = input("你: ").strip()
        if user_input.lower() == "退出":
            print("Agent: 再见！")
            break
        if not user_input:
            print("Agent: 请输入内容~")
            continue

        try:
            # 调用Agent（自动带上对话历史）
            response = agent_executor.invoke({"input": user_input})
            print(f"Agent: {response['output']}")
        except Exception as e:
            print(f"处理失败：{str(e)}")


if __name__ == "__main__":
    run_conversational_chat()