from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun  # 导入DuckDuckGo搜索工具
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

# 1. 初始化DuckDuckGo搜索工具
search_tool = DuckDuckGoSearchRun()
tools = [search_tool]  # 工具列表，仅包含搜索工具

# 2. 初始化大语言模型
llm = ChatTongyi(model="qwen-turbo", api_key=ALIBABA_API_KEY)

# 3. 调整提示词：明确工具用途（网络搜索获取外部信息）
# 提示词需引导模型在需要"最新数据、外部知识、实时信息"时调用搜索工具
prompt = """
你拥有调用DuckDuckGo搜索工具的能力，可用于获取网络上的信息（如实时新闻、最新数据、事件结果等）。

工作流程：
1. 分析问题是否需要外部信息（如"2025年最新电影榜单"需要搜索，"1+1等于几"不需要）；
2. 若需要，调用搜索工具获取结果；
3. 根据搜索结果整理回答，若搜索无结果，需如实说明。

请基于以上规则回答问题。
"""

# 4. 初始化Agent（结合模型、工具和提示逻辑）
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # 适合聊天场景的反应式Agent
    verbose=True,  # 打印思考过程（调试用）
    agent_kwargs={"system_message": prompt}  # 注入自定义提示词
)

# 5. 示例问题（适合用搜索工具解决的问题）
question = "2025年11月，中国上海有什么新闻？"  # 需要实时数据，必须搜索
result = agent.invoke(question)['output']
print(result)