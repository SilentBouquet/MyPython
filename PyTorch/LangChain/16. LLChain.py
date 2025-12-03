from langchain_classic.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化基础组件
llm = ChatTongyi(model="qwen-turbo", api_key=ALIBABA_API_KEY)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是技术助手，用简洁语言回答关于LangChain的问题"),
    ("placeholder", "{history}"),  # 关联Memory的对话历史
    ("human", "{question}")        # 用户输入变量
])
memory = ConversationBufferMemory(return_messages=True, memory_key="history")
parser = StrOutputParser()

# 2. 构建LLMChain
llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    output_parser=parser,
    verbose=True  # 打印流程日志（便于调试）
)

# 3. 多轮调用（验证Memory协同）
print(llm_chain.invoke({"question": "什么是LLMChain？"})['text'])
print(llm_chain.invoke({"question": "它能和Memory一起用吗？刚才你提到了什么？"})['text'])