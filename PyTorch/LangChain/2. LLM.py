from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

# 1. 基础同步调用（非流式）
llm_sync = ChatTongyi(
    model= "qwen-turbo",
    api_key=ALIBABA_API_KEY
)
response_sync = llm_sync.invoke("LangChain的Model I/O模块核心作用？")
print("基础调用：", response_sync.content)

# 2. 流式调用（实时输出）
llm_stream = ChatTongyi(
    model= "qwen-turbo",
    api_key=ALIBABA_API_KEY,
    streaming=True
)
response_stream = llm_stream.invoke("LangChain的Model I/O模块核心作用？")  # 逐字输出
print("\n流式调用：", response_stream.content)

# 3. 异步调用（高并发场景）
import asyncio


async def async_llm_call():
    llm_async = ChatTongyi(model= "qwen-turbo", api_key=ALIBABA_API_KEY)
    response_async = await llm_async.ainvoke("LangChain的Model I/O模块核心作用？")
    print("\n异步调用：", response_async.content)


asyncio.run(async_llm_call())