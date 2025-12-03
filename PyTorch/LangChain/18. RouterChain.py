from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

# 初始化大模型
llm = ChatTongyi(model="qwen-turbo", api_key=ALIBABA_API_KEY)

# 1. 定义子链
# 技术问答链
tech_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是技术专家，详细解释LangChain的技术原理"),
    ("human", "{question}")
])
tech_chain = tech_prompt | llm | StrOutputParser()  # 最新推荐写法

# 产品问答链
product_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是产品经理，用通俗语言介绍LangChain的功能和使用场景"),
    ("human", "{question}")
])
product_chain = product_prompt | llm | StrOutputParser()  # 最新推荐写法

# 2. 定义分类链（判断问题类型）
classify_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
请判断用户问题属于以下哪种类型，仅输出「技术问题」或「产品问题」，不要其他内容：
- 技术问题：涉及LangChain的原理、代码实现、模块架构等
- 产品问题：涉及LangChain的功能、使用场景、优势对比等

用户问题：{question}
    """
)
classify_chain = classify_prompt | llm | StrOutputParser()  # 分类链


# 3. 路由逻辑：先分类，再调用对应子链
def invoke_chain(question):
    # 第一步：分类问题
    category = classify_chain.invoke({"question": question}).strip()
    print(f"识别问题类型：{category}")  # 可选：打印分类结果

    # 第二步：调用对应链
    if category == "技术问题":
        return tech_chain.invoke({"question": question})
    elif category == "产品问题":
        return product_chain.invoke({"question": question})
    else:
        # 分类失败时的默认处理
        return "无法识别问题类型，请重新提问"


# 测试
print("技术问题测试：")
print(invoke_chain("LLMChain的内部数据流转逻辑是什么？"))
print("\n产品问题测试：")
print(invoke_chain("LangChain相比其他LLM框架有什么优势？"))