from langchain_classic.chains import SequentialChain, LLMChain, TransformChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

llm = ChatTongyi(model="qwen-turbo", api_key=ALIBABA_API_KEY)
parser = StrOutputParser()

# 1. 问题改写链
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("human", "将问题改写为更适合检索的关键词：{question}")
])
rewrite_chain = LLMChain(
    llm=llm,
    prompt=rewrite_prompt,
    output_parser=parser,
    output_key="rewritten_query"
)

# 2. 检索链（用TransformChain包装）
embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embedding)
retriever = vector_store.as_retriever(k=2)

def retrieval_transform(inputs: dict) -> dict:
    rewritten_query = inputs["rewritten_query"]
    docs = retriever.invoke(rewritten_query)
    context = "\n".join([doc.page_content for doc in docs])
    return {"context": context}

retrieval_chain = TransformChain(
    input_variables=["rewritten_query"],  # 明确输入变量（来自上一个链的输出）
    output_variables=["context"],  # 明确输出变量（供下一个链使用）
    transform=retrieval_transform  # 绑定转换函数
)

# 3. 回答生成链
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于上下文回答问题，不编造信息：\n{context}"),
    ("human", "{question}")
])
answer_chain = LLMChain(
    llm=llm,
    prompt=answer_prompt,
    output_parser=parser,
    output_key="final_answer"
)

# 4. 构建SequentialChain
rag_sequential_chain = SequentialChain(
    input_variables=["question"],
    output_variables=["final_answer"],
    chains=[rewrite_chain, retrieval_chain, answer_chain],
    verbose=True
)

# 调用
result = rag_sequential_chain.invoke({"question": "什么是机器学习？"})
print("最终回答：", result["final_answer"])