from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY


# 1. 定义Pydantic模型（输出格式）
class ProductInfo(BaseModel):
    product_name: str = Field(description="产品名称")
    price: float = Field(description="产品价格，需为数字")
    category: str = Field(description="产品分类（可选：电子、家居、服装）")
    features: list[str] = Field(description="产品核心特性，至少3项")


# 2. 初始化解析器
parser = PydanticOutputParser(pydantic_object=ProductInfo)

# 3. 构建Prompt（需注入解析器的格式说明）
prompt = PromptTemplate(
    input_variables=["product_description"],
    template="""根据以下产品描述提取结构化信息：
    产品描述：{product_description}
    输出格式必须严格遵循：{format_instructions}
    不要输出任何多余内容！""",
    partial_variables={"format_instructions": parser.get_format_instructions()}  # 注入格式说明
)

llm_sync = ChatTongyi(
    model= "qwen-turbo",
    api_key=ALIBABA_API_KEY
)

# 4. 构建链（Prompt→LLM→解析器）
chain = prompt | llm_sync | parser

# 5. 调用链（输出为Pydantic对象，支持属性访问）
result = chain.invoke({
    "product_description": "iPhone 15是苹果推出的智能手机，售价5999元，支持灵动岛、4800万像素摄像头、A17 Pro芯片"
})

print(result.product_name)  # 输出：iPhone 15
print(result.price)  # 输出：5999.0
print(result.features)  # 输出：["灵动岛", "4800万像素摄像头", "A17 Pro芯片"]