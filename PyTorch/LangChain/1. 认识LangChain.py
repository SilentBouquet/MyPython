import os
from Config import settings
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser

os.environ["DEEPSEEK_API_KEY"] = settings.get("DEEPSEEK_API_KEY", None)


# 自定义输出解析器
class CustomOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        # 假设返回的文本格式为：boy_names,girl_names
        try:
            # 按分号分割男孩和女孩的名字列表
            boy_list, girl_list = text.split(";")
            # 去除多余的空格和方括号，然后按逗号分割名字
            boy_names = [name.strip().strip('"') for name in boy_list.strip("[]").split(",")]
            girl_names = [name.strip().strip('"') for name in girl_list.strip("[]").split(",")]

            # 确保每个列表有3个名字
            if len(boy_names) != 3 or len(girl_names) != 3:
                raise ValueError("每个列表必须包含3个名字")

            return {"boy": boy_names, "girl": girl_names}
        except Exception as e:
            raise ValueError(f"解析结果失败：{str(e)}")


# 初始化 DeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",  # 使用 DeepSeek 的聊天模型
    temperature=0.7,        # 温度值，控制生成文本的随机性
    max_tokens=1024         # 最大生成的 token 数
)

prompt_sys = "你是一个起名大师，擅长根据各个国家的特色为男生或女生取一个个性化的名字。"
prompt_human = PromptTemplate.from_template("请模仿示例起三个{country}名字，比如，男孩经常被叫做{boy}，女孩经常被叫做{girl}。"
                                            "请返回两个列表，中间用分号隔开，每个列表里包含3个名字，用逗号隔开，第一个列表为男生的名字，第二个为女生的名字。"
                                            "注意，不要返回无关信息，只返回名字，并且严格按照我给定的格式。")
message = prompt_human.format_prompt(country="中国特色的", boy="狗蛋", girl="翠花").to_string()
print(message)

# 构造消息列表
messages = [
    ("system", prompt_sys),
    ("human", message)
]

# 调用模型进行回答
response = llm.invoke(messages)

# 输出模型的回应
print("AI Assistant Response:")
print(response.content)

# 使用自定义解析器解析结果
parser = CustomOutputParser()
try:
    parsed_output = parser.parse(response.content)
    print("Parsed Output:")
    print(parsed_output)
except ValueError as e:
    print(f"Error: {e}")