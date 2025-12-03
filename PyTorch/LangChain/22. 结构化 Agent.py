from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.schema.messages import SystemMessage
from langchain_classic.agents.format_scratchpad import format_to_openai_function_messages
from langchain_classic.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import datetime
import math
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY


# 1. 定义工具参数的Pydantic模型（参数校验）
class WeatherQueryParams(BaseModel):
    city: str = Field(..., description="需要查询天气的城市名称")
    date: Optional[str] = Field(None, description="查询的日期，格式为YYYY-MM-DD，默认是今天")


class CalculatorParams(BaseModel):
    num1: float = Field(..., description="第一个数字")
    num2: float = Field(..., description="第二个数字")
    operation: str = Field(..., description="运算操作，支持+, -, *, /, pow(幂运算)")


class TimeConversionParams(BaseModel):
    time: str = Field(..., description="需要转换的时间，格式为HH:MM")
    from_timezone: str = Field(..., description="原始时区，例如UTC, Asia/Shanghai")
    to_timezone: str = Field(..., description="目标时区，例如UTC, America/New_York")


# 2. 实现工具函数（内部用Pydantic校验）
def get_weather(city: str, date: Optional[str] = None) -> str:
    """查询指定城市和日期的天气情况"""
    try:
        # 用Pydantic模型校验参数
        params = WeatherQueryParams(city=city, date=date)
    except ValidationError as e:
        return f"参数错误：{e}"

    city = params.city
    date = params.date or datetime.date.today().strftime("%Y-%m-%d")

    # 模拟天气数据返回
    weather_conditions = ["晴朗", "多云", "小雨", "阴天"]
    temperature = round(15 + (hash(city + date) % 20) - 10, 1)
    condition = weather_conditions[hash(city + date) % len(weather_conditions)]

    return f"{date} {city}的天气：{condition}，气温{temperature}°C"


def calculate(num1: float, num2: float, operation: str) -> str:
    """进行数学计算"""
    try:
        # 用Pydantic模型校验参数
        params = CalculatorParams(num1=num1, num2=num2, operation=operation)
    except ValidationError as e:
        return f"参数错误：{e}"

    try:
        operations = {
            '+': params.num1 + params.num2,
            '-': params.num1 - params.num2,
            '*': params.num1 * params.num2,
            '/': params.num1 / params.num2 if params.num2 != 0 else "错误：除数不能为零",
            'pow': math.pow(params.num1, params.num2)
        }

        if params.operation not in operations:
            return f"错误：不支持的操作符 {params.operation}"

        result = operations[params.operation]
        return f"{params.num1} {params.operation} {params.num2} = {result}"
    except Exception as e:
        return f"计算出错：{str(e)}"


def convert_time(time: str, from_timezone: str, to_timezone: str) -> str:
    """将时间从一个时区转换到另一个时区"""
    try:
        # 用Pydantic模型校验参数
        params = TimeConversionParams(time=time, from_timezone=from_timezone, to_timezone=to_timezone)
    except ValidationError as e:
        return f"参数错误：{e}"

    try:
        # 简化的时区转换
        hour, minute = map(int, params.time.split(':'))
        tz_offsets = {"UTC": 0, "Asia/Shanghai": 8, "America/New_York": -5}

        if params.from_timezone not in tz_offsets or params.to_timezone not in tz_offsets:
            return f"支持的时区：{list(tz_offsets.keys())}"

        offset_diff = tz_offsets[params.to_timezone] - tz_offsets[params.from_timezone]
        total_minutes = hour * 60 + minute + offset_diff * 60
        total_minutes %= 1440  # 确保在0-1439分钟范围内
        new_hour, new_minute = total_minutes // 60, total_minutes % 60

        return f"{params.from_timezone} {params.time} → {params.to_timezone} {new_hour:02d}:{new_minute:02d}"
    except Exception as e:
        return f"转换出错：{str(e)}"


# 3. 创建工具集合
tools = [
    StructuredTool.from_function(
        func=get_weather,
        name="WeatherQuery",
        description="查询指定城市和日期的天气情况",
        args_schema=WeatherQueryParams  # 关联Pydantic模型用于参数解析
    ),
    StructuredTool.from_function(
        func=calculate,
        name="Calculator",
        description="进行数学计算，支持加、减、乘、除和幂运算",
        args_schema=CalculatorParams
    ),
    StructuredTool.from_function(
        func=convert_time,
        name="TimeConversion",
        description="将时间从一个时区转换到另一个时区",
        args_schema=TimeConversionParams
    )
]

# 4. 初始化LLM
llm = ChatTongyi(model="qwen-turbo", api_key=ALIBABA_API_KEY)

# 5. 创建Agent
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一个能使用工具的助手。根据问题选择合适工具，确保参数正确。"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
        }
        | prompt
        | llm
        | OpenAIFunctionsAgentOutputParser()
)

# 6. 初始化Agent执行器
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    # 测试1：天气查询
    print("测试1：天气查询")
    print(agent_executor.invoke({"input": "上海明天的天气怎么样？"}))

    # 测试2：数学计算
    print("\n测试2：数学计算")
    print(agent_executor.invoke({"input": "3的4次方是多少？"}))

    # 测试3：时区转换
    print("\n测试3：时区转换")
    print(agent_executor.invoke({"input": "现在是北京时间14:30，对应的纽约时间是多少？"}))