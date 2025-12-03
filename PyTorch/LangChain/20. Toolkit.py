import os
import ast
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.tools import Tool
from langchain_community.agent_toolkits.file_management.toolkit import FileManagementToolkit
from langchain_community.chat_models import ChatTongyi
from settings import ALIBABA_API_KEY

# 初始化大模型
llm = ChatTongyi(model="qwen-turbo", api_key=ALIBABA_API_KEY)

# 1. 文件管理工具包
file_dir = "./file_demo"
os.makedirs(file_dir, exist_ok=True)
file_tools = FileManagementToolkit(root_dir=file_dir).get_tools()


# 2. 自定义计算工具
def calculate_list_average(input_data: str) -> float:
    try:
        # 步骤1：将字符串解析为Python列表（使用ast.literal_eval安全解析）
        numbers = ast.literal_eval(input_data)
        # 步骤2：校验是否为列表且元素都是数字
        if not isinstance(numbers, list):
            raise ValueError("输入必须是列表形式（如'[1,2,3]'）")
        # 转换所有元素为数字（处理可能的字符串数字，如"5"→5）
        numbers = [float(n) for n in numbers]
        # 步骤3：计算平均值
        return sum(numbers) / len(numbers) if numbers else 0.0
    except Exception as e:
        return f"计算失败：{str(e)}"  # 返回错误信息，方便Agent调试


calculator_tool = Tool(
    name="ListAverageCalculator",
    func=calculate_list_average,
    description="计算列表中数字的平均值，输入应为字符串形式的列表（如'[1,3,5]'），工具会自动解析并计算。"
)
custom_tools = [calculator_tool]

# 3. 初始化Agent
agent = initialize_agent(
    file_tools + custom_tools,
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False  # 如需调试，设为True查看工具调用过程
)

# 执行任务
if __name__ == "__main__":
    # 1. 创建文件（确保内容是字符串形式的列表）
    agent.invoke(f"在{file_dir}下创建data.txt，内容写'[2,4,6,8,10]'")

    # 2. 读取文件并计算平均值（明确要求将读取的字符串传给计算工具）
    res = agent.invoke(f"读取{file_dir}/data.txt的内容，将内容作为输入传给ListAverageCalculator工具计算平均值")
    print("平均值结果：", res)  # 预期输出：6.0

    # 3. 删除文件
    agent.invoke(f"删除{file_dir}/data.txt")