import base64
from zhipuai import ZhipuAI

# 配置API密钥和模型编码
API_KEY = "9c54b7a87cc94de3b1a2e41ae4cd0074.LZOvOWuKp41T1Iig"  # 替换为你的API密钥
MODEL_CODE = "glm-4v-flash"  # 替换为你想使用的模型编码


def call_glm4_api(student_answer, reference_answer, score):
    """
    调用GLM-4 API来评分学生答案。

    参数:
        student_answer (str): 学生的答案文本。
        reference_answer (str): 参考答案文本。

    返回:
        dict: 包含得分和建议的响应。
    """
    # 配置API客户端
    client = ZhipuAI(api_key=API_KEY)

    # 构造消息列表
    messages = [
        {"role": "system", "content": "你是一个评分助手，你的任务是根据上传的图片提取出学生的答案，再根据参考答案对学生的答案进行评分，并提供改进建议。"
                                      "评分要求，对于选择题和填空题，若答案一致，则给满分。对于其他题型，则要细致的分析。"},
        {"role": "user", "content": f"上传的图片: {student_answer}"},
        {"role": "user", "content": f"参考答案: {reference_answer}"},
        {"role": "user", "content": f"请给出你识别到的学生答案、一个分数（满分为{score}分）和改进建议。"},
    ]

    try:
        # 调用GLM-4模型
        response = client.chat.completions.create(
            model=MODEL_CODE,
            messages=messages,
        )

        # 提取模型返回的内容
        if response.choices:
            return response.choices[0].message.content
        else:
            print("API返回结果中未找到评分内容。")
            return None
    except Exception as e:
        print(f"调用API时发生错误: {e}")
        return None


def main():
    # 示例学生答案和参考答案
    img_path = 'test_images/test7.jpg'
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')
    reference_answer = "7"
    score = 4

    # 调用API进行评分
    result = call_glm4_api(img_base, reference_answer, score)

    if result:
        print("模型反馈:")
        print(result)
    else:
        print("无法获取模型反馈，请检查API配置或网络连接。")


if __name__ == "__main__":
    main()