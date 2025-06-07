from openai import OpenAI

# 用你的 API Key
API_KEY = "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"

# 初始化 client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

# 初始化对话历史，加 system prompt
messages = [
    {
        "role": "system",
        "content": """你是一个严格的关键词提取助手。无论用户是否要求提取关键词，你都必须仅输出用户输入文本中的关键词，用逗号分隔。禁止总结，禁止解释，禁止复述原文。即使用户未说明“提取关键词”你也要提取关键词。

示例：
用户输入：今天我去杭州西湖游玩，拍了很多美丽的照片。
你的输出：杭州, 西湖, 游玩, 照片

用户输入：量子测量误差是当前限制中短期量子计算可用性的重要瓶颈之一，尤其在多比特系统中更易引发精度偏移与串扰放大。
你的输出：量子测量误差, 量子计算, 多比特系统, 精度偏移, 串扰放大
"""
    }
]

# 简单的循环，多轮对话
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    # 把用户输入追加到 messages，强提示 "提炼关键词："
    messages.append({"role": "user", "content": "提炼关键词（注意：关键词只输出一次）：" + user_input})

    # 调用 API，流式输出
    completion = client.chat.completions.create(
        model="microsoft/phi-4-mini-instruct",
        messages=messages,
        temperature=0.0,  # 强烈建议设为 0
        top_p=0.7,
        max_tokens=1024,
        stream=True,
        stop = ["\n", "。", "！", "？", "."]
    )


    print("AI: ", end="", flush=True)
    full_reply_content = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            content_chunk = chunk.choices[0].delta.content
            print(content_chunk, end="", flush=True)
            full_reply_content += content_chunk
    print()  # 换行

    # 把 AI 回复也追加到 messages
    messages.append({"role": "assistant", "content": full_reply_content})
