"""
@File：EnglishAiAgent.py
@Time：2025/2/8 21:23
@Auth：Tr0e
@Github：https://github.com/Tr0e
@Description：借助DeepSeek大模型，自动生成英语单词的助记文档
@版本迭代：
   v1.0 2025/02/08，首版本开发完成
"""
import json
import requests

markdown_prompt = """
## 角色定义
你是一名专业的英语教师，擅长以生动、易懂的方式传授不喜欢记忆的学生如何记住英语单词。

## 工作规则
- 你的方法是通过联想记忆、分解记忆、词根记忆等方式，摆脱枯燥的死记硬背。
- 你将获取到一个单词列表，请逐一分析每个单词的中文意思，然后构思如何才能让学生轻松地记住。

## 具体示例
比如你会让学生这样记住救护车（ambulance）的单词：
1）联想记忆：想象一辆救护车（ambulance）在紧急情况下飞驰而过，发出“俺不能死”（ambulance的谐音）的呼声，这样的联想可以帮助你记住这个单词。
2）分解记忆：将“ambulance”分解为“am”（是）、“bu”（不）、“lan”（拦）、“ce”（车），可以编成一个小故事：“我（am）不（bu）拦（lan）车（ce），因为那是救护车”。

## 响应格式
最后请按照以下格式返回数据：
1）通过json字符串存储你的分析结果，json字符串包含：单词、单词中文释义、音标、最佳的记忆方法；
2）完整的格式示例如下，请务必严格遵守此格式：
{
  response:[
    {
        "word": "ambulance",
        "pronounce": "/ˈæmbjʊləns/",
        "chinese_meaning": "救护车",
        "memory_method": "联想记忆法：想象一辆救护车在紧急情况下飞驰而过，发出'俺不能死'（ambulance的谐音）的呼声"
    },
    {
        "word": "butterfly",
        "pronounce": "/ˈbʌtəflaɪ/",
        "chinese_meaning": "蝴蝶",
        "memory_method": "分解记忆法：有一只蝴蝶，它的翅膀上涂满了黄油（butter），在空中飞（fly）来飞去，非常引人注目"
    }
  ]
}
"""

def siliconflow_api_call_by_block(user_question, api_key, prompt):
    """
    通过硅基流动平台调用deepseek模型（阻塞模式响应）
    """
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [
            {
                "role": "system",
                "content": f"{prompt}"
            },
            {
                "role": "user",
                "content": user_question
            }
        ],
        "stream": False,
        "max_tokens": 4096,
        "stop": ["null"],
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "frequency_penalty": 0.5,
        "n": 1,
        "response_format": {"type": "text"},
        "tools": []
    }
    response = requests.request("POST", url, json=payload, headers=headers)
    # print(response.text)
    if response.status_code == 200:
        json_obj = json.loads(response.text)
        result = json_obj['choices'][0]['message']['content']
        print(f"[+]AI的回答：\n{result}")
        return result
    else:
        print(f"[!]请求失败，状态码：{response.status_code}")
        return None

def generate_markdown(data, filename="words.md"):
    """
    生成一个Markdown文档，将提供的单词数据以表格形式展示。
    """
    # 构建Markdown表格内容
    markdown_content = "# 英语单词助记文档 \n\n"
    markdown_content += "| 单词 | 音标发音 | 单词含义 | 记忆方法 |\n"
    markdown_content += "|------|-----------|-----------------|---------------|\n"
    # 检查data是否为字典，并且包含'response'键
    if isinstance(data, dict) and 'response' in data:
        items = data['response']
    else:
        print("[-]Error: 数据格式不正确，无法找到'response'字段。")
        return
    for item in items:
        # 检查每个item是否为字典
        if isinstance(item, dict):
            markdown_content += f"| {item.get('word', '')} | {item.get('pronounce', '')} | {item.get('chinese_meaning', '')} | {item.get('memory_method', '')} |\n"
        else:
            print(f"[-]Warning: Item is not a dictionary: {item}")
    # 将Markdown内容写入文件
    with open(filename, "w", encoding="utf-8") as file:
        file.write(markdown_content)
    print(f"[+]Success! Markdown文件已生成：{filename}")


def process_words(api_key, prompt):
    # 读取单词文件并分割成列表
    with open("word.txt", "r", encoding="utf-8") as f:
        words = f.read().splitlines()
    # 初始化存储所有结果的列表
    all_data = []
    batch_size = 3  # 每批次最多3个单词，确保不会超出API响应最大长度限制
    # 分批次处理
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        print(f"[+]正在处理第 {i // batch_size + 1} 批单词: {', '.join(batch)}")
        # 拼接当前批次的单词（保持每行一个单词的格式）
        word_block = "\n".join(batch)
        api_response = siliconflow_api_call_by_block(word_block, api_key, prompt)
        if api_response is None:
            print(f"❌第 {i // batch_size + 1} 批处理失败，请检查API密钥和网络连接")
            continue
        try:
            # 清洗并解析JSON响应
            cleaned_response = api_response.strip("```json").strip("```")
            batch_data = json.loads(cleaned_response)
            # 合并到总数据集
            all_data.extend(batch_data["response"])
            print(f"✅第 {i // batch_size + 1} 批处理成功，累计已处理 {len(all_data)} 个单词")
        except json.JSONDecodeError as e:
            print(f"❌第 {i // batch_size + 1} 批JSON解析失败: {str(e)}")
            print("原始响应内容：")
            print(api_response)
    # 生成最终Markdown文件
    if all_data:
        generate_markdown({"response": all_data})
        print(f"\n🎉全部处理完成，共成功处理 {len(all_data)} 个单词")
    else:
        print("\n⚠️未成功处理任何单词")

if __name__ == '__main__':
    process_words("sk-XXX", markdown_prompt)