# _*_coding :utf-8 _*_
# @Time : 2024/7/19 10:17
# @Author : William niu
# @File : new_integration
# @Project : OnlineResearch_NLP
# _*_coding :utf-8 _*_
# @Time : 2024/7/9 10:00
# @Author : William niu
# @File : gpt3.5pay
# @Project : OnlineResearch_NLP
import pandas as pd
import os
import torch
import json
import openai
import re
import glob
import faiss
import httpx
import requests
import numpy as np

from transformers import BertTokenizer, BertModel

PROXY = "http://127.0.0.1:7890"

os.environ['HTTP_PROXY'] = os.environ['http_proxy'] = PROXY
os.environ['HTTPS_PROXY'] = os.environ['https_proxy'] = PROXY
os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost,.local'

client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.gpt.ge/v1"
    # base_url="https://gf.gpt.ge/v1/"
)
openai.default_headers = {"x-foo": "true"}


#COT
# Rule_file = r"E:\UIBE\OnlineResearch_NLP\LLMRAG\Experiment\llamaindex_RAG\Code\Rules.xlsx"
# Weight_file = r"E:\UIBE\OnlineResearch_NLP\LLMRAG\Experiment\llamaindex_RAG\Code\Weight.xlsx"
# Rule_text = pd.read_excel(Rule_file, header=0)
# Weight_text = pd.read_excel(Weight_file, header=0)
# Rule_data = Rule_text.values  # 舍弃行号
# Weight_data = Weight_text.values
# Rules = {}
# num = 0
# for i in range(0, 5):
#     Rules[Rule_data[0][i]] = {}
#     for j in range(1, len(Rule_data)):
#         if Rule_data[j][i] is np.nan:
#             break
#         Rules[Rule_data[0][i]][Rule_data[j][i]] = Weight_data[num][0]
#         num += 1
# print(Rules)
Rule_file = r"E:\UIBE\OnlineResearch_NLP\LLMRAG\Experiment\llamaindex_RAG\Code\Rules.xlsx"
Weight_file = r"E:\UIBE\OnlineResearch_NLP\LLMRAG\Experiment\llamaindex_RAG\Code\Weight.xlsx"
Rule_text = pd.read_excel(Rule_file, header=0)
Weight_text = pd.read_excel(Weight_file, header=0)
Rule_data = Rule_text.values  # 舍弃行号
Weight_data = Weight_text.values

Rules = []

num = 0

# 遍历每一列并将每一列的所有键值对存储为一个列表
for i in range(0, 5):
    rule = []
    for j in range(1, len(Rule_data)):
        if pd.isna(Rule_data[j][i]):
            break
        key = Rule_data[j][i]
        value = Weight_data[num][0]
        rule.append({key: value})
        num += 1
    # 将该列的列表添加到总列表中
    Rules.append(rule)

print(Rules)
# completion = openai.chat.completions.create(
#     model="gpt-3.5-turbo-0125",
#     messages=[
#         {
#             "role": "user",
#             "content": "langchain可以读取xlsx文件吗",
#         },
#     ],
# )
# print(completion.choices[0].message.content)

# def generate_rating_prompt(row, dimension):
#     if {dimention} is
#     """Generate a simple request prompt for each row"""
#     row_description = ", ".join(f"{col}: {row[col]}" for col in row.index if pd.notna(row[col]))
#     return f"Please assess the company's {dimension} based on the provided data and explain the rating reasons solely in Chinese(limited to 20 words). Expected answers: 低, 较低, 中, 较高, or 高. Information included: {row_description}."
index_file_path = "E:\\UIBE\\OnlineResearch_NLP\\LLMRAG\\Experiment\\faiss_index.index"
faiss_index = faiss.read_index(index_file_path)
folder_path =  r'E:\UIBE\OnlineResearch_NLP\LLMRAG\Experiment\llamaindex_RAG\json\源数据-编码-json'
json_files = glob.glob(os.path.join(folder_path, '*.json'))

all_documents = []

# 提取文本内容的函数
def extract_case_text(case_dict):
    return " ".join([f"{key}: {value}" for key, value in case_dict.items()])

# 遍历每一个 JSON 文件
for json_file in json_files:
    try:
        with open(json_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        for key, value in entry.items():
                            if isinstance(value, list):
                                for case in value:
                                    if isinstance(case, dict):
                                        case_text = extract_case_text(case)
                                        all_documents.append(case_text)
                    else:
                        print(f"跳过不符合预期结构的条目: {entry}")
            else:
                print(f"文件 {json_file} 的内容不符合预期：预期为列表形式。")
    except json.JSONDecodeError as e:
        print(f"JSON 解码错误: {e} 文件: {json_file}")
    except Exception as e:
        print(f"处理文件 {json_file} 时出现错误: {e}")

def generate_rating_prompt(row, dimension):
    """Generate a simple request prompt for each row"""
    row_description = ", ".join(f"{col}: {row[col]}" for col in row.index if pd.notna(row[col]))
    company_code = row["企业名称"]
    if dimension in ["科技创新能力", "合规遵从能力", "财务资金能力"]:
        tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
        model_bert = BertModel.from_pretrained('bert-base-uncased')
        def vectorize_query(query):
            inputs = tokenizer_bert(query, return_tensors='pt', padding=True, truncation=True, max_length=128)
            model_bert.eval()
            with torch.no_grad():
                outputs = model_bert(**inputs)
                query_vector = outputs.last_hidden_state[:, 0, :].numpy()[0]
            return query_vector
        
        query = "企业编码为{company_code}的企业{dimension}如何?"
        query_vector = vectorize_query(query)
        def search_faiss(query_vector, faiss_index, k=5):
            query_vector = np.array([query_vector]).astype('float32')
            distances, indices = faiss_index.search(query_vector, k)
            return distances, indices

        distances, indices = search_faiss(query_vector, faiss_index, k=5)

        def get_context_documents(indices, all_documents):
            context_docs = [all_documents[i] for i in indices[0]]
            return context_docs

        # 假设 `all_documents` 是原始文档的列表
        context_documents = get_context_documents(indices, all_documents)
        context_text = " ".join(context_documents)
        return f"现在我给你提供一些背景信息：{context_text}和评价规则{rule}，评价规则中的Key是企业{dimension}方面的指标，如果Key的value是”重要“，则代表该Key对于可以用于评估公司该维度的属性。每一个属性的value，代表该属性对评价影响的权重。请根据提供的数据 {row_description} 和背景信息来评估该公司的 {dimension}, 回答必须是以下五种等级之一：“低”、“较低”、“中”、“较高 ”或 “高”。紧接着说明评级理由（限 30 字以内），评级理由中可以适当加入背景信息中的细节来辅助说明。请务必用中文回答。"
    else:
        return f"请根据所提供的数据{row_description}对公司的{dimension}进行评估，回答必须是以下五种等级之一：“低”、“较低”、“中”、“较高 ”或 “高”。紧接着说明评级理由（限 30 字以内），请务必用中文回答。"


def get_gpt_rating_and_reason(client, prompt, retries=0):
    """Obtain a rating result and its reason from GPT-3.5"""
    if retries > 2:
        raise Exception("多次尝试后仍无法确定有效评级。")
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-3.5-turbo",
    )
    response_text = chat_completion.choices[0].message.content.strip()
    print(response_text)
    allowed_ratings = ["较低", "低", "中", "较高", "高"]
    for rating in allowed_ratings:
        pattern = re.escape(rating) + r'(.*)'
        match = re.search(pattern, response_text)
        if match:
            reason = match.group(1).strip(" ,.;:，。；：")
            filtered_reason = re.sub(r'[^\u4e00-\u9fff，。；：、]', '', reason)
            return rating, filtered_reason
            break
    if retries <= 2:
        return get_gpt_rating_and_reason(client, prompt, retries + 1)
    return "无法确定评级", "未能提供原因"

def process_excel(file_path, output_file_path):
    data = pd.read_excel(file_path)
    results = pd.DataFrame()
    results["企业编码"] = data["企业名称"]

    dimensions = [
        "科技创新能力",
        "质量管控能力",
        "生产服务能力",
        "财务资金能力",
        "合规遵从能力"
    ]
    reasons = []

    for dimension,rule in zip(dimensions,Rules):
        results[dimension], temp_reasons = zip(*data.apply(
            lambda row: get_gpt_rating_and_reason(
                client,
                generate_rating_prompt(row, dimension)
            ), axis=1
        ))
        reasons.append(temp_reasons)

    # Consolidate all reasons into column G
    results["评分原因"] = ['; '.join(r) for r in zip(*reasons)]

    results.to_excel(output_file_path, index=False)

# File paths
file_path = r"E:\UIBE\OnlineResearch_NLP\LLMRAG\Experiment\llamaindex_RAG\Data_Source\完整数据集-脱敏-201-300.xlsx"
output_file_path = r"E:\UIBE\OnlineResearch_NLP\LLMRAG\Experiment\llamaindex_RAG\Data_Source\Results\完整数据集-脱敏(GPT英文版第一次尝试)201-300.xlsx"

# Process the Excel file
process_excel(file_path, output_file_path)



















