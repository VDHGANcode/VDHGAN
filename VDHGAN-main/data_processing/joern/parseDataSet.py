import csv
import os
import json
import pandas as pd
from tqdm import tqdm
from tqdm import trange
import sys

def devign_export_func_to_files(csv_file):
    output_folder = './output_func2'  # 输出文件夹名称
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹，如果不存在的话
    # 设置字段大小限制
    csv.field_size_limit(1000000)  # 设置较大的字段大小限制
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            project, commit_id, target, func = row
            filename = f'{commit_id}.c'
            file_path = os.path.join(output_folder, filename)  # 构建文件的完整路径
            with open(file_path, 'w') as func_file:
                func_file.write(func)


def Reveal_export_func_to_files(output_folder):
    non_vul_file = './Reveal_dataset/non-vulnerables.json'  # 替换为你的JSON文件路径
    vul_file = './Reveal_dataset/vulnerables.json'
    with open(non_vul_file, encoding='utf-8') as file1:
        non_vul_data = json.load(file1)
    with open(vul_file, encoding='utf-8') as file2:
        vul_data = json.load(file2)
    count_non = 0
    count_vul = 0
    all_new_data = []

    for record in tqdm(non_vul_data):
        #commit_id = record['hash']
        count_non += 1
        func = record['code']
        filename = f"non_vul_sample_{count_non}.c"
        filepath = os.path.join(output_folder, filename)
        data = {
            'graph_name': f"non_vul_sample_{count_non}",
            'func': func,
            'target': 0
        }
        all_new_data.append(data)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(func)
    for record in tqdm(vul_data):
        #commit_id = record['hash']
        count_vul += 1
        func = record['code']
        filename = f"vul_sample_{count_vul}.c"
        filepath = os.path.join(output_folder, filename)
        data = {
            'graph_name': f"vul_sample_{count_vul}",
            'func': func,
            'target': 1
        }
        all_new_data.append(data)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(func)
    output_path = open('./Reveal_dataset/Reveal_dataset.json', 'w')
    json.dump(all_new_data, output_path)


def Fan_export_func_to_files(output_file):
    # 读取CSV文件
    data = pd.read_csv("F:\DataSet\MSR_data_cleaned\MSR_data_cleaned.csv")

    # 创建空列表存储处理后的数据
    processed_data = []
    id_counter = 1

    # 遍历每行数据
    for index, record in tqdm(data.iterrows()):
        # 检查漏洞是否为0
        if record['vul'] == 0:
            filepath = os.path.join(output_file, f"non_vul_{id_counter}.c")
            # 记录func_before和目标为0的数据
            data = {'graph_name': f"non_vul_{id_counter}",
                    'func': record['func_before'], 'target': 0}
            processed_data.append(data)
            id_counter += 1
            #with open(filepath, 'w', encoding='utf-8') as file:
            #    file.write(record['func_before'])
        elif record['vul'] == 1:
            filepath1 = os.path.join(output_file, f"vul_{id_counter}.c")
            filepath2 = os.path.join(output_file, f"non_vul_{id_counter}.c")
            # 记录func_before和目标为1的数据
            data1 = {'graph_name': f"non_vul_{id_counter}",
                     'func': record['func_before'], 'target': 1}
            processed_data.append(data1)
            id_counter += 1
            # with open(filepath1, 'w', encoding='utf-8') as file:
            #     file.write(record['func_before'])
            #
            # # 记录func_after和目标为0的数据
            # data2 = {'graph_name': f"non_vul_{id_counter}",
            #          'func': record['func_after'], 'target': 0}
            # processed_data.append(data2)
            # id_counter += 1
            # with open(filepath2, 'w', encoding='utf-8') as file:
            #     file.write(record['func_after'])

    # 将处理后的数据存储为JSON文件
    output_path = open('./Fan_dataset/Fan_dataset_2.json', 'w')
    json.dump(processed_data, output_path)
    print('samples_count:', id_counter)


def save_func_to_file(data, output_folder):
    for record in data:
        commit_id = record['commit_id']
        func = record['func']
        filename = f"{commit_id}.c"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, 'w') as file:
            file.write(func)
# 使用示例
#csv_file = './functioncsv/Devign_utf8.csv'  # 替换为您的CSV文件路径
#export_func_to_files(csv_file)

#修改为实验数据集
dataset = 'devign'

if dataset == 'devign':
    json_file = './function/function.json'  # 替换为你的JSON文件路径
    with open(json_file) as file:
        data = json.load(file)
    output_folder = './output_func3'  # 替换为你想要保存文件的文件夹路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    save_func_to_file(data, output_folder)
elif dataset == 'Reveal':
    output_folder = './Reveal_func'  # 替换为你想要保存文件的文件夹路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    Reveal_export_func_to_files(output_folder)
elif dataset == 'Fan':
    output_folder = './Fan_func_2'  # 替换为你想要保存文件的文件夹路径
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    Fan_export_func_to_files(output_folder)
