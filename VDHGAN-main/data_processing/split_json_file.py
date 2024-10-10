import json
import random

def split_json_file(input_file, train_file, val_file, test_file, train_ratio, val_ratio):
    # 读取原始JSON文件
    with open(input_file, 'r') as f:
        data = json.load(f)

    # 随机打乱数据
    random.shuffle(data)

    # 计算划分的索引位置
    total_length = len(data)
    train_length = int(total_length * train_ratio)
    val_length = int(total_length * val_ratio)

    # 划分数据集
    train_data = data[:train_length]
    val_data = data[train_length:train_length+val_length]
    test_data = data[train_length+val_length:]

    # 写入划分后的JSON文件
    with open(train_file, 'w') as f:
        json.dump(train_data, f)
    with open(val_file, 'w') as f:
        json.dump(val_data, f)
    with open(test_file, 'w') as f:
        json.dump(test_data, f)


dataset = "devign"
input_file = ''
train_file = ''
test_file = ''
val_file = ''
# 设置输入文件路径和输出文件路径
if dataset == "devign":
    input_file = '../devign_dataset/devign_raw_code/function.json'
    train_file = '../devign_dataset/devign_data_split_2/train_raw_code.json'
    val_file = '../devign_dataset/devign_data_split_2/valid_raw_code.json'
    test_file = '../devign_dataset/devign_data_split_2/test_raw_code.json'
elif dataset == "Reveal":
    input_file = '../Reveal_dataset/Reveal_raw_code/Reveal_dataset.json'
    train_file = '../Reveal_dataset/Reveal_data_split/train_raw_code.json'
    val_file = '../Reveal_dataset/Reveal_data_split/valid_raw_code.json'
    test_file = '../Reveal_dataset/Reveal_data_split/test_raw_code.json'
elif dataset == "Fan":
    input_file = '../Fan_dataset/Fan_raw_code/Fan_dataset.json'
    train_file = '../Fan_dataset/Fan_data_split/train_raw_code.json'
    val_file = '../Fan_dataset/Fan_data_split/valid_raw_code.json'
    test_file = '../Fan_dataset/Fan_data_split/test_raw_code.json'

# 设置训练集、验证集和测试集的比例
train_ratio = 0.8  # 80% 用于训练集
val_ratio = 0.1  # 10% 用于验证集，剩下的 10% 自动分配给测试集

# 调用函数进行划分
split_json_file(input_file, train_file, val_file, test_file, train_ratio, val_ratio)
