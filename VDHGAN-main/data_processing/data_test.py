import os
import shutil
from tqdm import tqdm

def process_subfolders(folder_path):
    # 遍历devign_parsed文件夹的子文件夹
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    for subfolder_name in tqdm(subfolders, desc='Processing subfolders', unit='folder'):
        subfolder_path = os.path.join(folder_path, subfolder_name)

        # 检查子文件夹是否为目录
        if os.path.isdir(subfolder_path):
            # 删除devign_parsed/name1/tmp路径下的nodes.csv和edges.csv
            tmp_folder_path = os.path.join(subfolder_path, "tmp")
            nodes_csv_path = os.path.join(tmp_folder_path, "nodes.csv")
            edges_csv_path = os.path.join(tmp_folder_path, "edges.csv")
            if os.path.exists(nodes_csv_path):
                os.remove(nodes_csv_path)
            if os.path.exists(edges_csv_path):
                os.remove(edges_csv_path)

            # 将devign_parsed/name1/tmp/name1路径下的nodes.csv和edges.csv移动到devign_parsedname1/tmp中
            src_nodes_csv_path = os.path.join(subfolder_path, "tmp", subfolder_name + '.c', "nodes.csv")
            src_edges_csv_path = os.path.join(subfolder_path, "tmp", subfolder_name + '.c', "edges.csv")
            dst_nodes_csv_path = os.path.join(subfolder_path, "nodes.csv")
            dst_edges_csv_path = os.path.join(subfolder_path, "edges.csv")
            if os.path.exists(src_nodes_csv_path):
                shutil.move(src_nodes_csv_path, dst_nodes_csv_path)
            if os.path.exists(src_edges_csv_path):
                shutil.move(src_edges_csv_path, dst_edges_csv_path)

            # 删除devign_parsed/tmp
            if os.path.exists(tmp_folder_path):
                shutil.rmtree(tmp_folder_path)

# 指定devign_parsed文件夹的路径
devign_parsed_folder_path = "../devign_dataset/devign_parsed"

# 指定Reveal_parsed文件夹的路径
Reveal_parsed_folder_path = "../Reveal_dataset/Reveal_parsed"

# 指定Reveal_parsed文件夹的路径
#Fan_parsed_folder_path = "../Fan_dataset/Reveal_parsed"
Fan_parsed_folder_path = "../../joern_2/Fan_parsed"

# 调用函数进行处理
process_subfolders(devign_parsed_folder_path)
