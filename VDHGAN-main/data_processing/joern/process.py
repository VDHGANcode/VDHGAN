import os
import json
import time
import subprocess
import datetime
import signal
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from tqdm import trange
from ...data_test import process_subfolders

def process(file_path, start, end):
    '''
    frame = pd.read_json(file_path, lines=True)
    files = list(frame['file_name'])
    timeout = 5
    '''
    i = start
    timeout = 1500
    files = os.listdir(file_path)
    print(len(files))
    if end > len(files):
        end = len(files)
    with tqdm(total=end - start, desc=f'Processing files {start + 1}/{end}', unit='file') as pbar:
        while i < end:
            #slicer = "bash ./slicer.sh " + file_path + "  " + str(files[i]) + "  1 " + "parsed/" + str(files[i])
            slicer = "bash ./slicer.sh " + file_path + "  " + str(files[i]) + "  1 " + "./parsed/" + str(
                files[i]).rstrip(".c")
            start0 = datetime.datetime.now()
            process1 = subprocess.Popen(slicer, shell=True)
            while process1.poll() is None:
                #time.sleep(0.2)
                end0 = datetime.datetime.now()
                if (end0 - start0).seconds > timeout:
                    os.kill(process1.pid, signal.SIGKILL)
                    os.waitpid(-1, os.WNOHANG)
            pbar.update(1)
            i += 1

    # for i in trange(end - start, desc=f'Processing files {start + 1}/{end}', unit='file'):
    #     # slicer = "bash ./slicer.sh " + file_path + "  " + str(files[i]) + "  1 " + "parsed/" + str(files[i])
    #     slicer = "bash ./slicer.sh " + file_path + "  " + str(files[i]) + "  1 " + "./parsed/" + str(
    #         files[i]).rstrip(".c")
    #     start0 = datetime.datetime.now()
    #     process1 = subprocess.Popen(slicer, shell=True)
    #     while process1.poll() is None:
    #         # time.sleep(0.2)
    #         end0 = datetime.datetime.now()
    #         if (end0 - start0).seconds > timeout:
    #             os.kill(process1.pid, signal.SIGKILL)
    #             os.waitpid(-1, os.WNOHANG)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--file_path', help='funtions dic.', default='./output_func3')
    parser.add_argument('--file_path', help='funtions dic.', default='./Fan_func')
    parser.add_argument('--start', help='start functions number to parsed', type=int, default=107680) #130424
    parser.add_argument('--end', help='end functions number to parsed', type=int, default=300000) #default=4500
    args = parser.parse_args()
    file_path = args.file_path
    start = args.start
    end = args.end
    files = os.listdir(file_path)
    #path_1 = './output_example'
    process(file_path, start, end)

    # 指定devign_parsed文件夹的路径
    devign_parsed_folder_path = "../devign_dataset/devign_parsed"
    Reveal_parsed_folder_path = "../Reveal_dataset/Reveal_parsed"
    Fan_parsed_folder_path = "../../joern_2/Fan_parsed"

    process_subfolders(devign_parsed_folder_path)

