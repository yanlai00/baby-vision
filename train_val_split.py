import os
import argparse
import numpy as np
import math
import random
from shutil import move

parser = argparse.ArgumentParser(description='Read SAYCam videos')
parser.add_argument('--data', metavar='DIR', help='path to SAYCam videos')
parser.add_argument('--save-dir', default='', type=str, help='save directory')
parser.add_argument('--val-ratio', default=0.2, type=float, help='ratio of validation data')

if __name__ == '__main__':

    args = parser.parse_args()

    class_list = os.listdir(args.data)
    class_list.sort()

    for class_idx in class_list:
        src_dir_name = os.path.join(args.data, class_idx)
        dst_dir_name = os.path.join(args.save_dir, class_idx)
        os.mkdir(dst_dir_name)

        file_list = os.listdir(src_dir_name)

        for file_indx in file_list:
            file_name = os.path.join(src_dir_name, file_indx)
            random_number = random.random()
            if random_number < 0.2:
                # copy file
                move(file_name, dst_dir_name)
        print(f'finished class {class_idx}')
            
