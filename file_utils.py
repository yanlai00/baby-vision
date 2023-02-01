import os
from shutil import move
import os.path as osp

def flatten_imagefolder(root_dir):
    level1_dirs = os.listdir(root_dir)
    for level1_dir in level1_dirs:
        abs_level1_dir = osp.join(root_dir, level1_dir)
        level2_dirs = os.listdir(abs_level1_dir)
        for level2_dir in level2_dirs:
            src_dir = osp.join(abs_level1_dir, level2_dir)
            dst_dir = osp.join(root_dir, f'{level1_dir}_{level2_dir}')
            move(src_dir, dst_dir)

if __name__ == "__main__":
    flatten_imagefolder('/mnt/wwn-0x5000c500e421004a/yy2694/datasets/saycam/tc/val/')

