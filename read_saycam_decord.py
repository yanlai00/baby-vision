# This file has a similar function as read_saycam.py, but might be a little bit faster with the 
# 'decord' video decoding package. Install here: https://github.com/dmlc/decord (the CPU version should suffice).

import os
import argparse
import numpy as np
from decord import VideoReader
from decord import cpu
import math
import cv2
import time

parser = argparse.ArgumentParser(description='Read SAYCam videos')
parser.add_argument('data', metavar='DIR', help='path to SAYCam videos')
parser.add_argument('--partition', default='SAY', type=str, help='which partition to process. Choices: [S, A, Y, SAY]')
parser.add_argument('--save-dir', default='', type=str, help='save directory')
parser.add_argument('--fps', default=5, type=int, help='sampling rate (frames per second)')
parser.add_argument('--seg-len', default=288, type=int, help='segment length (seconds)')

if __name__ == '__main__':

    args = parser.parse_args()

    file_list = os.listdir(args.data)
    if args.partition == 'S':
        file_list = [file for file in file_list if file.startswith('S_')]
    elif args.partition == 'A':
        file_list = [file for file in file_list if file.startswith('A_')]
    elif args.partition == 'Y':
        file_list = [file for file in file_list if file.startswith('Y_')]
    else:
        assert args.partition == 'SAY'

    file_list.sort()

    class_counter = 0
    img_counter = 0
    file_counter = 0
    
    final_size = 224
    resized_minor_length = 256
    edge_filter = False
    n_imgs_per_class = args.seg_len * args.fps

    curr_dir_name = os.path.join(args.save_dir, 'class_{:04d}'.format(class_counter))
    os.mkdir(curr_dir_name)

    for file_indx in file_list:
        file_name = os.path.join(args.data, file_indx)
        print(file_name)

        try:
            vr = VideoReader(file_name, ctx=cpu(0))
            frame_count = len(vr)
            frame_rate = math.floor(vr.get_avg_fps())

            # print('Total frame count: ', frame_count)
            # print('Native frame rate: ', frame_rate)

            # take every sample_rate frames (30: 1fps, 15: 2fps, 10: 3fps, 6: 5fps, 5: 6fps, 3: 10fps, 2: 15fps, 1: 30fps)
            sample_rate = frame_rate // args.fps + 1

            frame_indices = list(range(0, frame_count, sample_rate))

            a = time.time()
            frames = vr.get_batch(frame_indices).asnumpy()
            b = time.time()
            # print('Data Loading Time:', b-a)
        except:
            print('ERROR!!! SKIPPING')
            continue

        frame_height, frame_width = frames.shape[1], frames.shape[2]

        # Resize
        new_height = frame_height * resized_minor_length // min(frame_height, frame_width)
        new_width = frame_width * resized_minor_length // min(frame_height, frame_width)

        flag = True

        c = time.time()

        for frame in frames:

            # Resize
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

            # Crop
            height, width, _ = resized_frame.shape
            startx = width // 2 - (final_size // 2)
            starty = height // 2 - (final_size // 2) - 16
            cropped_frame = resized_frame[starty:starty + final_size, startx:startx + final_size]
            assert cropped_frame.shape[0] == final_size and cropped_frame.shape[1] == final_size, \
                (cropped_frame.shape, height, width)

            if edge_filter:
                cropped_frame = cv2.Laplacian(cropped_frame, cv2.CV_64F, ksize=5)
                img_min = cropped_frame.min()
                img_max = cropped_frame.max()
                cropped_frame = np.uint8(255 * (cropped_frame - img_min) / (img_max - img_min))

            cv2.imwrite(os.path.join(curr_dir_name, 'img_{:04d}.jpeg'.format(img_counter)), cropped_frame[::-1, ::-1, :])
            img_counter += 1

            if img_counter == n_imgs_per_class:
                img_counter = 0
                class_counter += 1
                curr_dir_name = os.path.join(args.save_dir, 'class_{:04d}'.format(class_counter))
                os.mkdir(curr_dir_name)

        d = time.time()
        # print("Data Saving Time:", d-c)

        file_counter += 1
        print('Completed video {:4d} of {:4d}'.format(file_counter, len(file_list)))
