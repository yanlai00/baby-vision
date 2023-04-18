# Transforms the toybox dataset into images (as in the classification benchmark of the baby-vision paper).

import os
import os.path as osp
import argparse
import numpy as np
from decord import VideoReader
from decord import cpu
import math
import cv2
import time

parser = argparse.ArgumentParser(description='Read SAYCam videos')
parser.add_argument('data', metavar='DIR', help='path to toybox videos')
parser.add_argument('--split', type=str, help='which split method to use. Choices: [iid, exemplar]')
parser.add_argument('--save-dir', default='', type=str, help='save directory')
parser.add_argument('--fps', default=1, type=int, help='sampling rate (frames per second)')

if __name__ == '__main__':

    args = parser.parse_args()

    os.mkdir(args.save_dir)
    train_save_dir = osp.join(args.save_dir, 'train')
    val_save_dir = osp.join(args.save_dir, 'val')
    os.mkdir(train_save_dir)
    os.mkdir(val_save_dir)

    all_objects = set()
    num_folders_per_object = 30

    categories = os.listdir(args.data) # 'animals', 'households', 'vehicles' 

    for category in categories:
        category_dir = osp.join(args.data, category)
        video_folders = os.listdir(category_dir)
        video_folders.sort()
        for video_folder in video_folders:
            object_name, object_id, _ = video_folder.split('_')
            object_train_save_dir = osp.join(train_save_dir, object_name)
            object_val_save_dir = osp.join(val_save_dir, object_name)
            if object_name not in all_objects:
                all_objects.add(object_name)
                os.mkdir(object_train_save_dir)
                os.mkdir(object_val_save_dir)
                img_counter = 0
            video_folder_dir = osp.join(category_dir, video_folder)
            file_list = os.listdir(video_folder_dir)
            
            final_size = 224
            resized_minor_length = 256

            t1 = time.time()

            for file_indx in file_list:
                file_name = os.path.join(video_folder_dir, file_indx)
                # print(file_name)

                try:
                    vr = VideoReader(file_name, ctx=cpu(0))
                    frame_count = len(vr)
                    frame_rate = math.floor(vr.get_avg_fps())

                    # print(f"frame rate {frame_rate}") # 29 or 30

                    # take every sample_rate frames (30: 1fps, 15: 2fps, 10: 3fps, 6: 5fps, 5: 6fps, 3: 10fps, 2: 15fps, 1: 30fps)
                    sample_rate = frame_rate // args.fps + 1

                    frame_indices = list(range(0, frame_count, sample_rate))

                    frames = vr.get_batch(frame_indices).asnumpy()
                except RuntimeError:
                    print(f'ERROR!!! Skipping {file_name}')
                    continue

                frame_height, frame_width = frames.shape[1], frames.shape[2]

                # print(f"frame height {frame_height}, frame width {frame_width}") # 1080 * 1920

                # Resize
                new_height = frame_height * resized_minor_length // min(frame_height, frame_width)
                new_width = frame_width * resized_minor_length // min(frame_height, frame_width)

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

                    if args.split == 'iid':
                        training = np.random.random() < 0.5
                    elif args.split == 'exemplar':
                        training = int(object_id) <= (num_folders_per_object * 9 // 10)

                    if training:
                        cv2.imwrite(os.path.join(object_train_save_dir, 'img_{:04d}.jpeg'.format(img_counter)), cropped_frame[:, :, ::-1])
                    else:
                        cv2.imwrite(os.path.join(object_val_save_dir, 'img_{:04d}.jpeg'.format(img_counter)), cropped_frame[:, :, ::-1])

                    img_counter += 1

            t2 = time.time()        
        
            print(f'Finished folder {video_folder}, Time {(t2 - t1):.2f} seconds')

