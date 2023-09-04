from math import pi
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import json
from collections import defaultdict

def get_target_seq_frames(samples_txt, target_seq, exclude_seq_frame_json=None):
    seq_frames = defaultdict(list)
    with open(samples_txt, 'r') as sample_txt_file:
        lines = sample_txt_file.readlines()
    for line in lines:
        seq, frame = line.split(',')
        if not(int(seq) in target_seq):
            continue
        frame = frame.split('_')[1].split('.')[0]
        if not exclude_seq_frame_json is None:
            if not(int(frame) in exclude_seq_frame_json[seq]):
                seq_frames[seq].append(frame)
        else:
            seq_frames[seq].append(frame)
    return  seq_frames


def get_rdr_lidar_frame_difference(data_root, target_seqs):
    rdr_lidar_frame_difference = {}
    for seq in target_seqs:
        with open(os.path.join(data_root, str(seq), 'info_calib', 'calib_radar_lidar.txt'), 'r') as f:
            line = f.readlines()[1].strip()
        rdr_lidar_frame_difference[str(seq)] = int(line.split(',')[0])
    return rdr_lidar_frame_difference

    


def get_labels(seq_frames, ds_root, rdr_lidar_frame_difference):
    new_labels = []
    for seq, frames in seq_frames.items():
        print(f'Now processing {seq}')
        for frame_name in tqdm(frames):
            label_path = os.path.join(ds_root, seq, 'label', f'{frame_name}.json')
            if not os.path.exists(label_path):
                print(f'{frame_name} has no label')
                continue
            with open(label_path, 'r') as label_file:
                label = json.load(label_file)
            objs = []
            for obj in label:
                if obj is None:
                    continue
                psr = obj['psr']
                xyz = list(psr['position'].values())
                euler = list(psr['rotation'].values())
                lwh = list(psr['scale'].values())
                converted_obj = {'obj_id': obj['obj_id'], 'obj_type': obj['obj_type'], 'euler': euler, 'xyz': xyz, 'lwh': lwh}
                objs.append(converted_obj)
            rdr_frame = '{:0=5}'.format(int(frame_name) + rdr_lidar_frame_difference[seq])
            label_frame = {'seq': seq, 'frame': frame_name, 'rdr_frame': rdr_frame, 'objs': objs}
            new_labels.append(label_frame)
    return new_labels


def get_gt_viz_format(seq_root, label_dir='label'):
    out_label_file = '/mnt/ssd1/kradar_dataset/viz/kradar_gt_viz_format.json'
    '''
     format: 
      seq:{
        frame:[]
      }
    '''
    all_seq_label = defaultdict(dict)
    for seq in tqdm(sorted(os.listdir(seq_root))):
        seq_path = os.path.join(seq_root, seq)
        if not os.path.isdir(seq_path):
            continue
        label_path = os.path.join(seq_path, label_dir)
        if not os.path.isdir(label_path):
            continue
        for label_file in sorted(os.listdir(label_path)):
            label_file_path = os.path.join(label_path, label_file)
            with open(label_file_path, 'r') as f:
                objs = json.load(f)
            new_objs = []
            for obj in objs:
                if obj is None:
                    continue
                keys = list(obj.keys())
                for key in keys:
                    if key not in ['obj_id', 'obj_type', 'psr']:
                        del obj[key]
                obj['score'] = 1.0
                new_objs.append(obj)
            frame = label_file.split('.')[0]
            all_seq_label[seq][frame] = new_objs
    with open(out_label_file, 'w') as f:
        json.dump(dict(all_seq_label), f, indent=2)


def get_gt_viz_format(seq_root, label_dir='label'):
    out_label_file = '/mnt/ssd1/kradar_dataset/viz/kradar_gt_viz_format.json'
    '''
     format: 
      seq:{
        frame:[]
      }
    '''
    all_seq_label = defaultdict(dict)
    for seq in tqdm(sorted(os.listdir(seq_root))):
        seq_path = os.path.join(seq_root, seq)
        if not os.path.isdir(seq_path):
            continue
        label_path = os.path.join(seq_path, label_dir)
        if not os.path.isdir(label_path):
            continue
        for label_file in sorted(os.listdir(label_path)):
            label_file_path = os.path.join(label_path, label_file)
            with open(label_file_path, 'r') as f:
                objs = json.load(f)
            new_objs = []
            for obj in objs:
                if obj is None:
                    continue
                keys = list(obj.keys())
                for key in keys:
                    if key not in ['obj_id', 'obj_type', 'psr']:
                        del obj[key]
                obj['score'] = 1.0
                new_objs.append(obj)
            frame = label_file.split('.')[0]
            all_seq_label[seq][frame] = new_objs
    with open(out_label_file, 'w') as f:
        json.dump(dict(all_seq_label), f, indent=2)

def get_gt_viz_format_file(gt_file_paths, coord_type='radar'):
    tr_LR_tvec = [2.54, -0.3, -0.7] if coord_type=='radar' else [0.0, 0.0, 0.0]
    train_path, test_path = gt_file_paths
    dir_name, train_file_name = os.path.split(train_path)
    file_name_split = train_file_name.split('.')[0].split('_')[:]
    out_label_file = os.path.join(dir_name, '_'.join([*file_name_split[:2], 'all', *file_name_split[3:]]) + '_viz_format.json')
    '''
     format: 
      seq:{
        frame:[]
      }
    '''
    with open(train_path, 'r') as f:
        train_labels = json.load(f)['train']
    with open(test_path, 'r') as f:
        test_labels = json.load(f)['test']

    all_seq_label = defaultdict(dict)
    print('Now processing train labels')
    for label in tqdm(train_labels):
        seq = label['seq']
        frame = label['frame']
        objs = label['objs']
        new_objs = []
        for obj in objs:
            if obj is None:
                continue
            frame_obj = {}
            frame_obj['obj_id'] = obj['obj_id']
            frame_obj['obj_type'] = obj['obj_type']
            frame_obj['psr'] = {}
            x, y, z = obj['xyz']
            x_r, y_r, z_r = obj['euler']
            l, w, h = obj['lwh']
            frame_obj['psr']['position'] = {'x': x + tr_LR_tvec[0], 'y': y + tr_LR_tvec[1], 'z': z + tr_LR_tvec[2]}
            frame_obj['psr']['rotation'] = {'x': x_r, 'y': y_r, 'z': z_r}
            frame_obj['psr']['scale'] = {'x': l, 'y': w, 'z': h}
            new_objs.append(frame_obj)
        all_seq_label[seq][frame] = new_objs
    print('Now processing test labels')
    for label in tqdm(test_labels):
        seq = label['seq']
        frame = label['frame']
        objs = label['objs']
        new_objs = []
        for obj in objs:
            if obj is None:
                continue
            frame_obj = {}
            frame_obj['obj_id'] = obj['obj_id']
            frame_obj['obj_type'] = obj['obj_type']
            frame_obj['psr'] = {}
            x, y, z = obj['xyz']
            x_r, y_r, z_r = obj['euler']
            l, w, h = obj['lwh']
            frame_obj['psr']['position'] = {'x': x + tr_LR_tvec[0], 'y': y + tr_LR_tvec[1], 'z': z + tr_LR_tvec[2]}
            frame_obj['psr']['rotation'] = {'x': x_r, 'y': y_r, 'z': z_r}
            frame_obj['psr']['scale'] = {'x': l, 'y': w, 'z': h}
            new_objs.append(frame_obj)
        all_seq_label[seq][frame] = new_objs
    with open(out_label_file, 'w') as f:
        json.dump(dict(all_seq_label), f, indent=2)


def collect_original_label_format(data_root, src_dirs, dst_dir):
    for seq_name in tqdm(sorted(os.listdir(data_root))):
        seq_dst_dir = os.path.join(dst_dir, seq_name)
        if not os.path.exists(seq_dst_dir):
            os.makedirs(seq_dst_dir)
        seq_dir = os.path.join(data_root, seq_name)
        for src_folder in src_dirs:
            src_dir = os.path.join(seq_dir, src_folder)
            if os.path.exists(src_dir):
                shutil.copytree(src_dir, seq_dst_dir, dirs_exist_ok=True)




if __name__ == '__main__':
    # train_samples = '/home/andy/ipl/CenterPoint/configs/kradar/resources/split/train.txt'
    # test_samples = '/home/andy/ipl/CenterPoint/configs/kradar/resources/split/test.txt'
    # save_name = 'refined_v3.json'
    # save_root_path = '/mnt/ssd1/kradar_dataset/labels'
    # is_kitti_format = False # kitti format means in object coordinate frame, x is forward (L), y is down (H), z is left (W)
    # kradar_root = '/mnt/nas_kradar/kradar_dataset/dir_all'
    # print(f'Start to preparing files in {save_root_path}')
    # target_seq = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,\
    #    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,\
    #    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,\
    #     53, 54, 55, 56]
    # rdr_lidar_frame_difference = get_rdr_lidar_frame_difference(kradar_root, target_seq)
    # train_seq_frames = get_target_seq_frames(train_samples, target_seq)
    # print('Start preparing train label.')
    # train_labels = get_labels(train_seq_frames, kradar_root, rdr_lidar_frame_difference)
    # print('Finish train label generation.')
    # print('Start preparing test label.')
    # test_seq_frames = get_target_seq_frames(test_samples, target_seq)
    # test_labels = get_labels(test_seq_frames, kradar_root, rdr_lidar_frame_difference)
    # print('Finish test label generation.')
    # print(f'Writing to {save_name}')
    # output_labels = {} # {split_type: [{obj_type, obj_id, objs: []}]}
    # output_labels['train'] = train_labels
    # output_labels['test'] = test_labels
    # with open(os.path.join(save_root_path, save_name), 'w') as output_labels_file:
    #     json.dump(output_labels, output_labels_file, indent=2)

    get_gt_viz_format_file(['/mnt/ssd1/kradar_dataset/labels/refined_v3_train_Radar_roi1_Sedan_BusorTruck.json', '/mnt/ssd1/kradar_dataset/labels/refined_v3_test_Radar_roi1_Sedan_BusorTruck.json'])