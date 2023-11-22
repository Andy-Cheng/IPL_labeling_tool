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


obj_type_mapping_old = {
    'Car': 'Car',
    'Van': 'Car',
    'Truck': 'Car',
    'Bus': 'Car',
    'Pedestrian': 'Pedestrian',
    'Bicycle': 'Bicycle',
    'BicycleRider': 'BicycleRider'
}

obj_type_mapping = {
    'Car': 'Car',
    'Van': 'Car',
    'Truck': 'BusorTruck',
    'Bus': 'BusorTruck',
}

def get_labels(ds_root, target_seqs, label_dir_name='label'):
    seq_start_end = defaultdict(dict)
    train_labels, test_labels = [], []
    for seq in target_seqs:
        print(f'Now processing {seq}')
        seq_start_end['train'][seq] = []
        seq_start_end['test'][seq] = []
        current_start_end = []
        for frame_id in tqdm(range(2, 1798)):
            label_path = os.path.join(ds_root, seq, label_dir_name, f'{frame_id:0>6}.json')
            if not os.path.exists(label_path):
                label = []
            else:
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
                    obj_type = obj['obj_type']
                    if not obj_type in obj_type_mapping.keys():
                        print(f'Unknown object type {obj_type}')
                        continue
                    converted_obj = {'obj_id': obj['obj_id'], 'obj_type': obj_type_mapping[obj_type], 'euler': euler, 'xyz': xyz, 'lwh': lwh}
                    objs.append(converted_obj)
            rdr_frame = f'{frame_id:0>6}'
            label_frame = {'seq': seq, 'frame': rdr_frame, 'rdr_frame': rdr_frame, 'objs': objs}
            if int((frame_id-2) / 150)%2 == 0: # every 150 frames belong test or train set
                train_labels.append(label_frame)
                if (frame_id-2) % 150 == 0:
                    current_start_end = [len(train_labels)-1]
                elif (frame_id-2) % 150 == 149:
                    current_start_end.append(len(train_labels)-1)
                    seq_start_end['train'][seq].append(current_start_end)
            else:
                test_labels.append(label_frame)
                if (frame_id-2) % 150 == 0:
                    current_start_end = [len(test_labels)-1]
                elif (frame_id-2) % 150 == 149:
                    current_start_end.append(len(test_labels)-1)
                    seq_start_end['test'][seq].append(current_start_end)
    return train_labels, test_labels, seq_start_end


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
    save_name = 'CRUW3DCarTruck.json'
    save_root_path = '/mnt/ssd3/CRUW3D/labels'
    cruw_root = '/mnt/ssd3/CRUW3D/seqs'
    print(f'Start to preparing files in {save_root_path}')
    target_seq = ['2021_1120_1616', '2021_1120_1618',  '2022_0203_1439', '2022_0203_1441', '2022_0203_1443', '2022_0203_1445', '2022_0217_1251', '2022_0217_1307', '2021_1120_1632', '2021_1120_1634'] 
    train_labels, test_labels, seq_start_end  = get_labels(cruw_root, target_seq, label_dir_name='label')
    print(f'Writing to {save_name}')
    output_labels = {} # {split_type: [{obj_type, obj_id, objs: []}]}
    output_labels['train'] = train_labels
    output_labels['test'] = test_labels
    with open(os.path.join(save_root_path, save_name), 'w') as output_labels_file:
        json.dump(output_labels, output_labels_file, indent=2)
    with open(os.path.join(save_root_path, 'cruw22_seq_start_end.json'), 'w') as output_file:
        json.dump(seq_start_end, output_file, indent=2)

    # get_gt_viz_format_file(['/mnt/ssd1/kradar_dataset/labels/refined_v3numpoints_train_Radar_roi1_Sedan_BusorTruck.json', '/mnt/ssd1/kradar_dataset/labels/refined_v3numpoints_test_Radar_roi1_Sedan_BusorTruck.json'])