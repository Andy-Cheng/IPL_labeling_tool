import os
import json
from tqdm import tqdm
from collections import defaultdict
from PIL import ImageColor
import random
import argparse

def get_rdr_lidar_frame_difference(data_root, target_seqs):
    rdr_lidar_frame_difference = {}
    for seq in target_seqs:
        with open(os.path.join(data_root, str(seq), 'info_calib', 'calib_radar_lidar.txt'), 'r') as f:
            line = f.readlines()[1].strip()
        rdr_lidar_frame_difference[str(seq)] = int(line.split(',')[0])
    return rdr_lidar_frame_difference

def rand_colors():
    return '#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
     


def parse_args():
    parser = argparse.ArgumentParser(description="Transform tracking result to the labeling tool format.")
    parser.add_argument("pred_files_root", help="The root folder of the prediction files.")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    new_pred = {}
    target_seq = list(range(1, 59))
    for i in [51, 52, 57, 58]:
        target_seq.remove(i)
    # target_seq = [5]
    target_seq = [str(i) for i in target_seq]
    rdr_lidar_frame_difference = get_rdr_lidar_frame_difference('/mnt/nas_kradar/kradar_dataset/dir_all', target_seq)
    # pred_files_root = '/mnt/nas_kradar/kradar_dataset/detection_result/icra/triplet_cosine_one_pos_hard_neg/SCT_radar_tracking_0914_icra_triplet_cosine_one_pos_hard_neg_diou'
    # pred_files_root = '/mnt/nas_kradar/kradar_dataset/SCT_CVPR_Result/CVPR_demo'
    pred_files_root = args.pred_files_root
    # color_table_path = '/mnt/nas_kradar/kradar_dataset/detection_result/icra/triplet_cosine_one_pos_hard_neg/color_table.json'
    cls_id_to_name = ['Sedan', 'BusorTruck']

    # with open(color_table_path, 'r') as f:
    #     color_table = json.load(f)

    tracking_id_to_color, used_colors = {}, []




    for seq in tqdm(target_seq):
        pred_file = os.path.join(pred_files_root, f'{seq}_predbbx.txt')
        pred_seq = defaultdict(dict)
        with open(pred_file, 'r') as f:
            for line in f.readlines():
                # frame_name,trackid,tl_x,tl_y,tl_z,xl,yl,zl,theta,conf,class
                rdr_frame, trackid, tl_x, tl_y, tl_z, xl, yl, zl, theta, conf, cls, cost, iou_cost, embs_cost = line.strip().split(',')
                tl_x, tl_y, tl_z = float(tl_x)+float(xl)/2, float(tl_y)+float(yl)/2, float(tl_z)+float(zl)/2
                tl_x += 2.54
                tl_y += -0.3
                tl_z += -0.7
                rdr_frame = int(rdr_frame)
                frame = rdr_frame - rdr_lidar_frame_difference[seq]
                frame = f'{frame:05}'
                rdr_frame = f'{rdr_frame:05}'
                psr = {'position': {'x': tl_x, 'y': tl_y, 'z':tl_z}, 'rotation':{'x': 0, 'y': 0, 'z': float(theta)}, 'scale': {'x': float(xl), 'y': float(yl), 'z': float(zl)}}
                if trackid in tracking_id_to_color:
                    clr = tracking_id_to_color[trackid]
                else:
                    clr = rand_colors()
                    while clr in used_colors:
                        clr = rand_colors()
                    tracking_id_to_color[trackid] = clr
                    used_colors.append(clr)
                obj = {'obj_id': trackid, 'obj_type': cls_id_to_name[int(cls)-1], 'score': float(conf), 'psr': psr, 'color': clr}
                if frame in pred_seq:
                    pred_seq[frame]['objs'].append(obj)
                else:
                    pred_seq[frame]['objs'] = [obj]
        new_pred[seq] = dict(pred_seq)

    save_file_path = os.path.join(os.path.dirname(pred_files_root), 'viz_format.json')
    with open(save_file_path, "w") as f:
        json.dump(new_pred, f, indent=2)