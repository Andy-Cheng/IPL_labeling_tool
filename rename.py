import os
import shutil

def read_frame_meta(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        os64_idx, cam_front_idx = lines[0].split(' ')[1].split('=')[1].split('_')[1:3]
        os64_idx = int(os64_idx)
        cam_front_idx = int(cam_front_idx)
    return os64_idx, cam_front_idx


def main():
    seq_root = '/mnt/nas_kradar/kradar_dataset/dir_all'
    cam_name = 'cam-front-undistort'
    for seq_name in os.listdir(seq_root):
        print('Now processing {}'.format(seq_name))
        seq_dir = os.path.join(seq_root, seq_name)
        cam_dir = os.path.join(seq_dir, cam_name)
        tmp_cam_dir = os.path.join(seq_dir, 'cam-front-undistort-tmp')
        if not os.path.exists(tmp_cam_dir):
            os.rename(cam_dir, tmp_cam_dir)
        info_label_dir = os.path.join(seq_dir, 'info_label')
        for txt_file in os.listdir(info_label_dir): 
            if txt_file.endswith('.txt'):
                label_file = os.path.join(info_label_dir, txt_file)
                os64_idx, cam_front_idx = read_frame_meta(label_file)
                img_old_name = '{}_{:0=5}.png'.format(cam_name, cam_front_idx)
                os.makedirs(cam_dir, exist_ok=True)
                shutil.copy(os.path.join(tmp_cam_dir, img_old_name), os.path.join(cam_dir, '{}_{:0=5}.png'.format(cam_name, os64_idx)))

        for img_name in os.listdir(cam_dir):
            if img_name.endswith('.png'):
                new_img_name = img_name.split('_')[1]
                new_img_name = f'{cam_name}_{new_img_name}'
                new_img_file = os.path.join(cam_dir, new_img_name)
                img_file = os.path.join(cam_dir, img_name)
                os.rename(img_file, new_img_file)
# main()


def copy_calib():
    calib_file = '/mnt/nas_kradar/kradar_dataset/dir_all/13/calib/camera/cam-front-undistort.json'
    seq_root = '/mnt/nas_kradar/kradar_dataset/dir_all'
    for seq_name in os.listdir(seq_root):
        if seq_name == '13':
            continue
        print('Now processing {}'.format(seq_name))
        seq_calib_dir = os.path.join(seq_root, seq_name, 'calib', 'camera')
        os.makedirs(seq_calib_dir, exist_ok=True)
        shutil.copy(calib_file, os.path.join(seq_calib_dir, 'cam-front-undistort.json'))

copy_calib()