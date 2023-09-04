import os 
from distutils.dir_util import copy_tree

if __name__ == '__main__':
    # copy_folder = ['info_label_pseudo_pvrcnn', 'info_label', 'os2-64', 'os1-128']
    copy_folder = ['label']
    seq_root_dir =  '/home/andy/labeling_tools/SUSTechPOINTS/data'
    dst_root_dir = '/mnt/nas_kradar/kradar_dataset/dir_all'
    # os.makedirs(dst_root_dir, exist_ok=True)
    for seq_name in os.listdir(seq_root_dir):
        seq_dir = os.path.join(seq_root_dir, seq_name)
        for folder_name in copy_folder:
            print('Copying {}/{}'.format(seq_name, folder_name))
            src_dir = os.path.join(seq_dir, folder_name)
            dst_dir = os.path.join(dst_root_dir, seq_name, folder_name)
            os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
            copy_tree(src_dir, dst_dir)