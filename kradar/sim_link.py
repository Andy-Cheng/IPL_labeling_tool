import os

if __name__ == '__main__':
    root_dir = '/mnt/ssd2/icra_demo_1'
    dst_root_dir = '/mnt/nas_kradar/kradar_dataset/dir_all'
    name_mapping = {'bev_radar': 'Radar', 'cam': 'Camera'}
    for seq in os.listdir(root_dir):
        seq_dir = os.path.join(root_dir, seq)
        for sensor in os.listdir(seq_dir):
            sensor_dir = os.path.join(seq_dir, sensor)
            dst_dir = os.path.join(dst_root_dir, seq, name_mapping[sensor])
            if os.path.islink(dst_dir):
                os.remove(dst_dir)
            os.symlink(sensor_dir, dst_dir)


