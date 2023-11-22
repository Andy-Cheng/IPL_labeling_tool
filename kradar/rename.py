import os

def get_rdr_lidar_frame_difference(data_root, target_seqs):
    rdr_lidar_frame_difference = {}
    for seq in target_seqs:
        with open(os.path.join(data_root, str(seq), 'info_calib', 'calib_radar_lidar.txt'), 'r') as f:
            line = f.readlines()[1].strip()
        rdr_lidar_frame_difference[str(seq)] = int(line.split(',')[0])
    return rdr_lidar_frame_difference


target_seq = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,\
18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,\
35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,\
53, 54, 55, 56]
rdr_lidar_frame_difference = get_rdr_lidar_frame_difference('/mnt/nas_kradar/kradar_dataset/dir_all', target_seq)

if __name__ == '__main__':
    root_dir = '/mnt/ssd2/icra_demo_1'
    for seq in sorted(os.listdir(root_dir)):
        if not seq == '54':
            continue
        seq_dir = os.path.join(root_dir, seq)
        for sensor in os.listdir(seq_dir):
            sensor_dir = os.path.join(seq_dir, sensor)
            for file in sorted(os.listdir(sensor_dir)):
                rdr_frame = int(file.split('.')[0])
                frame = rdr_frame - rdr_lidar_frame_difference[seq]
                frame = f'{frame:05}'
                old_file_path = os.path.join(sensor_dir, file)
                new_file_path = os.path.join(sensor_dir, f'{frame}.png')
                os.rename(old_file_path, new_file_path)

