import os
import json
import numpy as np
from tqdm import tqdm
# class_name_mapping_kradar_to_sus = {
#     'Seden': 'Sedan',
#     'Cyclist': 'Bicycle',
#     'Pedestrian': 'Pedestrian',
# }

class_name_mapping_kradar_to_sus = {
    'Bus or Truck': 'BusorTruck',
    'Bicycle Group': 'BicycleGroup',
    'Pedestrian Group': 'PedestrianGroup',
}

class_name_mapping_sus_to_kradar = {
    'BusorTruck': 'Bus or Truck',
    'BicycleGroup': 'Bicycle Group',
    'PedestrianGroup': 'Pedestrian Group',
}



def get_dict_object(line, is_heading_in_rad=True):
    '''
    * in : e.g., '*, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> One Example
    * in : e.g., '*, 0, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> There are labels like this too
    * out: tuple ('Sedan', idx_cls, [x, y, z, theta, l, w, h], idx_obj)
    *       None if idx_cls == -1 or header != '*'
    '''
    list_values = line.split(',')

    if list_values[0] != '*':
        return None

    offset = 0
    if(len(list_values)) == 11:
        # print('* Exception error (Dataset): length of values is 11')
        offset = 1
    else:
        # print('* Exception error (Dataset): length of values is 10')
        # print(path_label)
        pass

    cls_name = list_values[2+offset][1:]

    idx_obj = int(list_values[1+offset])
    x = float(list_values[3+offset])
    y = float(list_values[4+offset])
    z = float(list_values[5+offset])
    theta = float(list_values[6+offset])
    if is_heading_in_rad:
        theta = theta*np.pi/180.
    l = 2*float(list_values[7+offset])
    w = 2*float(list_values[8+offset])
    h = 2*float(list_values[9+offset])
    cls_name = class_name_mapping_kradar_to_sus[cls_name] if cls_name in class_name_mapping_kradar_to_sus else cls_name
    obj_dict = {
        "psr": {
            "position": {
                "x": x,
                "y": y,
                "z": z
            },
            "scale": {
                "x": l,
                "y": w,
                "z": h
            },
            "rotation": {
                "x": 0.,
                "y": 0.,
                "z": theta
            }
        },
        "obj_type": cls_name,
        "obj_id": str(idx_obj)
    }

    return obj_dict
        


def kradar_to_sus(txt_file, dst_dir):
    save_file_name = '{:0=5}.json'.format(int(os.path.basename(txt_file).split('.')[0].split('_')[1]))
    save_file_name = os.path.join(dst_dir, save_file_name)
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    list_objects = []
    for line in lines:
        obj_dict = get_dict_object(line)
        if obj_dict is not None:
            list_objects.append(obj_dict)
    with open(save_file_name, 'w') as f:
        json.dump(list_objects, f, indent=2)


def get_object_line(obj_dict):
    '''
    * in : e.g., '*, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> One Example
    * in : e.g., '*, 0, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> There are labels like this too
    * out: tuple ('Sedan', idx_cls, [x, y, z, theta, l, w, h], idx_obj)
    *       None if idx_cls == -1 or header != '*'
    '''
    # if obj_dict is None:
    #     return None
    psr = obj_dict['psr']
    position = psr['position']
    scale = psr['scale']
    rotation = psr['rotation']
    x = position['x']
    y = position['y']
    z = position['z']
    theta = rotation['z'] * 180. / np.pi
    l = scale['x']
    w = scale['y']
    h = scale['z']
    cls_name = obj_dict['obj_type']
    idx_obj = obj_dict['obj_id']
    if cls_name in class_name_mapping_sus_to_kradar:
        cls_name = class_name_mapping_sus_to_kradar[cls_name]
    obj_line = '*, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(idx_obj, cls_name, x, y, z, theta, l/2., w/2., h/2.)
    return obj_line 

def sus_to_kradar(input_json_file, dst_txt_name, original_txt_name):
    with open(original_txt_name, 'r') as f:
        lines = f.readlines()
    first_line = lines[0]

    with open(input_json_file, 'r') as f:
        list_objects = json.load(f)
    
    with open(dst_txt_name, 'w') as f:
        f.write(first_line)
        lines = []
        for obj_dict in list_objects:
            if obj_dict is None:
                continue
            obj_line = get_object_line(obj_dict)
            lines.append(obj_line)
        f.writelines(lines)


    

def sus_to_kradar_seq():
    seq_to_skip = [] # [51, 52, 57, 58]
    seq_to_skip = [str(i) for i in seq_to_skip]
    seq_root = '/mnt/nas_kradar/kradar_dataset/dir_all'
    src_dst_pairs = [ ('label', 'refined_label_all')]
    original_label_folder = 'info_label'
    for seq_name in tqdm(sorted(os.listdir(seq_root))):
        if seq_name in seq_to_skip:
            continue
        seq_dir = os.path.join(seq_root, seq_name)
        for src_folder, dst_folder in src_dst_pairs:
            src_dir = os.path.join(seq_dir, src_folder)
            dst_dir = os.path.join(seq_dir, dst_folder)
            original_label_dir = os.path.join(seq_dir, original_label_folder)
            if os.path.exists(src_dir):
                print(f'Processing {src_dir}')
                os.makedirs(dst_dir, exist_ok=True)
                for txt_file in sorted(os.listdir(original_label_dir)):
                    if txt_file.endswith('.txt'):
                        input_json_file = '{:0=5}.json'.format(int(os.path.basename(txt_file).split('.')[0].split('_')[1]))
                        input_json_file = os.path.join(src_dir, input_json_file)
                        original_txt_file = os.path.join(original_label_dir, txt_file)  
                        dst_txt_file = os.path.join(dst_dir, txt_file)
                        if os.path.exists(dst_txt_file):
                            continue
                        sus_to_kradar(input_json_file, dst_txt_file, original_txt_file)


def main():
    seq_root = '/mnt/nas_kradar/kradar_dataset/dir_all'
    src_dst_pairs = [('refined_label_all', 'label')]
    for seq_name in sorted(os.listdir(seq_root)):
        seq_dir = os.path.join(seq_root, seq_name)
        for src_folder, dst_folder in src_dst_pairs:
            src_dir = os.path.join(seq_dir, src_folder)
            dst_dir = os.path.join(seq_dir, dst_folder)
            if os.path.exists(src_dir):
                print(f'Processing {src_dir}')
                os.makedirs(dst_dir, exist_ok=True)
                for txt_file in sorted(os.listdir(src_dir)):
                    if txt_file.endswith('.txt'):
                        txt_file = os.path.join(src_dir, txt_file)
                        kradar_to_sus(txt_file, dst_dir)


                

def rename_pcd():
    seq_root = '/mnt/disk2/kradar_dataset/dir_all'
    for seq_name in os.listdir(seq_root):
        seq_dir = os.path.join(seq_root, seq_name)
        for folder_name in ['os2-64', 'os1-128']:
            folder_dir = os.path.join(seq_dir, folder_name)
            if os.path.exists(folder_dir):
                print(f'Processing {folder_dir}')
                for pcd_file in os.listdir(folder_dir):
                    if pcd_file.endswith('.pcd'):
                        new_pcd_file = pcd_file.split('_')[1]
                        pcd_file = os.path.join(folder_dir, pcd_file)
                        new_pcd_file = os.path.join(folder_dir, new_pcd_file)
                        os.rename(pcd_file, new_pcd_file)

if __name__ == '__main__':
    main()
    # sus_to_kradar_seq()