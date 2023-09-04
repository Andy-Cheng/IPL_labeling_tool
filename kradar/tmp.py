import os 
import json 
import numpy as np


seq_label_root = '/mnt/nas_kradar/kradar_dataset/dir_all/1/label'
for json_file_name in os.listdir(seq_label_root):
    json_path = os.path.join(seq_label_root, json_file_name)
    with open(json_path, 'r') as f:
        objs = json.load(f)
    new_objs = []
    for obj in objs:
        if  obj is None or not 'obj_id' in obj:
            continue
        r = obj['psr']['rotation']
        s = obj['psr']['scale']
        if s['x'] < 0:
            s['x'] = -s['x']
            r['z'] = round(r['z'] + float(np.pi), 4)
            obj['psr']['rotation'] = r
            obj['psr']['scale'] = s
        new_objs.append(obj)
    
    with open(json_path, 'w') as f:
        json.dump(new_objs, f, indent=2)