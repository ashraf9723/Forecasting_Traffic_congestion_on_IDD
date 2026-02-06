import json
import pandas as pd
import numpy as np

def load_idd_sequence(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extracting GPS and OBD Speed
    # Note: IDD JSON structure varies, adjust keys accordingly
    speeds = [frame['obd']['speed'] for frame in data['frames']]
    coords = [[frame['gps']['lat'], frame['gps']['lon']] for frame in data['frames']]
    
    return np.array(speeds), np.array(coords)