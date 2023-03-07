import os
import json

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Created dir ===>', path)
        
def load_json(json_path):
    """_summary_

    Args:
        json_path (dict): _description_
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    return data
        
def save_json(json_object, save_path):
    """_summary_

    Args:
        json_object (dict): _description_
        save_path (str): _description_
    """
    with open(save_path, 'w') as file:
        json.dump(json_object, file, indent=4)
        
