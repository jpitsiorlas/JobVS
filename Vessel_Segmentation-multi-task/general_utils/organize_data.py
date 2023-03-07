import os
import shutil
import nibabel
import numpy as np

from tqdm import tqdm
from utils import save_json
from arguments import ArgsInit

        
def find_path(name, modality, path):
    """_summary_

    Args:
        name (str): _description_
        modality (str): _description_
        path (str): _description_

    Returns:
        str: _description_
    """
    found_paths = []
    root_path = os.path.join(path, name)
    for root, _, files in os.walk(root_path):
        for filename in files:
            if modality in filename.lower() and '.nii.gz' in filename.lower():
                path_name = os.path.join(root_path, root, filename)
                found_paths.append(path_name)
                
    assert len(found_paths) == 1
                
    return found_paths[0]

def main(args):
    """_summary_

    Args:
        args (_type_): _description_
    """
    data = []
    for root, _, files in tqdm(os.walk(args.anns_data_path)):
        for file in files:
            if '.nii' not in file:
                continue

            # # Get MRI name
            name = file
            output_path = os.path.join(args.smile_data_path, name)
            if not os.path.exists(output_path):
                 os.makedirs(output_path)
                
            # Load annotation
            ann_src = os.path.join(root, file)
            ann = nibabel.load(ann_src)
            ann_mod = ann.get_fdata() 
            print('ann_mod', ann_mod.shape)
            
            # Load and transform original MRI
            oriroot= root.split('/TrainManualLabels')[0]
            im_src = os.path.join(oriroot,'TrainVolumes', file+'.gz') #find_path(name, args.original_data_path)
            im = nibabel.load(im_src)
            im_mod = im.get_fdata()
            
            # Save annotation
            ann_dst = os.path.join(output_path, 'label.nii.gz')
            ann_mod = nibabel.Nifti1Image(
                        ann_mod.astype(ann.get_data_dtype()), ann.affine)
            nibabel.save(ann_mod, ann_dst)
            
            #Save image smile_data_path
            im_dst = os.path.join(output_path, 'image.nii.gz')
            im_mod = nibabel.Nifti1Image(
                        im_mod.astype(im.get_data_dtype()), im.affine)
            nibabel.save(im_mod, im_dst)
            
            cur_data = {'image': im_dst.replace(args.smile_data_path+'/', ''), 
                        'label': ann_dst.replace(args.smile_data_path+'/', ''),
                    }
            
            data.append(cur_data)
            
    #assert len(data) == len(os.listdir(args.anns_data_path))
    # Create dataset json
    images_json = {
            "description": 'ShinyIcarus Eurecom and CinfonIA',
            "labels": {
                        "0": "background",
                        "1": "vessel"
                        },
            "modality": {"0": "tof"},
            "name": "smileuhura",
            "root": args.smile_data_path,
            "data": data        
    }
    json_dir = os.path.join(args.smile_data_path, args.json_data_file)
    save_json(images_json, json_dir)
    print('Json file saved at =====> ', json_dir)
    
    # Save a copy of the data version in the dataset path.
    shutil.copy(json_dir, os.path.join(args.dataset_path, 'data.json'))
    

if __name__ == "__main__":
    args = ArgsInit().return_args()
    main(args)