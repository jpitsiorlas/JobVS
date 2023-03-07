######################################################

# Organize data in 'tof' folder (args.tof_data_path) with the images and labels. 
# We take from the original data and the annotations folder and generate 
# symlinks for better organization.
# This data management allows a data version control.

DATASET_VERSION='tof_ver_monai'
DATA_VERSION='all_data_mondai.json'

python organize_data.py --json_data_file $DATA_VERSION --dataset_version $DATASET_VERSION
python create_dataset.py --dataset_version $DATASET_VERSION
######################################################
