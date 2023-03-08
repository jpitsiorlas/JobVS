from global_features import Global_features
from task_preprocessing import Preprocess_datasets
from image_sizes import Calculate_sizes

data_root = '/home/pitsiorl/Shiny_Icarus/data/dataset/original/'
out_directory = '/home/pitsiorl/Shiny_Icarus/data/dataset/original/'
num_workers = 30
process_again=True

# Calculate the statistics of the original dataset
Global_features(data_root, num_workers)

# # Preprocess the datasets
#Preprocess_datasets(out_directory, data_root, num_workers, remake=process_again)

# # Calculate the dataset sizes (used to plan the experiments)
#Calculate_sizes(out_directory, num_workers, remake=process_again)

