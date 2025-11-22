import os
import shutil
import subprocess
import yaml

# Set paths directly as specified (fill the .... in paths with directory of your machine for DGCNN) 
config_file = r"C:\.....\DFR-main\registration2\config\scape_r.yaml"

with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
    
scape_adni_folder = r"C:\......\DFR-main\registration2\data\scape_try"

temp_train_dir = os.path.join(scape_adni_folder, "shapes_train")
temp_test_dir = os.path.join(scape_adni_folder, "shapes_test")

# Define the command to run the registration
dgcnn_command = "python test.py"

def setup_directories():
    # Clear the contents of the shapes_train directory
    for filename in os.listdir(temp_train_dir):
        file_path = os.path.join(temp_train_dir, filename)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

    # Delete the cache files
    cache_files = [os.path.join(scape_adni_folder, 'cache_scape_dg_train.pt'), os.path.join(scape_adni_folder, 'cache_scape_dg_test.pt')]
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)

def run_registration(input_shape, input_filename):
    # Copy the input shape to the shapes_train directory with the actual filename
    dest_shape_path = os.path.join(temp_train_dir, input_filename)
    shutil.copy(input_shape, dest_shape_path)

    # Run the registration command
    try:
        subprocess.run(dgcnn_command, shell=True, check=True, cwd=r"C:\......\DFR-main\registration2")

    except subprocess.CalledProcessError as e:
        print(f"Registration command failed for {input_shape} with error: {e}")
        return
def run_reg(input_folder):
    # Loop through each input shape and deform it
    for filename in os.listdir(input_folder):
        if filename.endswith(".off"):
            input_shape = os.path.join(input_folder, filename)

            # Setup directories for each shape
            setup_directories()

            # Run registration for the current shape
            run_registration(input_shape, filename)


# Same procedure for NC and MCI
input_folder= r"C:\.....\AD_Aligned_Offs" #folder path of the aligned shapes
run_reg(input_folder)
