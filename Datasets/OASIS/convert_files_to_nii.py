import os
import nibabel as nib

# Function to convert .hdr and .img files to .nii
def convert_hdr_to_nifti(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.nifti.hdr'):
                hdr_file = os.path.join(root, file)
                img_file = hdr_file.replace('.nifti.hdr', '.nifti.img')

                if os.path.exists(img_file):
                    analyze_img = nib.load(hdr_file)

                    nii_file = hdr_file.replace('.nifti.hdr', '.nii')

                    # Save the NIfTI (.nii) file
                    nib.save(analyze_img, nii_file)

                    print(f"Converted: {nii_file}")
                else:
                    print(f"Missing .img file for {hdr_file}")

# Set the main directory path where the subfolders are located
main_directory = r'C:\.....\OASIS\OAS2_RAW_PART1\OAS2_RAW_PART1' 
# main_directory = r'C:\.....\OASIS\OAS2_RAW_PART2\OAS2_RAW_PART2' 

# Convert all .hdr/.img files in subfolders
convert_hdr_to_nifti(main_directory)
