# ** This code removes an intermediary folder from the ADNI downloads. Specifically, it removes the folder named, 'MPRAGE ADNI Confirmed' or as the name can be for easier organization ****
import os
import shutil

main_path = r"C:\.....\AIBL\AD"
# main_path = r"C:\.....\AIBL\CN"
# main_path = r"C:\.....\AIBL\MCI"

for folder_name in os.listdir(main_path):
    folder_path = os.path.join(main_path, folder_name)
    
    if os.path.isdir(folder_path):
        subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

        # Proceed if there is exactly one intermediary folder
        if len(subfolders) == 1:
            intermediary_path = os.path.join(folder_path, subfolders[0])

            # Move each item (folder or file) from intermediary to the parent folder
            for item in os.listdir(intermediary_path):
                src = os.path.join(intermediary_path, item)
                dst = os.path.join(folder_path, item)

                # If destination exists, you may want to rename or skip (optional logic)
                if os.path.exists(dst):
                    print(f"Skipping {src}, because {dst} already exists.")
                    continue
                
                shutil.move(src, dst)

            # Delete the now-empty intermediary folder
            os.rmdir(intermediary_path)

print("Intermediary folders removed and contents moved up.")
