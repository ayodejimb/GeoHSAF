import os
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import re

atlas_file = r"C:\...\atlas.off"

deformed_folder = r"C:\.....\deformed_meshes"  #folder of deformed meshes from DGCNN
original_folder = r"C:\.....\aligned_meshes" #folder of aligned meshes

# ====================================

for filename in os.listdir(deformed_folder):
    if filename.endswith('_atlas.off'):
        new_filename = filename.replace('_atlas', '')
        old_file = os.path.join(deformed_folder, filename)
        new_file = os.path.join(deformed_folder, new_filename)
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} → {new_filename}")

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def extract_point_matches_ordered(atlas_file, deformed_file, original_file):
    # Load atlas, deformed, and original meshes
    atlas_mesh = trimesh.load_mesh(atlas_file, process=False) 
    deformed_mesh = trimesh.load_mesh(deformed_file)
    original_mesh = trimesh.load_mesh(original_file)

    # Extract vertices from meshes
    atlas_points = np.array(atlas_mesh.vertices)
    deformed_points = np.array(deformed_mesh.vertices)
    original_points = np.array(original_mesh.vertices)
    
    deformed_tree = cKDTree(deformed_points)
    _, nearest_indices = deformed_tree.query(atlas_points)
    corresponding_original_points = original_points[nearest_indices]

    point_matches = [(atlas_points[i], corresponding_original_points[i]) for i in range(len(atlas_points))]
    return point_matches

def extracting_matches(atlas_file, deformed_folder, original_folder, matches_folder):
    deformed_files = sorted(os.listdir(deformed_folder), key=natural_sort_key)
    original_files = sorted(os.listdir(original_folder), key=natural_sort_key)

    assert len(deformed_files) == len(original_files), "The number of deformed shapes and original shapes must match."

    for i in range(len(deformed_files)):
        deformed_file = os.path.join(deformed_folder, deformed_files[i])
        original_file = os.path.join(original_folder, original_files[i])

        point_matches = extract_point_matches_ordered(atlas_file, deformed_file, original_file)

        # Save point matches using absolute path to the matches folder
        matches_file = os.path.join(matches_folder, f"matched_{os.path.splitext(deformed_files[i])[0]}.txt")
        with open(matches_file, 'w') as f:
            for match in point_matches:
                atlas_point_str = ' '.join(map(str, match[0]))
                original_point_str = ' '.join(map(str, match[1]))
                f.write(f"{atlas_point_str} -> {original_point_str}\n")

        print(f"Extracted point matches for {deformed_files[i]} and saved to {matches_file}")


matches_folder = r"C:\.....\matched_meshes"   # output folder here
extracting_matches(atlas_file, deformed_folder, original_folder, matches_folder)