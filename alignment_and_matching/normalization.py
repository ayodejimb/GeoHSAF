import os
import shutil

# Define the function to center and normalize the vertices of a mesh
def normalize_mesh(vertices):
    center = [sum(v[i] for v in vertices) / len(vertices) for i in range(3)]
    normalized_vertices = [[v[i] - center[i] for i in range(3)] for v in vertices]
    max_coord = max(max(abs(v[i]) for v in normalized_vertices) for i in range(3))
    normalized_vertices = [[v[i] / max_coord for i in range(3)] for v in normalized_vertices]
    return normalized_vertices

def read_off_file(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if lines[0].strip() == 'OFF':
            num_vertices, num_faces, _ = map(int, lines[1].strip().split())
            for line in lines[2:2 + num_vertices]:
                vertex = list(map(float, line.strip().split()))
                vertices.append(vertex)
            for line in lines[2 + num_vertices:]:
                face = list(map(int, line.strip().split()[1:]))
                faces.append(face)
    return vertices, faces

def write_off_file(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        file.write('OFF\n')
        file.write(f'{len(vertices)} {len(faces)} 0\n')
        for vertex in vertices:
            file.write(' '.join(map(str, vertex)) + '\n')
        for face in faces:
            file.write(f'{len(face)} {" ".join(map(str, face))}\n')

# Define the function to process a single file
def process_file(input_file, output_folder):
    vertices, faces = read_off_file(input_file)
    normalized_vertices = normalize_mesh(vertices)
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    write_off_file(output_file, normalized_vertices, faces)

def normalize(input_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    # Process each file in the input folder
    for file in files:
        if file.endswith(".off"):
            input_file = os.path.join(input_folder, file)
            process_file(input_file, output_folder)

    print("Normalization complete.")

# Folder containing your mesh files (same procedure for NC and MCI)
input_folder = r"C:\.....\AD_Offs"
output_folder = r"C:\.....\AD_Normalized_Offs"
normalize(input_folder)
