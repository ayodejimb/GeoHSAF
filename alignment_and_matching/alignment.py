import trimesh
import open3d as o3d
import numpy as np
import os

reference_mesh = r"C:\...\atlas.off"  # the atlas path

# Load the meshes
def load_mesh(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    return mesh

def save_off(file_path, mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    with open(file_path, 'w') as f:
        f.write("OFF\n")
        f.write(f"{vertices.shape[0]} {triangles.shape[0]} 0\n")
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        for triangle in triangles:
            f.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")

def procrustes_alignment(X, Y):
    A = np.dot(Y.T, X)
    U, S, Vt = np.linalg.svd(A)
    R = np.dot(U, Vt)
    Y_aligned = np.dot(Y, R)
    return Y_aligned

def rigid_reg(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    reference_pcd = load_mesh(reference_mesh)

    file_names = sorted([file_name for file_name in os.listdir(input_folder) if file_name.endswith('.off')])
    meshes = [load_mesh(os.path.join(input_folder, file_name)) for file_name in file_names]

    aligned_meshes_list = []
    for mesh in meshes:
        mesh_aligned = procrustes_alignment(np.asarray(reference_pcd.vertices), np.asarray(mesh.vertices))
        mesh.vertices = o3d.utility.Vector3dVector(mesh_aligned)
        aligned_meshes_list.append(mesh)

    for mesh, file_name in zip(aligned_meshes_list, file_names):
        output_path = os.path.join(output_folder, file_name)
        save_off(output_path, mesh)


# Folder containing your mesh files (same procedure for NC and MCI)
input_folder =  r"C:\....\AD_Normalized_Offs"
output_folder = r"C:\...\AD_Aligned_Offs"

rigid_reg(input_folder, output_folder)
