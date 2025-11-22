import os

def read_vtk(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    points = []
    polygons = []
    points_section = False
    polygons_section = False

    for line in lines:
        if line.startswith("POINTS"):
            points_section = True
            continue
        elif line.startswith("POLYGONS"):
            points_section = False
            polygons_section = True
            continue
        elif line.strip() == "":
            continue
        if points_section:
            points.append(line.strip())
        elif polygons_section:
            if line.startswith("3 "):
                polygons.append(line.strip()[2:])

    return points, polygons

def write_off(file_path, points, polygons):
    with open(file_path, 'w') as file:
        file.write("OFF\n")
        file.write(f"{len(points)} {len(polygons)} 0\n")
        for point in points:
            file.write(f"{point}\n")
        for polygon in polygons:
            file.write(f"3 {polygon}\n")

def convert_vtk_to_off(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".vtk"):
            vtk_file_path = os.path.join(input_folder, filename)
            off_file_path = os.path.join(output_folder, filename.replace(".vtk", ".off"))

            points, polygons = read_vtk(vtk_file_path)
            write_off(off_file_path, points, polygons)

# same for VTks from NC and MCI
input_folder = r"C:\....\AD_VTKs"
output_folder = r"C:\....\AD_Offs"

convert_vtk_to_off(input_folder, output_folder)
