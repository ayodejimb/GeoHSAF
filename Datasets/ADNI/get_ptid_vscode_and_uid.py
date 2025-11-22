import pandas as pd

file_path = r"C:\.......\ADNI\AD_only.csv"
output_file = "AD_ptid_imageuid_list.txt"

# file_path = r"C:\.......\ADNI\CN_and_SMC_only.csv"
# output_file = "CN_and_SMC_ptid_imageuid_list.txt"

# file_path = r"C:\.......\ADNI\EMCI_and_LMCI_only.csv"
# output_file = "EMCI_and_LMCI_ptid_imageuid_list.txt"

df = pd.read_csv(file_path, low_memory=False)

ptid_dict = {}

# Iterate through the DataFrame
for _, row in df.iterrows():
    ptid = row["PTID"]
    viscode = row["VISCODE"]
    imageuid = row["IMAGEUID"]
    
    if ptid not in ptid_dict:
        ptid_dict[ptid] = [[], []]
    
    # Append VISCODE and IMAGEUID to the respective lists
    imageuid = int(imageuid) if pd.notna(imageuid) else imageuid
    ptid_dict[ptid][0].append(viscode)
    ptid_dict[ptid][1].append(imageuid)

# print(ptid_dict)

# Filter to select only the time points where we have Imageuid
for ptid, values in ptid_dict.items():
    viscodes, imageuids = values 
    
    # Create a new filtered list excluding NaN imageuids
    cleaned_viscodes, cleaned_imageuids = zip(*[
        (viscode, imageuid) for viscode, imageuid in zip(viscodes, imageuids) if pd.notna(imageuid)
    ]) if any(pd.notna(imageuid) for imageuid in imageuids) else ([], [])

    # Update the dictionary
    ptid_dict[ptid] = [list(cleaned_viscodes), list(cleaned_imageuids)]

# Remove empty vscode and uid ***
ptid_dict = {key: value for key, value in ptid_dict.items() if value[0]}
# print(len(ptid_dict.keys())) 

# *** Now we output the PTID and the IMAGEUIDs for downloading in ADNI ****
ptid_list = []
imageuid_list = []
for ptid, (viscodes, imageuids) in ptid_dict.items():
    for imageuid in imageuids:
        ptid_list.append(ptid) 
        imageuid_list.append(str(imageuid))

# Ensure their lengths match
assert len(ptid_list) == len(imageuid_list), "Lengths do not match!"

with open(output_file, "w") as file:
    file.write("PTIDs:\n")
    file.write(", ".join(ptid_list) + "\n\n") 

    file.write("IMAGEUIDs:\n")
    file.write(", ".join(imageuid_list) + "\n")

print(f"File saved as: {output_file}")
print(f"Total PTIDs: {len(ptid_list)} | Total IMAGEUIDs: {len(imageuid_list)}") 