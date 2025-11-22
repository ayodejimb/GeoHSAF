import pandas as pd

# Load the CSV file
file_path = r"C:\.......\ADNIMERGE_27Mar2025.csv"
df = pd.read_csv(file_path, low_memory=False)

# filtered_df = df[df["DX_bl"].isin(["CN", "SMC"])]
# output_file = "CN_and_SMC_only.csv" 
# filtered_df.to_csv(output_file, index=False)

# filtered_df = df[df["DX_bl"].isin(["AD"])]
# output_file = "AD_only.csv" 
# filtered_df.to_csv(output_file, index=False)

filtered_df = df[df["DX_bl"].isin(["EMCI", "LMCI"])]
output_file = "EMCI_and_LMCI_only.csv" 
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data saved to {output_file}")
