import pandas as pd

# Load your CSV file
df = pd.read_csv(r'C:\....\AIBL\aibl_19Sep2019\Data_extract_3.3.0\aibl_pdxconv_01-Jun-2018.csv')  #from imaging data folder from AIBL

# ***** Use the block below to get RIDs of Subjects for downloading *****


# *************** AD here begins ***************
filtered = df[(df['VISCODE'] == 'bl') & (df['DXCURREN'] == 3)]
rid_values = filtered['RID'].unique().tolist()
print("RID values where VISCODE == 'bl' and DXCURREN == 3:")
print(rid_values)
# *************** AD here ends ***************

# *************** MCI here begins ***************
filtered = df[(df['VISCODE'] == 'bl') & (df['DXCURREN'] == 2)]
rid_values = filtered['RID'].unique().tolist()
print("RID values where VISCODE == 'bl' and DXCURREN == 2:")
print(rid_values) 
# *************** MCI here ends ***************

# *************** CN here begins ***************
filtered = df[(df['VISCODE'] == 'bl') & (df['DXCURREN'] == 1)]
rid_values = filtered['RID'].unique().tolist()

# Sort the CN subjects based on the number of available scans
rid_timepoint_counts = {
    rid: len(df[df['RID'] == rid])
    for rid in rid_values
}
# Sort RIDs by number of VISCODEs (time points), descending
sorted_rids = sorted(rid_timepoint_counts.items(), key=lambda x: x[1], reverse=True)
sorted_CN_rids = [rid for rid, count in sorted_rids]
# print(sorted_CN_rids[:496])
# *************** CN here ends **************