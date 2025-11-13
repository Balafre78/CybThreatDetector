import kagglehub
import os
import pandas as pd
path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")

print("Path to dataset files:", path)

csv_files = []
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            csv_files.append(os.path.join(path, file))

dataframes = [pd.read_csv(file) for file in csv_files]
# df1 = Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
# df2 = Monday-WorkingHours.pcap_ISCX.csv
# df3 = Friday-WorkingHours-Morning.pcap_ISCX.csv
# df4 = Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
# df5 = Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
# df6 = Tuesday-WorkingHours.pcap_ISCX.csv
# df7 = Wednesday-workingHours.pcap_ISCX.csv
# df8 = Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv

print(f"{len(dataframes)} CSV files loaded.")

# Merging all files
df_train = pd.concat(dataframes, ignore_index=True)
# Save the new merged dataset in order to avoid downloading everytime the same files
df_train.to_csv("cyberdataset_train.csv", index=False)