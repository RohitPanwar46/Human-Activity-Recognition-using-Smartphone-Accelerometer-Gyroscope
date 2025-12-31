import os
import pandas as pd

def merge_accelerometer_gyroscope(FolderPath):
    
    acc = pd.read_csv(FolderPath + "Accelerometer.csv")
    gyro = pd.read_csv(FolderPath + "Gyroscope.csv")

    acc = acc.rename(columns={
        "Acceleration x (m/s^2)": "ax",
        "Acceleration y (m/s^2)": "ay",
        "Acceleration z (m/s^2)": "az"
    })

    gyro = gyro.rename(columns={
        "Gyroscope x (rad/s)": "gx",
        "Gyroscope y (rad/s)": "gy",
        "Gyroscope z (rad/s)": "gz"
    })

    merged = pd.merge_asof(
        acc.sort_values("Time (s)"),
        gyro.sort_values("Time (s)"),
        on="Time (s)",
        direction="nearest",
        tolerance=0.01
    )

    merged = merged.dropna()

    merged = merged[
        ~(
            (merged["ax"] == 0) &
            (merged["ay"] == 0) &
            (merged["az"] == 0) &
            (merged["gx"] == 0) &
            (merged["gy"] == 0) &
            (merged["gz"] == 0)
        )
    ]

    merged = merged.reset_index(drop=True)

    return merged


def trim_df(df, start_trim, end_trim):
    t_start = start_trim
    t_end = df["Time (s)"].iloc[-1] - end_trim

    trimmed_df = df[
        (df["Time (s)"] >= t_start) &
        (df["Time (s)"] <= t_end)
    ].reset_index(drop=True)

    return trimmed_df


def get_folders_with_files(base_path, file1, file2):
    result = []
    for entry in os.scandir(base_path):
        if entry.is_dir():
            path1 = os.path.join(entry.path, file1)
            path2 = os.path.join(entry.path, file2)
            if os.path.isfile(path1) and os.path.isfile(path2):
                result.append(entry.name)
    return result

def save_df_to_csv():
    Start_trim = 5  # seconds
    End_trim = 5    # seconds
    dataFrames = {}
    
    data_folders = get_folders_with_files("data", "Accelerometer.csv", "Gyroscope.csv")

    for folder in data_folders:
        merged_df = merge_accelerometer_gyroscope(os.path.join("data", folder) + "/")
        trimmed_df = trim_df(merged_df, Start_trim, End_trim)
        trimmed_df.to_csv(os.path.join("data", folder, "trimmed_merged_data.csv"), index=False)
        dataFrames[folder] = trimmed_df
        
    return dataFrames


if __name__ == "__main__":
    dfs_dict = save_df_to_csv()
    for key, df in dfs_dict.items():
        print(f"Data from folder: {key}")
        print(df.head())

    