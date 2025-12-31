import os
import pandas as pd 
import numpy as np

def create_windows(data, window_size, step_size):
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        windows.append(data.iloc[start:end])
    return windows

def extract_features(window):
    # Accelerometer
    ax = window["ax"].values
    ay = window["ay"].values
    az = window["az"].values

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    # Gyroscope
    gx = window["gx"].values
    gy = window["gy"].values
    gz = window["gz"].values

    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    return {
        # Accelerometer features
        "mean_ax": np.mean(ax),
        "std_ax":  np.std(ax),
        "mean_ay": np.mean(ay),
        "std_ay":  np.std(ay),
        "mean_az": np.mean(az),
        "std_az":  np.std(az),
        "mean_acc_mag": np.mean(acc_mag),
        "std_acc_mag":  np.std(acc_mag),

        # Gyroscope features
        "mean_gx": np.mean(gx),
        "std_gx":  np.std(gx),
        "mean_gy": np.mean(gy),
        "std_gy":  np.std(gy),
        "mean_gz": np.mean(gz),
        "std_gz":  np.std(gz),
        "mean_gyro_mag": np.mean(gyro_mag),
        "std_gyro_mag":  np.std(gyro_mag),
    }

def get_final_df():
    WINDOW_SIZE = 100   # 2 seconds * 50 Hz
    STEP_SIZE = 50      # 50% overlap
    # discover all trimmed_merged_data.csv files under data/ and its subfolders
    data_root = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "data")
    data_root = os.path.abspath(data_root)

    train_rows = []
    test_rows = []

    for root, dirs, files in os.walk(data_root):
        for fname in files:
            if fname.lower().startswith("trimmed_merged") and fname.lower().endswith('.csv'):
                fpath = os.path.join(root, fname)
                try:
                    df = pd.read_csv(fpath)
                except Exception:
                    continue

                # infer label and split from parent folder name
                parent = os.path.basename(root).lower()
                label = None
                split = 'train'
                if 'walking' in parent:
                    label = 'walking'
                elif 'sitting' in parent:
                    label = 'sitting'
                elif 'standing' in parent:
                    label = 'standing'
                elif 'stairs' in parent:
                    label = 'stairs'
                elif 'running' in parent:
                    label = 'running'

                if 'test' in parent:
                    split = 'test'

                if label is None:
                    # try to find label token in filepath
                    for tok in ['walking','sitting','standing','stairs','running']:
                        if tok in fpath.lower():
                            label = tok
                            break

                if label is None:
                    continue

                windows = create_windows(df, WINDOW_SIZE, STEP_SIZE)
                feature_rows = [extract_features(w) for w in windows]
                for r in feature_rows:
                    r['label'] = label

                if split == 'test':
                    test_rows.extend(feature_rows)
                else:
                    train_rows.extend(feature_rows)

    final_df = pd.DataFrame(train_rows)
    final_test_df = pd.DataFrame(test_rows)

    data = {
        'training_df': final_df,
        'testing_df': final_test_df
    }

    return data

if __name__ == "__main__":
    data = get_final_df()
    print("Training DataFrame Preview:")
    print(data["training_df"].head())
    print(data["training_df"].shape)
    print(data["training_df"]['label'].value_counts())
    
    print("\nTesting DataFrame Preview:")
    print(data["testing_df"].head())
    print(data["testing_df"].shape)
    print(data["testing_df"]['label'].value_counts())
    
    print("\nMissing Values:")
    print(data["training_df"].isna().sum().sum())
    print(data["testing_df"].isna().sum().sum())