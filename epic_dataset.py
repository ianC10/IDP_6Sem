import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def sync_epic_data():
    print("🔄 Loading EPIC CSVs...")
    imu = pd.read_csv('epic_imu.csv')
    fp = pd.read_csv('epic_fp.csv')
    ik = pd.read_csv('epic_ik.csv')

    # Target: 100Hz for ESP32 (10ms steps)
    target_fs = 100 
    
    # EPIC Default Samplerates
    imu_fs = 200
    fp_fs = 1000
    ik_fs = 200

    def resample(df, original_fs, name):
        duration = len(df) / original_fs
        new_indices = np.linspace(0, len(df)-1, int(duration * target_fs))
        resampled = {col: interp1d(np.arange(len(df)), df[col], kind='linear')(new_indices) 
                    for col in df.columns}
        return pd.DataFrame(resampled)

    print("📏 Resampling to 100Hz...")
    imu_100 = resample(imu, imu_fs, "IMU")
    fp_100 = resample(fp, fp_fs, "Force")
    ik_100 = resample(ik, ik_fs, "Kinematics")

    # Find shortest length to sync
    min_len = min(len(imu_100), len(fp_100), len(ik_100))
    
    # FUSE THE BRAIN
    # We select the most critical features for the prosthetic
    master = pd.DataFrame({
        'accel_z': imu_100['shank_Accel_Z'].iloc[:min_len],
        'gyro_y':  imu_100['shank_Gyro_Y'].iloc[:min_len],
        'force_z': fp_100['force_z'].iloc[:min_len], # Vertical impact
        'target_knee': ik_100['knee_angle_r'].iloc[:min_len] # The angle to mimic
    })

    master.to_csv("FINAL_TRAINING_DATA.csv", index=False)
    print(f"🎉 DONE! 'FINAL_TRAINING_DATA.csv' created with {min_len} synchronized rows.")

sync_epic_data()