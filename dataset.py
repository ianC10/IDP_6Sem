import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

class PhaseOneIntegrator:
    def __init__(self, target_fs=100):
        self.fs = target_fs # 100Hz for ESP32
        
    def resample(self, df, original_fs):
        """Standardizes data to 100Hz using Cubic Spline Interpolation."""
        duration = len(df) / original_fs
        new_indices = np.linspace(0, len(df)-1, int(duration * self.fs))
        resampled_data = {col: interp1d(np.arange(len(df)), df[col], kind='cubic')(new_indices) 
                         for col in df.columns}
        return pd.DataFrame(resampled_data)

    def process_uci(self, path):
        print("⚙️ Processing UCI Multivariate Gait Data...")
        df = pd.read_csv(path)
        
        # Isolate Knee Joint (Joint 2)
        knees = df[df['joint'] == 2]
        
        # Separate Left (Leg 2) and Right (Leg 1)
        left_knee = knees[knees['leg'] == 2]['angle'].reset_index(drop=True)
        right_knee = knees[knees['leg'] == 1]['angle'].reset_index(drop=True)
        
        # Combine into a clean DataFrame
        min_len = min(len(left_knee), len(right_knee))
        uci_clean = pd.DataFrame({
            'L_Knee': left_knee.iloc[:min_len], 
            'R_Knee': right_knee.iloc[:min_len]
        })
        
        # Resample from 60Hz to 100Hz
        return self.resample(uci_clean, original_fs=60)

    def process_huga(self, path):
        print("⚙️ Processing HuGaDB Data...")
        df = pd.read_csv(path)
        
        print("\n🔍 HuGaDB Column Headers Detected:")
        print(df.columns.tolist())
        
        # For now, we will grab the first 4 columns assuming they are IMU data
        # We will refine this once you paste the exact column names!
        imu_cols = df.columns[1:5] 
        print(f"-> Temporarily using columns: {list(imu_cols)} for synchronization.")
        
        huga_clean = df[imu_cols].copy()
        
        # Resample from 50Hz to 100Hz
        return self.resample(huga_clean, original_fs=50)

    def fuse_datasets(self, uci_path, huga_path):
        uci_df = self.process_uci(uci_path)
        huga_df = self.process_huga(huga_path)
        
        # Sync to shortest length
        min_len = min(len(uci_df), len(huga_df))
        
        master = pd.concat([
            uci_df.iloc[:min_len].reset_index(drop=True),
            huga_df.iloc[:min_len].reset_index(drop=True)
        ], axis=1)
        
        # Save Phase 1 Dataset
        master.to_csv("phase1_echo_dataset.csv", index=False)
        print(f"\n✅ Phase 1 Fusion Complete! Saved {len(master)} rows to 'phase1_echo_dataset.csv'")
        
        # Plot to verify synchronization
        plt.figure(figsize=(12, 5))
        plt.plot(master['L_Knee'][:300], label='Left Knee (Master)')
        plt.plot(master['R_Knee'][:300], label='Right Knee (Slave Target)')
        plt.title("ESP32 Bilateral Target Angles (100Hz Synchronized)")
        plt.legend()
        plt.show()

# --- RUN SCRIPT ---
integrator = PhaseOneIntegrator()
integrator.fuse_datasets("gait.csv", "HuGaDB_v2_various_18_08.csv")