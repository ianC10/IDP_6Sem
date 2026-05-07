import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

class GaitDataAnalyzer:
    def __init__(self, sample_rate=100):
        self.fs = sample_rate
        self.scaler = MinMaxScaler()
        self.master_df = pd.DataFrame()

    def analyze_dataset(self, df, name="Dataset"):
        """Generates statistical summaries and basic plots."""
        print(f"\n--- Analysis for {name} ---")
        print(df.describe())
        
        # Plot first 500 samples to see waveform
        plt.figure(figsize=(12, 4))
        for col in df.columns[:3]: # Plot first 3 features
            plt.plot(df[col][:500], label=col)
        plt.title(f"Sensor Waveforms: {name}")
        plt.legend()
        plt.show()

    def standardize_frequency(self, df, original_fs):
        """Resamples data to match our target ESP32 frequency (100Hz)."""
        duration = len(df) / original_fs
        new_indices = np.linspace(0, len(df)-1, int(duration * self.fs))
        resampled_data = {}
        for col in df.columns:
            f = interp1d(np.arange(len(df)), df[col], kind='cubic')
            resampled_data[col] = f(new_indices)
        return pd.DataFrame(resampled_data)

    def process_uci_data(self, file_path):
        """Loads UCI Gait data: Focuses on Bilateral Joint Angles."""
        # Assuming CSV structure: [time, L_Ankle, L_Knee, L_Hip, R_Ankle, R_Knee, R_Hip]
        df = pd.read_csv(file_path)
        df = self.standardize_frequency(df, original_fs=60) # UCI is often 60Hz
        self.analyze_dataset(df, "UCI Bilateral Angles")
        return df[['L_Knee', 'R_Knee']] # Focus on the joint we are echoing

    def process_physionet_imu(self, file_path):
        """Loads IMU/FSR data: Focuses on raw sensor inputs."""
        # Assuming CSV structure: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, fsr_heel, fsr_toe]
        df = pd.read_csv(file_path)
        df = self.standardize_frequency(df, original_fs=200) # PhysioNet often 200Hz
        self.analyze_dataset(df, "PhysioNet IMU/FSR")
        return df

    def compile_master_dataset(self, angles_df, sensors_df):
        """Merges Kinematic (Angles) and Kinetic (Sensors) into one file."""
        min_len = min(len(angles_df), len(sensors_df))
        
        # Combine: Input (Healthy IMU + FSR) -> Output (Target Prosthetic Angle)
        combined = pd.concat([
            sensors_df.iloc[:min_len].reset_index(drop=True),
            angles_df.iloc[:min_len].reset_index(drop=True)
        ], axis=1)
        
        # Normalize all values to 0-1 for ESP32 TinyML efficiency
        combined_scaled = pd.DataFrame(
            self.scaler.fit_transform(combined), 
            columns=combined.columns
        )
        
        combined_scaled.to_csv("echo_gait_master_dataset.csv", index=False)
        print("\nSUCCESS: Master Dataset compiled into 'echo_gait_master_dataset.csv'")
        return combined_scaled

# --- EXECUTION ---
# analyzer = GaitDataAnalyzer()
# uci_angles = analyzer.process_uci_data('uci_data.csv')
# sensor_inputs = analyzer.process_physionet_imu('physionet_data.csv')
# master = analyzer.compile_master_dataset(uci_angles, sensor_inputs)