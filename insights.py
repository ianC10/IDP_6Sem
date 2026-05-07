import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. SETUP: Create images directory
if not os.path.exists('images'):
    os.makedirs('images')

# 2. LOAD & PRE-PROCESS
df = pd.read_csv('FINAL_TRAINING_DATA.csv')
# Define Gait Phase (Stance vs Swing) using force threshold
threshold = df['force_z'].mean() * 0.2
df['gait_phase'] = (df['force_z'] > threshold).map({True: 'Stance', False: 'Swing'})
df['gait_phase_num'] = (df['force_z'] > threshold).astype(int)

# ==========================================
# 3. GRAPH GENERATION (EDA)
# ==========================================

# Graph 1: Bilateral Kinematic Symmetry (The Echo)
plt.figure(figsize=(12, 6))
plt.plot(df['Time'][:500], df['master_knee'][:500], label='Master (Left Knee)', color='blue', alpha=0.8)
plt.plot(df['Time'][:500], df['target_knee'][:500], label='Slave (Right Knee)', color='red', linestyle='--', alpha=0.8)
plt.title('Bilateral Kinematic Symmetry (Left vs Right Knee)')
plt.xlabel('Time (s)')
plt.ylabel('Angle (Degrees)')
plt.legend()
plt.savefig('images/1_kinematic_symmetry.png', dpi=300)

# Graph 2: Sensor Feature Importance
features = ['accel_z', 'gyro_y', 'force_z', 'master_knee']
X, y = df[features], df['target_knee']
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(10, 6))
importances.plot(kind='barh', color='teal')
plt.title('Sensor Vitality for Knee Prediction')
plt.savefig('images/2_feature_importance.png', dpi=300)

# Graph 3: Gait Cycle Phase Portrait (State Space)
plt.figure(figsize=(10, 8))
plt.scatter(df['accel_z'], df['gyro_y'], c=(df['gait_phase'] == 'Stance'), cmap='coolwarm', alpha=0.4, s=10)
plt.title('Gait Cycle Phase Portrait (Accel Z vs Gyro Y)')
plt.xlabel('Linear Acceleration Z')
plt.ylabel('Angular Velocity Y')
plt.savefig('images/3_phase_portrait.png', dpi=300)

# Graph 4: Force vs Angle Relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='force_z', y='target_knee', hue='gait_phase', alpha=0.5)
plt.title('Knee Angle vs Ground Reaction Force')
plt.savefig('images/4_force_angle_relation.png', dpi=300)

# Graph 5: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=['gait_phase', 'Time']).corr(), annot=True, cmap='RdBu', center=0)
plt.title('Sensor Correlation Heatmap')
plt.savefig('images/5_correlation_heatmap.png', dpi=300)

# Graph 6: IMU Impact Timing (Shock vs Stance)
plt.figure(figsize=(12, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()
ax1.plot(df['Time'][:300], df['accel_z'][:300], color='black', label='Accel Z')
ax2.plot(df['Time'][:300], df['force_z'][:300], color='green', label='Force Z', alpha=0.6)
plt.title('Impact Timing: Acceleration Spike vs Force Loading')
plt.savefig('images/6_impact_timing.png', dpi=300)

# Graph 7: Distribution of Knee Angles by Gait Phase
plt.figure(figsize=(10, 6))
sns.boxplot(x='gait_phase', y='target_knee', data=df, palette='Set2')
plt.title('Knee Angle Distribution by Gait Phase')
plt.savefig('images/7_angle_distribution.png', dpi=300)

plt.close('all')

# 4. EXPLANATION FILE GENERATION
explanation = """
GAIT ANALYSIS GRAPH EXPLANATIONS
--------------------------------
1. Kinematic Symmetry: Shows how the 'Slave' knee mimics the 'Master' with a phase shift.
2. Feature Importance: Identifies 'Master Knee' and 'Force' as top predictors.
3. Phase Portrait: Visualizes the 'Walking Orbit' of the IMU sensors.
4. Force/Angle Relation: Shows joint behavior under loading.
5. Heatmap: Statistical inter-dependencies of sensors.
6. Impact Timing: Crucial for fine-tuning heel-strike detection.
7. Boxplots: ROM (Range of Motion) stats for Stance vs Swing.
"""
with open('graph_explanations.txt', 'w') as f: f.write(explanation)

# ==========================================
# 4. NEW: SUMMARY & CLASSIFICATION IMAGES
# ==========================================

# A. Generate Classification Report Data
clf_features = ['accel_z', 'gyro_y', 'master_knee']
X_clf, y_clf = df[clf_features], df['gait_phase_num']
X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = clf.predict(X_test)
report_str = classification_report(y_test, y_pred, target_names=['Swing', 'Stance'])

# B. Save Summary Image (Graph 8)
desc_summary = df[['accel_z', 'gyro_y', 'force_z', 'master_knee', 'target_knee']].describe().to_string()
plt.figure(figsize=(10, 6))
plt.text(0.05, 0.95, "Biomechanical Statistical Summary", fontsize=14, fontweight='bold', va='top')
plt.text(0.05, 0.85, desc_summary, fontsize=10, family='monospace', va='top')
plt.axis('off')
plt.savefig('images/8_descriptive_summary.png', dpi=300, bbox_inches='tight')

# C. Save Classification Report Image (Graph 9)
plt.figure(figsize=(8, 5))
plt.text(0.05, 0.95, "Gait Phase Classification (Predictive Accuracy)", fontsize=14, fontweight='bold', va='top')
plt.text(0.05, 0.85, report_str, fontsize=10, family='monospace', va='top')
plt.axis('off')
plt.savefig('images/9_classification_report.png', dpi=300, bbox_inches='tight')

print("✅ Analysis expanded. Images 8 and 9 are now in the /images folder.")

print("✅ EDA Complete. Check the 'images/' folder and 'graph_explanations.txt'.")