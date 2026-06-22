import tensorflow as tf

# 1. Load your trained model WITHOUT the training compiler/metrics
# THIS IS THE FIX: compile=False
model = tf.keras.models.load_model("epic_prosthetic_model.h5", compile=False)

# 2. Initialize the converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Apply Edge Optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Enable specific operations needed for LSTM layers
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS    
]

# 5. Execute the conversion
tflite_model = converter.convert()

# 6. Save the binary file
with open("prosthetic_model.tflite", "wb") as f:
    f.write(tflite_model)
    
print("✅ TFLite model generated successfully!")