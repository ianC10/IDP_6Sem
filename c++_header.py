import os

# 1. Read the binary TFLite model
with open("prosthetic_model.tflite", "rb") as f:
    tflite_content = f.read()

# 2. Format it as a C++ hex array
hex_lines = [', '.join([f'0x{b:02x}' for b in tflite_content[i:i+12]]) for i in range(0, len(tflite_content), 12)]
hex_array = ',\n  '.join(hex_lines)

# 3. Write out the .h file
with open("model_data.h", "w") as f:
    f.write("const unsigned char prosthetic_model_tflite[] = {\n  ")
    f.write(hex_array)
    f.write("\n};\n")
    f.write(f"const unsigned int prosthetic_model_tflite_len = {len(tflite_content)};\n")

print("✅ model_data.h generated! Ready for Arduino IDE.")