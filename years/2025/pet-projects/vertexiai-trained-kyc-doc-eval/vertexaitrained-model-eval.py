
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os
import pandas as pd
from pathlib import Path

#MODEL_PATH = "./models/v2/trainmodel_800_Images_08122025_model-8734492883261849600_tflite_2025-12-08T04_22_54.603084Z_model.tflite"
MODEL_PATH = "./models/v2/model-6344911068476735488_tflite_2025-12-10T12_29_46.318311Z_model.tflite"


# Same alphabetical label order as training
labels = [
    "aadhar-front",
    "aadhar-back",
    "pan-front",
    "aadhar-full"
]

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image_structured(image_path):
    start_time = time.time()

    img = Image.open(image_path).convert("RGB")

    h, w = input_details[0]["shape"][1], input_details[0]["shape"][2]
    img = img.resize((w, h))

    img_array = np.array(img, dtype=np.uint8)
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    end_time = time.time()
    time_taken = round(end_time - start_time, 3)

    image_name = os.path.basename(image_path)

    # Get max confidence score
    max_conf = float(np.max(output))
    max_index = int(np.argmax(output))
    predicted_label = labels[max_index]

    # TRUE/FALSE per label
    scores_bool = ["TRUE" if i == max_index else "FALSE" for i in range(len(output))]

    return {
        "image_name": image_path,
        labels[0]: scores_bool[0],
        labels[1]: scores_bool[1],
        labels[2]: scores_bool[2],
        labels[3]: scores_bool[3],

        "predicted_label": predicted_label,
        "max_confidence": max_conf,
        "time_taken_in_sec": time_taken
    }




folder_path = "/mnt/kyc-data"   # CHANGE HERE

def find_all_images(folder_path: str) -> list:
    """Recursively find all image files in folder and subfolders."""
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"]
    image_paths = []
    
    # Use pathlib for recursive glob
    folder = Path(folder_path)
    for ext in image_extensions:
        image_paths.extend(folder.glob(f"**/{ext}"))
    
    # Convert Path objects to strings and sort for consistent order
    image_paths = sorted([str(p) for p in image_paths])
    return image_paths

image_paths = find_all_images(folder_path)

results = []

for i, img_path in enumerate(image_paths, 1):
    print(f"Processing ({i}/{len(image_paths)}): {os.path.basename(img_path)}")
    res = predict_image_structured(img_path)
    results.append(res)
    print(res)   # show progress


df = pd.DataFrame(results)
df.to_csv("results_final_V11.csv", index=False)
 

