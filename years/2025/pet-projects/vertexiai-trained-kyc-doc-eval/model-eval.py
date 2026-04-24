import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
from pathlib import Path
import glob

# Your label mapping
LABEL_MAP = {
    0: "pan_card",
    1: "aadhar_front",
    2: "aadhar_back",
    3: "aadhar_full"
}

class AutoMLEdgeLocalPredictor:
    def __init__(self, model_path: str):
        """Load the exported AutoML Edge .tflite model using TensorFlow Lite."""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        self.input_dtype = self.input_details[0]['dtype']
        print(f"Model input shape: {self.input_shape}, dtype: {self.input_dtype}")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image based on model's expected dtype."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = self.input_shape[1], self.input_shape[2]
        image = cv2.resize(image, (w, h))

        # Handle UINT8 vs FLOAT32 input
        if self.input_dtype == np.uint8:
            image = image.astype(np.uint8)
        else:
            image = image.astype(np.float32) / 255.0

        image = np.expand_dims(image, axis=0)
        return image

    def predict(self, image_path: str) -> dict:
        """Run prediction and return all class scores."""
        input_data = self.preprocess_image(image_path)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions = output_data[0]  # [num_classes]

        # Get top prediction
        top_idx = np.argmax(predictions)
        top_label = LABEL_MAP.get(int(top_idx), f"class_{top_idx}")
        top_confidence = float(predictions[top_idx])

        return {
            "predictions": predictions,
            "top_type": top_label,
            "top_confidence": top_confidence
        }

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

def process_all_images(model_path: str, image_folder: str = "./test-data", output_csv: str = "kyc_predictions.csv"):
    """Process all images in folder (recursive) and save results to CSV."""
    predictor = AutoMLEdgeLocalPredictor(model_path)
    
    # Find all image files recursively
    image_paths = find_all_images(image_folder)
    
    if not image_paths:
        print(f"No images found in {image_folder} or its subdirectories")
        return
    
    print(f"Found {len(image_paths)} images across {image_folder} and subdirectories")
    
    results = []
    for i, img_path in enumerate(image_paths, 1):
        try:
            print(f"Processing ({i}/{len(image_paths)}): {os.path.basename(img_path)}")
            pred_result = predictor.predict(img_path)
            predictions = pred_result["predictions"]
            
            # Create result row
            row = {
                "filepath": img_path,
                "filename": os.path.basename(img_path),
                "pan_card_score": float(predictions[0]),
                "aadhar_front_score": float(predictions[1]),
                "aadhar_back_score": float(predictions[2]),
                "aadhar_full_score": float(predictions[3]),
                "type": pred_result["top_type"]
            }
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            # Add row with error info
            row = {
                "filepath": img_path,
                "filename": os.path.basename(img_path),
                "pan_card_score": 0.0,
                "aadhar_front_score": 0.0,
                "aadhar_back_score": 0.0,
                "aadhar_full_score": 0.0,
                "type": "ERROR"
            }
            results.append(row)
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to {output_csv} ({len(results)} images processed)")
    
    # Summary stats
    print("\n📊 Summary by predicted type:")
    print(df.groupby("type").size().sort_values(ascending=False))
    
    print("\n🏆 Top 5 highest confidence predictions:")
    top_conf = df.nlargest(5, 'pan_card_score')
    print(top_conf[["filename", "type", "pan_card_score", "aadhar_front_score"]].to_string(index=False))

def main():
    model_path = (
        "./models/v2/"
        "trainmodel_800_Images_08122025_model-8734492883261849600_tflite_2025-12-08T04_22_54.603084Z_model.tflite"
    )
    process_all_images(model_path, "/mnt/kyc-data", "kyc_predictions.csv")

if __name__ == "__main__":
    main()
