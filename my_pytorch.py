import os
import json
import torch
import torchvision
import argparse
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw

# Argument Parser for Confidence Threshold
parser = argparse.ArgumentParser()
parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold for object detection")
args = parser.parse_args()
confidence_threshold = args.conf
print(f"Confidence threshold set to {confidence_threshold}")

# COCO Class Labels (Only 4 Categories) 
COCO_LABELS = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle"
}
VALID_LABELS = set(COCO_LABELS.keys())

# Load Pre-trained Faster R-CNN Model
def load_detection_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval() 
    return model

# Detect Objects in an Image
def detect_objects(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    # Extract bounding boxes, scores, and labels
    boxes = predictions[0]['boxes'].tolist()
    scores = predictions[0]['scores'].tolist()
    labels = predictions[0]['labels'].tolist()

    # Filter only valid objects based on confidence and category
    filtered_boxes = []
    filtered_labels = []
    
    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold and label in VALID_LABELS:
            filtered_boxes.append(box)
            filtered_labels.append(COCO_LABELS[label])  # Convert ID to name

    return filtered_boxes, filtered_labels, image

# Draw Transparent Boxes with Red Borders
def draw_transparent_boxes(image, boxes, labels):
    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = map(int, box)

        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.rectangle([x_min, y_min, x_max, y_max], fill=(255, 0, 0, 50))
        draw.text((x_min, y_min - 10), label, fill="red")

    return Image.alpha_composite(image, overlay).convert("RGB")

# Save Annotations Automatically
def save_annotations(image_path, boxes, labels, output_annotations_folder):
    os.makedirs(output_annotations_folder, exist_ok=True)

    annotation_data = {
        "image": os.path.basename(image_path),
        "objects": [{"label": label, "bbox": box} for box, label in zip(boxes, labels)]
    }

    annotation_filename = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    annotation_path = os.path.join(output_annotations_folder, annotation_filename)

    with open(annotation_path, "w") as f:
        json.dump(annotation_data, f, indent=4)

    print(f"Annotation saved: {annotation_path}")

# Process All Images in a Folder
def process_images_in_folder(input_folder, output_folder, annotation_folder, detection_model):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(annotation_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print("No images found in the folder!")
        return

    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        boxes, labels, image = detect_objects(detection_model, image_path)

        if not boxes:
            print(f"No valid objects detected in {image_file}, skipping.")
            continue

        output_image = draw_transparent_boxes(image, boxes, labels)
        output_filename = f"processed_{image_file}"
        output_path = os.path.join(output_folder, output_filename)
        output_image.save(output_path)

        save_annotations(image_path, boxes, labels, annotation_folder)

        print(f"mage processed & saved: {output_path}")

# Main Function 
if __name__ == "__main__":
    input_folder = "./objects"
    output_folder = "./output"
    annotation_folder = "./annotations"

    detection_model = load_detection_model()

    process_images_in_folder(input_folder, output_folder, annotation_folder, detection_model)

    print("All images processed successfully!")
