!pip install ultralytics
!pip install gradio
!pip install opencv-python
!pip install dill

from ultralytics import YOLO
import gradio as gr
import cv2
import os

model_path = 'best.pt'

if os.path.isfile(model_path):
    print(f"Loading model from {model_path}")
else:
    print(f"Model file {model_path} not found")

model = YOLO(model_path)
def predict(image):
    # Check if the input image is None (i.e., no image was uploaded)
    if image is None:
        return "Please Upload an Image to Proceed further.", None

    # Convert the input image from numpy to BGR format (if needed for YOLO)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Perform prediction on the input image
    results = model.predict(source=image_bgr)

    # Annotate the image with bounding boxes and labels
    annotated_frame = results[0].plot()  # Grabs the first (and only) result
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    # Extract information about detected objects
    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    labels = results[0].names  # Class names

    detected_objects = []
    for i, bbox in enumerate(detections):
        detected_objects.append({
            "Bounding Box": bbox.tolist(),
            "Label": labels[int(results[0].boxes.cls[i].item())],
            "Confidence Score": scores[i].item()
        })

    return annotated_frame_rgb, detected_objects

app = gr.Interface(
    fn=predict,  # The function to be called for predictions
    inputs=gr.Image(type="numpy", label="Upload Image"),  # Upload input as a NumPy array
    outputs=[
        gr.Image(type="numpy", label="Predicted Image"),  # Output the annotated image
        gr.JSON(label="Detection Results")  # Output the detection results in JSON format
    ],
    title="Plant Disease Detection App",
    description="Upload an image of a plant leaf, and the model will detect and classify any diseases. The output includes detected objects with bounding boxes, labels, and confidence scores.",
    flagging_dir="flagged_data"  # Specify a directory to save flagged data
)

# Launch the Gradio app

if __name__ == "__main__":
    app.launch(share=True)

!pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

