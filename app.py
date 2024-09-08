{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e5612d51-9fa1-4eae-a687-695a2ff87dfb",
   "metadata": {},
   "outputs": [],
   "source": [
    "from ultralytics import YOLO\n",
    "import gradio as gr\n",
    "import cv2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "abd56b9f-e1b0-4688-8161-7779c974346d",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_path = '/app/best.pt'\n",
    "model = YOLO(model_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "376a9dc0-bd7c-4a4c-beb8-2325a9c01750",
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict(image):\n",
    "    # Check if the input image is None (i.e., no image was uploaded)\n",
    "    if image is None:\n",
    "        return \"Please Upload an Image to Proceed further.\", None\n",
    "    \n",
    "    # Convert the input image from numpy to BGR format (if needed for YOLO)\n",
    "    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)\n",
    "    \n",
    "    # Perform prediction on the input image\n",
    "    results = model.predict(source=image_bgr)\n",
    "    \n",
    "    # Annotate the image with bounding boxes and labels\n",
    "    annotated_frame = results[0].plot()  # Grabs the first (and only) result\n",
    "    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)\n",
    "    \n",
    "    # Extract information about detected objects\n",
    "    detections = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes\n",
    "    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores\n",
    "    labels = results[0].names  # Class names\n",
    "    \n",
    "    detected_objects = []\n",
    "    for i, bbox in enumerate(detections):\n",
    "        detected_objects.append({\n",
    "            \"Bounding Box\": bbox.tolist(),\n",
    "            \"Label\": labels[int(results[0].boxes.cls[i].item())],\n",
    "            \"Confidence Score\": scores[i].item()\n",
    "        })\n",
    "    \n",
    "    return annotated_frame_rgb, detected_objects"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "952dd85d-76a7-426f-8bdd-2979d7c1fbc1",
   "metadata": {
    "scrolled": True
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running on local URL:  http://127.0.0.1:7860\n",
      "Running on public URL: https://ff821c3613596ec21a.gradio.live\n",
      "\n",
      "This share link expires in 72 hours. For free permanent hosting and GPU upgrades, run `gradio deploy` from Terminal to deploy to Spaces (https://huggingface.co/spaces)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"https://ff821c3613596ec21a.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "app = gr.Interface(\n",
    "    fn=predict,  # The function to be called for predictions\n",
    "    inputs=gr.Image(type=\"numpy\", label=\"Upload Image\"),  # Upload input as a NumPy array\n",
    "    outputs=[\n",
    "        gr.Image(type=\"numpy\", label=\"Predicted Image\"),  # Output the annotated image\n",
    "        gr.JSON(label=\"Detection Results\")  # Output the detection results in JSON format\n",
    "    ],\n",
    "    title=\"Plant Disease Detection App\",\n",
    "    description=\"Upload an image of a plant leaf, and the model will detect and classify any diseases. The output includes detected objects with bounding boxes, labels, and confidence scores.\",\n",
    "    flagging_dir=\"flagged_data\"  # Specify a directory to save flagged data\n",
    ")\n",
    "\n",
    "# Launch the Gradio app\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    app.launch(share=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "1f9000e7-43fe-48d9-9045-e7b9ff7d476b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
