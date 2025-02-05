import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ✅ Load YOLO Model
@st.cache_resource
def load_model():
    model_path = r"E:\New Folder\drone dataset\Yolo__Data\Building_detection _drone_model.pt"  # Ensure correct path
    return YOLO(model_path)

model = load_model()

# 🎨 **Streamlit UI**
st.title("🏢 Building Detection from Drone Images")
st.subheader("Upload an aerial image to detect buildings")

# Sidebar
with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ✅ Image Upload
uploaded_file = st.file_uploader("Upload an aerial image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # **Step 1: Read Image**
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # **Step 2: Run Detection**
    results = model.predict(img_array, conf=confidence)

    # **Step 3: Draw bounding boxes and zoom detected areas**
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)

    colors = ["red", "blue"]  # 🔥 Alternate between red & blue

    for i, result in enumerate(results):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            
            # 🔥 Assign color (Red for even, Blue for odd)
            box_color = colors[i % len(colors)]

            # **Draw bounding box**
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            # **Step 4: Crop and Zoom**
            cropped_building = image.crop((x1, y1, x2, y2))  # Crop the detected building
            zoomed_building = cropped_building.resize((x2-x1+50, y2-y1+50))  # Resize for zoom effect

            # **Step 5: Paste Zoomed Image**
            annotated_img.paste(zoomed_building, (x1, max(0, y1 - 50)))  # Place above detected area

    # **Step 6: Display Results**
    st.image(annotated_img, caption="Detected Buildings with Zoom & Colored Boxes", use_column_width=True)

    # **Show Detection Details**
    st.subheader("Detection Statistics")
    num_buildings = len(results[0].boxes)
    st.write(f"Detected Buildings: {num_buildings}")

# ✅ Footer
st.markdown("""
<div style="background-color:black; color:white; padding:10px; text-align:center;">
    Made with ❤️ by Mohid Khan | Data Scientist
</div>
""", unsafe_allow_html=True)
