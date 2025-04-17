import streamlit as st
import cv2
import numpy as np
from PIL import Image
import docx
import os
import re
from docx import Document

from ultralytics import YOLO

# Load YOLO model
model = YOLO("YOLOV8X.pt")  # <-- change if your model file has a different name

# Define class labels based on dataset
class_labels = {
    0: "DESIGN 1",
    1: "DESIGN 2",
    2: "DESIGN 3",
    3: "DESIGN 4",
    4: "GREENSTONE",
    5: "REDSTONE",
    6: "WHITE STONE"
}

# Detect stones and design using YOLO
def detect_stones(image_path):
    results = model.predict(source=image_path, show=True, conf=0.55, save=True, imgsz=640)

    stone_counts = {"GREENSTONE": 0, "REDSTONE": 0, "WHITE STONE": 0}
    design_detected = None

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = class_labels.get(cls_id, "Unknown")
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name in stone_counts:
                stone_counts[class_name] += 1
            elif "DESIGN" in class_name:
                design_detected = class_name

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(image_bgr, class_name, (x1+1, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(image_bgr, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    result_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_image)

    total_stones = sum(stone_counts.values())

    return {
        "Red Stones": stone_counts["REDSTONE"],
        "Green Stones": stone_counts["GREENSTONE"],
        "White Stones": stone_counts["WHITE STONE"],
        "Total Stones": total_stones,
        "Design": design_detected,
        "Image": result_pil
    }

# Extract job card data from docx only (reads tables)
def extract_job_card_data(docx_file=None):
    red = green = white = total = 0
    design = ""

    if docx_file:
        doc = Document(docx_file)
        for table in doc.tables:
            for row in table.rows:
                cells = row.cells
                if len(cells) < 2:
                    continue
                key = cells[0].text.strip().lower()
                value = cells[1].text.strip()
                
                if "red" in key:
                    red = int(value)
                elif "green" in key:
                    green = int(value)
                elif "white" in key:
                    white = int(value)
                elif "total" in key:
                    total = int(value)
                elif "design" in key:
                    design = value.upper()
                    if not design.startswith("DESIGN"):
                        design = f"DESIGN {design}"

    if total == 0:
        total = red + green + white

    return {
        "Red Stones": red,
        "Green Stones": green,
        "White Stones": white,
        "Total Stones": total,
        "Design": design
    }

# Streamlit UI
st.set_page_config(page_title="Jewel Inspection", layout="centered")
st.title("💎 Jewel Inspection")
st.markdown("Upload the **Job Card (.docx)** and **Jewelry Image** to inspect.")

col1, col2 = st.columns(2)

with col1:
    uploaded_docx = st.file_uploader("📄 Upload Job Card (.docx only)", type=["docx"])

with col2:
    uploaded_image = st.file_uploader("🖼️ Upload Jewelry Image", type=["jpg", "jpeg", "png"])

job_data = {}
result_data = {}

temp_image_path = "temp.jpg"

if uploaded_docx:
    job_data = extract_job_card_data(docx_file=uploaded_docx)

    with st.expander("📋 Job Card Details"):
        st.write(f"**White Stones:** {job_data['White Stones']}")
        st.write(f"**Red Stones:** {job_data['Red Stones']}")
        st.write(f"**Green Stones:** {job_data['Green Stones']}")
        st.write(f"**Total Stones:** {job_data['Total Stones']}")
        st.write(f"**Design Number:** {job_data['Design']}")

if uploaded_image:
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_image.read())

if uploaded_image and st.button("🔍 Inspect"):
    result_data = detect_stones(temp_image_path)
    if result_data:
        st.subheader("🧠 Detection Results")
        st.image(result_data["Image"], caption="Detected Image", use_column_width=True)

        st.markdown(f"""
        - **White Stones Detected:** {result_data['White Stones']}
        - **Red Stones Detected:** {result_data['Red Stones']}
        - **Green Stones Detected:** {result_data['Green Stones']}
        - **Total Stones Detected:** {result_data['Total Stones']}
        - **Design Detected:** {result_data['Design']}
        """)

        if job_data:
            match = (
                job_data["White Stones"] == result_data["White Stones"] and
                job_data["Red Stones"] == result_data["Red Stones"] and
                job_data["Green Stones"] == result_data["Green Stones"] and
                job_data["Design"] == result_data["Design"]
            )
            if match:
                st.success("✅ Design Matched!")
            else:
                st.error("❌ Design Not Matched!")
    else:
        st.error("⚠️ Could not process the image. Please check the format and try again.")
