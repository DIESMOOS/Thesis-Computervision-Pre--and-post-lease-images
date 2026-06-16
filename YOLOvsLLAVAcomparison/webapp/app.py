import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import shutil

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

PROPERTIES_DIR = ROOT / "data" / "properties"

YOLO_MODEL_PATH = ROOT / "models" / "best.pt"

CLASS_NAMES = {
    0: "damage",
    1: "crack",
    2: "mold",
    3: "wear",
    4: "asbestos"
    
}

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(str(YOLO_MODEL_PATH))

model = load_model()

# -------------------------------------------------
# PAGE
# -------------------------------------------------

st.set_page_config(layout="wide")

st.title("Housing Inspection Assistant")

# -------------------------------------------------
# PROPERTY SELECTION
# -------------------------------------------------

property_ids = sorted([
    p.name
    for p in PROPERTIES_DIR.iterdir()
    if p.is_dir()
])

selected_property = st.selectbox(
    "Select Property",
    property_ids
)

property_path = PROPERTIES_DIR / selected_property

pre_lease_path = property_path / "pre_lease"
post_lease_path = property_path / "post_lease"

# -------------------------------------------------
# SHOW PRE LEASE
# -------------------------------------------------

st.header("Pre-Lease Images")

pre_images = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    pre_images.extend(pre_lease_path.glob(ext))

cols = st.columns(4)

for i, img_path in enumerate(pre_images):

    with cols[i % 4]:
        st.image(
            str(img_path),
            caption=img_path.name,
            use_container_width=True
        )

# -------------------------------------------------
# UPLOAD POST LEASE
# -------------------------------------------------

st.header("Upload New Post-Lease Images")

uploaded_files = st.file_uploader(
    "Select images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:

    post_lease_path.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in uploaded_files:

        destination = post_lease_path / file.name

        with open(destination, "wb") as f:
            f.write(file.getbuffer())

    st.success(
        f"{len(uploaded_files)} image(s) uploaded."
    )

# -------------------------------------------------
# SHOW POST LEASE
# -------------------------------------------------

st.header("Current Post-Lease Images")

post_images = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    post_images.extend(post_lease_path.glob(ext))

cols = st.columns(4)

for i, img_path in enumerate(post_images):

    with cols[i % 4]:

        st.image(
            str(img_path),
            caption=img_path.name,
            use_container_width=True
        )

# -------------------------------------------------
# RUN YOLO
# -------------------------------------------------

if st.button("Run YOLO Inspection"):

    st.header("YOLO Results")

    total_findings = {}

    for image_path in post_images:

        results = model.predict(
            str(image_path),
            conf=0.25,
            verbose=False
        )

        result = results[0]

        st.subheader(image_path.name)

        plotted = result.plot()

        st.image(
            plotted,
            use_container_width=True
        )

        if len(result.boxes) == 0:

            st.success("No damage detected")

            continue

        detections = []

        for box in result.boxes:

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            label = CLASS_NAMES.get(
                cls,
                str(cls)
            )

            detections.append(
                f"{label} ({conf:.2f})"
            )

            total_findings[label] = (
                total_findings.get(label, 0) + 1
            )

        for d in detections:
            st.write("•", d)

    st.divider()

    st.header("Property Summary")

    if total_findings:

        for k, v in total_findings.items():

            st.write(
                f"{k}: {v}"
            )

    else:

        st.success(
            "No inspection findings."
        )