import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SpaceGaurd AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model():
    """Loads the YOLOv8 model from the specified path."""
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_yolo_model()

# --- HELPER FUNCTION FOR DETECTION & DISPLAY ---
# This function is unchanged and remains efficient.
def display_detection_results(image, confidence):
    """
    Performs object detection on an image and displays the results side-by-side.
    """
    if not model:
        st.error("Model is not loaded. Cannot perform detection.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.header("Original")
        st.image(image, use_container_width=True, caption="Your input image.")

    with col2:
        st.header("Detection Results")
        with st.spinner("Analyzing image... Please stand by."):
            results = model.predict(source=image, conf=confidence, save=False)
            
            if not results or len(results[0].boxes) == 0:
                st.warning("No objects were detected with the current confidence threshold.")
                st.image(image, use_container_width=True, caption="No detections found.")
                return

            result_img_array = results[0].plot()
            result_img = Image.fromarray(result_img_array[..., ::-1])
            st.image(result_img, caption="Detected safety equipment.", use_container_width=True)

            with st.expander("📊 View Detection Details"):
                num_detections = len(results[0].boxes)
                st.write(f"**Total Detections:** {num_detections}")
                names = model.names
                for box in results[0].boxes:
                    class_name = names[int(box.cls)]
                    conf = box.conf.item()
                    st.write(f"- **{class_name}**: {conf:.2f} confidence")

# --- UI COMPONENTS ---
st.title("🛰️ SpaceGaurd AI")
st.markdown("""
Welcome, Commander! Use this interface to detect critical safety equipment.
**Upload an image** or use your **live camera** to identify objects like oxygen tanks and fire alarms.
""")

st.sidebar.header("⚙️ Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
st.sidebar.markdown("---")

# --- MAIN INTERFACE WITH TABS ---
tab1, tab2 = st.tabs(["📁 Upload an Image", "📸 Live Camera"])

# --- TAB 1: FILE UPLOAD (Unchanged) ---
with tab1:
    st.header("Analyze an Image File")
    uploaded_file = st.file_uploader(
        "Choose an image file...",
        type=["jpg", "jpeg", "png", "webp"],
        help="Drag and drop an image of a station module."
    )
    if uploaded_file:
        uploaded_image = Image.open(uploaded_file)
        display_detection_results(uploaded_image, confidence_threshold)
    else:
        st.info("Please upload an image to begin analysis.", icon="⬆️")

# --- TAB 2: CAMERA INPUT with Start/Stop Buttons ---
with tab2:
    st.header("Analyze from Camera Snapshot")

    # Initialize the session state for camera activation
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False

    # Define callback functions to change the state
    def start_camera():
        st.session_state.camera_active = True

    def stop_camera():
        st.session_state.camera_active = False

    # Display buttons and camera based on the state
    if not st.session_state.camera_active:
        st.button("Start Camera", on_click=start_camera, type="primary")
        st.info("Click 'Start Camera' to activate your webcam.", icon="▶️")
    else:
        st.button("Stop Camera", on_click=stop_camera)
        
        st.info("Camera is active. Position the object and take a photo.", icon="📸")
        camera_photo = st.camera_input(
            "Point camera at an object and take a photo",
            key="webcam_photo",
            help="Allow browser access to your camera to enable this feature.",
            on_change=stop_camera # Optional: stop camera after taking a photo
        )

        if camera_photo:
            camera_image = Image.open(camera_photo)
            display_detection_results(camera_image, confidence_threshold)
            # The camera widget will reset, and thanks to on_change, the state is set to inactive.
            # The user will see the results and the 'Start Camera' button again.