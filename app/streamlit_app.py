import json
from pathlib import Path
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from ultralytics import YOLO


# ============================================================
# Streamlit Page Settings
# ============================================================

st.set_page_config(
    page_title="Smartphone AI System",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Custom UI Styling
# ============================================================

st.markdown(
    """
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #eef3f8 45%, #ffffff 100%);
        color: #111827;
    }

    /* Hide Streamlit default menu/footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main block spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 38px;
        border-radius: 28px;
        color: white;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
        margin-bottom: 26px;
    }

    .hero-title {
        font-size: 48px;
        font-weight: 900;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        font-size: 18px;
        color: #dbeafe;
        line-height: 1.6;
        max-width: 850px;
    }

    .tag-row {
        margin-top: 22px;
    }

    .tag {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 8px 14px;
        border-radius: 999px;
        margin-right: 8px;
        margin-bottom: 8px;
        color: #ffffff;
        font-size: 14px;
        font-weight: 600;
    }

    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
        height: 100%;
    }

    .small-card {
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 20px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .stage-number {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: #2563eb;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        font-size: 18px;
        margin-bottom: 14px;
    }

    .card-title {
        font-size: 21px;
        font-weight: 850;
        color: #0f172a;
        margin-bottom: 8px;
    }

    .card-text {
        font-size: 15px;
        color: #475569;
        line-height: 1.55;
    }

    /* Result Cards */
    .result-card {
        background: white;
        border-radius: 24px;
        padding: 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
        margin-bottom: 18px;
    }

    .result-card-good {
        border-left: 8px solid #16a34a;
    }

    .result-card-warning {
        border-left: 8px solid #f59e0b;
    }

    .result-card-error {
        border-left: 8px solid #dc2626;
    }

    .result-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .result-value {
        font-size: 28px;
        color: #0f172a;
        font-weight: 900;
        margin-bottom: 4px;
    }

    .result-sub {
        color: #475569;
        font-size: 15px;
    }

    /* Final Summary */
    .final-box {
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 28px;
        padding: 28px;
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.12);
        margin-top: 18px;
    }

    .final-title {
        font-size: 26px;
        font-weight: 900;
        color: #1e3a8a;
        margin-bottom: 16px;
    }

    .summary-item {
        background: white;
        border: 1px solid #dbeafe;
        padding: 16px;
        border-radius: 18px;
        margin-bottom: 10px;
    }

    .summary-key {
        color: #64748b;
        font-size: 13px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .summary-value {
        color: #0f172a;
        font-size: 20px;
        font-weight: 900;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 0.75rem 1.2rem;
        font-weight: 800;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28);
        transition: 0.2s ease-in-out;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.35);
        color: white;
    }

    /* Images */
    img {
        border-radius: 22px;
    }

    /* Section titles */
    .section-title {
        font-size: 30px;
        font-weight: 900;
        color: #0f172a;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 20px;
    }

    .divider {
        height: 1px;
        background: #e5e7eb;
        margin: 24px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Paths
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# Stage 1 - CNN Phone Detection
STAGE1_MODEL_PATH = PROJECT_ROOT / "models" / "phone_detection" / "best_phone_detection.keras"
STAGE1_CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "phone_detection" / "class_names.json"

# Stage 1 - Random Forest Phone Detection
RF_MODEL_PATH = PROJECT_ROOT / "models" / "phone_detection_ml" / "random_forest_phone_detection.pkl"
RF_CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "phone_detection_ml" / "class_names.json"

# Stage 2 - iPhone Detection
STAGE2_MODEL_PATH = PROJECT_ROOT / "models" / "iphone_yolo_stage23" / "weights" / "best.pt"

# Stage 3 - Price Prediction
STAGE3_MODEL_PATH = PROJECT_ROOT / "models" / "price_prediction" / "best_price_model.pkl"
ENCODER_MODEL_PATH = PROJECT_ROOT / "models" / "price_prediction" / "label_encoder_model.pkl"
ENCODER_CONDITION_PATH = PROJECT_ROOT / "models" / "price_prediction" / "label_encoder_condition.pkl"

CNN_IMG_SIZE = (224, 224)
RF_IMG_SIZE = (64, 64)


# ============================================================
# Safe File Check
# ============================================================

def check_file(path, name):
    if not path.exists():
        st.error(f"Missing {name}: {path}")
        st.stop()


check_file(STAGE1_MODEL_PATH, "CNN phone detection model")
check_file(STAGE1_CLASS_NAMES_PATH, "CNN class names file")
check_file(STAGE2_MODEL_PATH, "YOLO model")
check_file(STAGE3_MODEL_PATH, "price prediction model")
check_file(ENCODER_MODEL_PATH, "model encoder")
check_file(ENCODER_CONDITION_PATH, "condition encoder")


# ============================================================
# Load Models
# ============================================================

@st.cache_resource
def load_stage1_cnn():
    return tf.keras.models.load_model(STAGE1_MODEL_PATH)


@st.cache_data
def load_cnn_class_names():
    with open(STAGE1_CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


@st.cache_resource
def load_random_forest():
    if RF_MODEL_PATH.exists():
        return joblib.load(RF_MODEL_PATH)
    return None


@st.cache_data
def load_rf_class_names():
    if RF_CLASS_NAMES_PATH.exists():
        with open(RF_CLASS_NAMES_PATH, "r") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_stage2_yolo():
    return YOLO(str(STAGE2_MODEL_PATH))


@st.cache_resource
def load_stage3_price_model():
    price_model = joblib.load(STAGE3_MODEL_PATH)
    le_model = joblib.load(ENCODER_MODEL_PATH)
    le_condition = joblib.load(ENCODER_CONDITION_PATH)
    return price_model, le_model, le_condition


stage1_cnn_model = load_stage1_cnn()
cnn_class_names = load_cnn_class_names()
rf_model = load_random_forest()
rf_class_names = load_rf_class_names()
stage2_model = load_stage2_yolo()
price_model, le_model, le_condition = load_stage3_price_model()


# ============================================================
# Helper Functions
# ============================================================

def get_label_from_class_names(class_names, index):
    if isinstance(class_names, dict):
        return class_names[str(index)]
    return class_names[index]


def beautify_label(label):
    return str(label).replace("_", " ").replace("-", " ").title()


def normalize_label(label):
    return str(label).lower().replace(" ", "_").replace("-", "_")


def confidence_percent(confidence):
    if confidence is None:
        return "Not available"
    return f"{confidence * 100:.2f}%"


def confidence_progress(confidence):
    if confidence is not None:
        st.progress(min(max(float(confidence), 0.0), 1.0))


def render_hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">📱 Smartphone AI System</div>
            <div class="hero-subtitle">
                A complete AI pipeline that detects smartphones, recognizes iPhone models,
                and estimates the phone price using Deep Learning, YOLOv8, Machine Learning,
                and Streamlit.
            </div>
            <div class="tag-row">
                <span class="tag">TensorFlow / Keras</span>
                <span class="tag">YOLOv8</span>
                <span class="tag">Random Forest</span>
                <span class="tag">Scikit-learn</span>
                <span class="tag">Streamlit Demo</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_pipeline_cards():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="small-card">
                <div class="stage-number">1</div>
                <div class="card-title">Phone Detection</div>
                <div class="card-text">
                    The system first checks if the uploaded image contains a phone,
                    no phone, or multiple phones.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="small-card">
                <div class="stage-number">2</div>
                <div class="card-title">iPhone Model Detection</div>
                <div class="card-text">
                    YOLOv8 detects and classifies the iPhone model from the image
                    using object detection.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="small-card">
                <div class="stage-number">3</div>
                <div class="card-title">Price Prediction</div>
                <div class="card-text">
                    A machine learning model estimates the phone price based on model,
                    storage, condition, and battery health.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def section_header(title, subtitle):
    st.markdown(
        f"""
        <div class="section-title">{title}</div>
        <div class="section-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True
    )


def result_card(title, value, subtitle="", status="good"):
    status_class = {
        "good": "result-card-good",
        "warning": "result-card-warning",
        "error": "result-card-error"
    }.get(status, "result-card-good")

    st.markdown(
        f"""
        <div class="result-card {status_class}">
            <div class="result-label">{title}</div>
            <div class="result-value">{value}</div>
            <div class="result-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def final_summary(phone_status, model_name, price, cnn_conf=None, yolo_conf=None):
    price_text = "Not predicted" if price is None else f"${price:.2f}"

    st.markdown(
        f"""
        <div class="final-box">
            <div class="final-title">🎯 Final AI Result</div>

            <div class="summary-item">
                <div class="summary-key">Phone Status</div>
                <div class="summary-value">{phone_status}</div>
            </div>

            <div class="summary-item">
                <div class="summary-key">Detected Model</div>
                <div class="summary-value">{model_name}</div>
            </div>

            <div class="summary-item">
                <div class="summary-key">Estimated Price</div>
                <div class="summary-value">{price_text}</div>
            </div>

            <div class="summary-item">
                <div class="summary-key">Model Confidence</div>
                <div class="summary-value">
                    CNN: {confidence_percent(cnn_conf)} | YOLO: {confidence_percent(yolo_conf)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_uploaded_image(img):
    st.markdown("### Uploaded Image")
    st.image(img, use_container_width=True)


# ============================================================
# Stage 1 - CNN Functions
# ============================================================

def preprocess_stage1_cnn(img):
    img = img.convert("RGB").resize(CNN_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return arr


def predict_stage1_cnn(img):
    preds = stage1_cnn_model.predict(preprocess_stage1_cnn(img), verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    label = get_label_from_class_names(cnn_class_names, pred_idx)
    return label, confidence


# ============================================================
# Stage 1 - Random Forest Functions
# ============================================================

def preprocess_stage1_rf(img):
    img = img.convert("RGB").resize(RF_IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.flatten().reshape(1, -1)
    return arr


def predict_stage1_rf(img):
    if rf_model is None:
        return None, None

    features = preprocess_stage1_rf(img)
    pred_idx = int(rf_model.predict(features)[0])

    confidence = None
    if hasattr(rf_model, "predict_proba"):
        probs = rf_model.predict_proba(features)[0]
        confidence = float(np.max(probs))

    names = rf_class_names if rf_class_names is not None else cnn_class_names
    label = get_label_from_class_names(names, pred_idx)

    return label, confidence


# ============================================================
# Stage 2 - YOLO Functions
# ============================================================

def predict_stage2(img):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        temp_path = tmp.name
        img.convert("RGB").save(temp_path)

    try:
        results = stage2_model.predict(
            source=temp_path,
            conf=0.25,
            imgsz=800,
            save=False,
            show=False,
            verbose=False
        )

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return None, None

        best_idx = int(boxes.conf.argmax().item())

        cls_id = int(boxes.cls[best_idx].item())
        confidence = float(boxes.conf[best_idx].item())

        detected_model = result.names[cls_id]

        return detected_model, confidence

    finally:
        Path(temp_path).unlink(missing_ok=True)


# ============================================================
# Stage 3 - Price Prediction
# ============================================================

YOLO_TO_PRICE_MODEL = {
    "iPhone-11": "iPhone-11",
    "iPhone-11-Pro": "iPhone-11-Pro",
    "iPhone-12": "iPhone-12",
    "iPhone-12-Pro": "iPhone-12-Pro",
    "iPhone-13": "iPhone-13",
    "iPhone-13-Pro": "iPhone-13-Pro",
    "iPhone-14": "iPhone-14",
    "iPhone-14-Pro": "iPhone-14-Pro",
    "iPhone-15": "iPhone 15",
    "iPhone 15 Pro": "iPhone 15 Pro",
    "iPhone-1st-gen-": "iPhone-1s",
    "iPhone-7": "iPhone-7",
    "iPhone-8": "iPhone-8",
    "iPhone-SE": "iPhone-SE",
    "iPhone-Xr": "iPhone-Xr",
    "iPhone-Xs": "iPhone-Xs",
}


def storage_options(model):
    if model in ["iPhone-1s", "iPhone 1s", "iPhone-1st-gen-", "iPhone 1st Gen"]:
        return [4, 8, 16]

    elif model in ["iPhone-7", "iPhone-8", "iPhone 7", "iPhone 8"]:
        return [32, 64, 128, 256]

    else:
        return [128, 256, 512, 1024]


def predict_price(model_name, storage_gb, battery_health):
    condition_value = "used"

    if model_name not in le_model.classes_:
        return None

    if condition_value not in le_condition.classes_:
        return None

    sample_df = pd.DataFrame([{
        "Model": model_name,
        "Storage_GB": storage_gb,
        "Battery_Health": battery_health,
        "Condition": condition_value
    }])

    sample_df["Model"] = le_model.transform(sample_df["Model"])
    sample_df["Condition"] = le_condition.transform(sample_df["Condition"])

    predicted_price = price_model.predict(sample_df)[0]

    return float(predicted_price)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 📱 AI Demo Menu")
    st.markdown("Choose what you want to test.")

    mode = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🚀 Full Pipeline Demo",
            "🔍 Stage 1 - Phone Detection",
            "📦 Stage 2 - iPhone Detection",
            "💰 Stage 3 - Price Prediction"
        ]
    )

    st.markdown("---")
    st.markdown("### Project Stack")
    st.markdown(
        """
        - Python  
        - Streamlit  
        - TensorFlow / Keras  
        - YOLOv8  
        - Scikit-learn  
        - Random Forest  
        """
    )

    st.markdown("---")
    st.caption("Designed for a clean project demo.")


# ============================================================
# Home Page
# ============================================================

if mode == "🏠 Home":
    render_hero()
    render_pipeline_cards()

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    section_header(
        "What this project does",
        "This system takes an image and passes it through a complete AI pipeline."
    )

    col1, col2 = st.columns([1.15, 0.85])

    with col1:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Demo Flow</div>
                <div class="card-text">
                    <b>Step 1:</b> Upload an image of a phone.<br><br>
                    <b>Step 2:</b> The CNN model checks whether the image contains a phone.<br><br>
                    <b>Step 3:</b> YOLOv8 detects the iPhone model.<br><br>
                    <b>Step 4:</b> The price model estimates the phone price using storage and battery health.<br><br>
                    For your LinkedIn video, use the <b>Full Pipeline Demo</b> page.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="card">
                <div class="card-title">Best for LinkedIn</div>
                <div class="card-text">
                    Record a short video showing:<br><br>
                    ✅ Upload image<br>
                    ✅ Phone detected<br>
                    ✅ iPhone model predicted<br>
                    ✅ Price estimated<br><br>
                    Keep it around <b>30–60 seconds</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# Full Pipeline Demo
# ============================================================

elif mode == "🚀 Full Pipeline Demo":
    render_hero()

    section_header(
        "Full Pipeline Demo",
        "Upload one phone image and let the system run through detection, model recognition, and price prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload an image for the full pipeline",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="full_pipeline_upload"
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        left, right = st.columns([0.9, 1.1])

        with left:
            display_uploaded_image(img)

        with right:
            section_header("Stage 1", "Phone detection using the CNN model.")

            cnn_label, cnn_conf = predict_stage1_cnn(img)
            rf_label, rf_conf = predict_stage1_rf(img)

            normalized_cnn_label = normalize_label(cnn_label)

            if normalized_cnn_label == "no_phone":
                result_card(
                    "Stage 1 Result",
                    "No Phone Detected",
                    f"CNN confidence: {confidence_percent(cnn_conf)}",
                    status="error"
                )
                confidence_progress(cnn_conf)

                final_summary(
                    phone_status="No phone detected",
                    model_name="Pipeline stopped",
                    price=None,
                    cnn_conf=cnn_conf,
                    yolo_conf=None
                )

            elif normalized_cnn_label == "multiple_phones":
                result_card(
                    "Stage 1 Result",
                    "Multiple Phones Detected",
                    f"CNN confidence: {confidence_percent(cnn_conf)}",
                    status="warning"
                )
                confidence_progress(cnn_conf)

                final_summary(
                    phone_status="Multiple phones detected",
                    model_name="Please upload one clear phone",
                    price=None,
                    cnn_conf=cnn_conf,
                    yolo_conf=None
                )

            else:
                result_card(
                    "Stage 1 Result",
                    "Phone Detected",
                    f"CNN confidence: {confidence_percent(cnn_conf)}",
                    status="good"
                )
                confidence_progress(cnn_conf)

                if rf_model is not None:
                    st.caption(
                        f"Random Forest baseline: {beautify_label(rf_label)}"
                        + (f" | Confidence: {confidence_percent(rf_conf)}" if rf_conf is not None else "")
                    )

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                section_header("Stage 2", "iPhone model recognition using YOLOv8.")

                detected_model, yolo_conf = predict_stage2(img)

                if detected_model is None:
                    result_card(
                        "Stage 2 Result",
                        "No Supported iPhone Model Detected",
                        "YOLO could not detect a model from the supported classes.",
                        status="error"
                    )

                    final_summary(
                        phone_status="Phone detected",
                        model_name="Not detected",
                        price=None,
                        cnn_conf=cnn_conf,
                        yolo_conf=None
                    )

                else:
                    yolo_status = "good" if yolo_conf >= 0.40 else "warning"

                    result_card(
                        "Stage 2 Result",
                        detected_model,
                        f"YOLO confidence: {confidence_percent(yolo_conf)}",
                        status=yolo_status
                    )
                    confidence_progress(yolo_conf)

                    model_for_price = YOLO_TO_PRICE_MODEL.get(detected_model)

                    if model_for_price is not None and model_for_price in le_model.classes_:
                        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                        section_header("Stage 3", "Choose phone details and estimate the price.")

                        col_storage, col_battery = st.columns(2)

                        with col_storage:
                            storage = st.selectbox(
                                "Storage",
                                storage_options(model_for_price),
                                key="full_storage"
                            )

                        with col_battery:
                            battery = st.slider(
                                "Battery Health (%)",
                                min_value=70,
                                max_value=100,
                                value=90,
                                key="full_battery"
                            )

                        if st.button("Predict Final Price", key="full_predict_btn"):
                            price = predict_price(model_for_price, storage, battery)

                            if price is not None:
                                result_card(
                                    "Stage 3 Result",
                                    f"${price:.2f}",
                                    f"Model: {model_for_price} | Storage: {storage}GB | Battery: {battery}%",
                                    status="good"
                                )

                                final_summary(
                                    phone_status="Phone detected",
                                    model_name=detected_model,
                                    price=price,
                                    cnn_conf=cnn_conf,
                                    yolo_conf=yolo_conf
                                )
                            else:
                                result_card(
                                    "Stage 3 Result",
                                    "Price Prediction Failed",
                                    "The selected phone is not available in the price model.",
                                    status="error"
                                )
                    else:
                        result_card(
                            "Stage 3 Result",
                            "Price Not Available",
                            "This detected iPhone model is not available in the price prediction model.",
                            status="warning"
                        )


# ============================================================
# Stage 1 Interface
# ============================================================

elif mode == "🔍 Stage 1 - Phone Detection":
    render_hero()

    section_header(
        "Stage 1: Phone Detection",
        "Compare the final CNN model with the Random Forest machine learning baseline."
    )

    uploaded_file = st.file_uploader(
        "Upload image for Stage 1",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="stage1_upload"
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        left, right = st.columns([0.9, 1.1])

        with left:
            display_uploaded_image(img)

        with right:
            cnn_label, cnn_conf = predict_stage1_cnn(img)
            rf_label, rf_conf = predict_stage1_rf(img)

            result_card(
                "CNN Deep Learning Result",
                beautify_label(cnn_label),
                f"Confidence: {confidence_percent(cnn_conf)}",
                status="good" if cnn_conf >= 0.60 else "warning"
            )
            confidence_progress(cnn_conf)

            if cnn_conf < 0.60:
                st.warning("CNN confidence is low. The image may be unclear or different from the training data.")

            if rf_model is None:
                result_card(
                    "Random Forest Baseline",
                    "Model Not Found",
                    "Expected path: models/phone_detection_ml/random_forest_phone_detection.pkl",
                    status="warning"
                )
            else:
                result_card(
                    "Random Forest Baseline",
                    beautify_label(rf_label),
                    f"Confidence: {confidence_percent(rf_conf)}",
                    status="good"
                )
                confidence_progress(rf_conf)

            result_card(
                "Final Decision",
                beautify_label(cnn_label),
                "The final Stage 1 decision uses CNN. Random Forest is shown only for comparison.",
                status="good"
            )


# ============================================================
# Stage 2 Interface
# ============================================================

elif mode == "📦 Stage 2 - iPhone Detection":
    render_hero()

    section_header(
        "Stage 2: iPhone Model Detection",
        "Upload an image and YOLOv8 will detect the supported iPhone model."
    )

    uploaded_file = st.file_uploader(
        "Upload image for Stage 2",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="stage2_upload"
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        left, right = st.columns([0.9, 1.1])

        with left:
            display_uploaded_image(img)

        with right:
            detected_model, yolo_conf = predict_stage2(img)

            if detected_model is None:
                result_card(
                    "YOLO Result",
                    "No iPhone Detected",
                    "The model did not detect a supported iPhone class.",
                    status="error"
                )
            else:
                result_card(
                    "YOLO Result",
                    detected_model,
                    f"Confidence: {confidence_percent(yolo_conf)}",
                    status="good" if yolo_conf >= 0.40 else "warning"
                )
                confidence_progress(yolo_conf)

                model_for_price = YOLO_TO_PRICE_MODEL.get(detected_model)

                if model_for_price is not None and model_for_price in le_model.classes_:
                    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                    section_header(
                        "Optional Price Prediction",
                        "Use the detected model and select the phone details."
                    )

                    col_storage, col_battery = st.columns(2)

                    with col_storage:
                        storage = st.selectbox(
                            "Storage",
                            storage_options(model_for_price),
                            key="stage2_storage"
                        )

                    with col_battery:
                        battery = st.slider(
                            "Battery Health (%)",
                            min_value=70,
                            max_value=100,
                            value=90,
                            key="stage2_battery"
                        )

                    if st.button("Predict Price", key="stage2_predict_btn"):
                        price = predict_price(model_for_price, storage, battery)

                        if price is not None:
                            result_card(
                                "Estimated Price",
                                f"${price:.2f}",
                                f"Model: {model_for_price} | Storage: {storage}GB | Battery: {battery}%",
                                status="good"
                            )
                        else:
                            result_card(
                                "Estimated Price",
                                "Prediction Failed",
                                "The price model could not generate a result.",
                                status="error"
                            )
                else:
                    result_card(
                        "Price Model",
                        "Not Available",
                        "This detected iPhone model is not available in the price prediction model.",
                        status="warning"
                    )


# ============================================================
# Stage 3 Interface
# ============================================================

elif mode == "💰 Stage 3 - Price Prediction":
    render_hero()

    section_header(
        "Stage 3: Price Prediction",
        "Select the phone details and estimate the expected used phone price."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        model_name = st.selectbox(
            "iPhone Model",
            list(le_model.classes_),
            key="stage3_model"
        )

    with col2:
        storage = st.selectbox(
            "Storage",
            storage_options(model_name),
            key="stage3_storage"
        )

    with col3:
        battery = st.slider(
            "Battery Health (%)",
            min_value=70,
            max_value=100,
            value=90,
            key="stage3_battery"
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if st.button("Predict Price", key="stage3_predict_btn"):
        price = predict_price(model_name, storage, battery)

        if price is not None:
            result_card(
                "Estimated Price",
                f"${price:.2f}",
                f"Model: {model_name} | Storage: {storage}GB | Battery: {battery}%",
                status="good"
            )

            final_summary(
                phone_status="Manual price prediction",
                model_name=model_name,
                price=price,
                cnn_conf=None,
                yolo_conf=None
            )
        else:
            result_card(
                "Estimated Price",
                "Prediction Failed",
                "The selected model is not available in the saved price model.",
                status="error"
            )