# 🎥 دوربین زنده — تشخیص علائم راهنمایی و رانندگی
# Live camera page for the traffic-sign app (multipage: put this file in
# pages/1_camera_live.py next to streamlit_app.py).
#
# Single mode: "عکس با دوربین" — live preview, tap "عکس بگیر", YOLO finds
# sign-like regions, the ResNet18 classifier labels EACH one
# (all 43 classes), boxes are numbered and the Persian names are
# listed below. There is NO live-video mode in this version.
#
# Camera switch: both modes use streamlit-webrtc with a facingMode
# constraint, so the user can pick front (جلو / selfie) or back
# (عقب / environment) camera. st.camera_input cannot do this — it
# always opens the default camera with no switch control.
#
# How it works: the ResNet18 model is a CLASSIFIER (it labels one cropped
# image). Finding WHERE the signs are in a photo is a different job —
# object DETECTION — done here by YOLOv8n. YOLO (COCO weights) only knows
# "stop sign" and "traffic light" as sign-like objects, so live detection
# works for stop signs; every box it finds is then classified by ResNet18
# for the full 43-class Persian label. Detecting ALL 43 sign types live
# needs YOLO fine-tuned on traffic-sign detection data (GTSDB) — that is
# the natural next step after this page works.
#
# Drawing is done with PIL, not cv2: Streamlit Cloud has no libGL, so
# plain opencv-python cannot import there. requirements.txt must include
# opencv-python-headless (after ultralytics) so ultralytics' own cv2
# import works headless.
#
# New dependencies (add to requirements.txt):
#   ultralytics
#   opencv-python-headless
#   streamlit-webrtc

from pathlib import Path

import av
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from torchvision import transforms
from ultralytics import YOLO

from traffic_ai.models import build_model

st.set_page_config(
    page_title="دوربین زنده — تشخیص علائم",
    page_icon="🎥",
    layout="centered",
)

st.title("🎥 دوربین زنده — تشخیص علائم")
st.caption(
    "دوربین رو جلوی تابلو بگیر؛ مدل دور هر تابلو خط می‌کشه و می‌گه چیه."
)

# =========================================================
# تنظیمات
# =========================================================

MODEL_PATH = Path("runs/gtsrb_resnet18/best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# YOLO (COCO) class ids that look like traffic signs.
YOLO_SIGN_CLASSES = {
    9: "Traffic light",
    11: "Stop",
}

FA_NAMES = {
    "Speed limit (20 km/h)": "محدودیت سرعت ۲۰ کیلومتر بر ساعت",
    "Speed limit (30 km/h)": "محدودیت سرعت ۳۰ کیلومتر بر ساعت",
    "Speed limit (50 km/h)": "محدودیت سرعت ۵۰ کیلومتر بر ساعت",
    "Speed limit (60 km/h)": "محدودیت سرعت ۶۰ کیلومتر بر ساعت",
    "Speed limit (70 km/h)": "محدودیت سرعت ۷۰ کیلومتر بر ساعت",
    "Speed limit (80 km/h)": "محدودیت سرعت ۸۰ کیلومتر بر ساعت",
    "End speed limit (80 km/h)": "پایان محدودیت سرعت ۸۰ کیلومتر بر ساعت",
    "Speed limit (100 km/h)": "محدودیت سرعت ۱۰۰ کیلومتر بر ساعت",
    "Speed limit (120 km/h)": "محدودیت سرعت ۱۲۰ کیلومتر بر ساعت",
    "No passing": "سبقت ممنوع",
    "No passing for vehicles over 3.5 tons": "سبقت ممنوع برای وسایل نقلیه بالای ۳٫۵ تن",
    "Right-of-way at the next intersection": "حق تقدم در تقاطع بعدی",
    "Right-of-way at next intersection": "حق تقدم در تقاطع بعدی",
    "No passing for vehicles over 3.5 t": "سبقت ممنوع برای وسایل نقلیه بالای ۳٫۵ تن",
    "Dangerous curve right": "پیچ خطرناک به راست",
    "Priority road": "جاده اصلی",
    "Yield": "رعایت حق تقدم",
    "Stop": "ایست",
    "No vehicles": "ورود وسایل نقلیه ممنوع",
    "Vehicles over 3.5 tons prohibited": "ورود وسایل نقلیه بالای ۳٫۵ تن ممنوع",
    "No entry": "ورود ممنوع",
    "General caution": "احتیاط",
    "Dangerous curve to the left": "پیچ خطرناک به چپ",
    "Dangerous curve to the right": "پیچ خطرناک به راست",
    "Double curve": "پیچ‌های متوالی",
    "Bumpy road": "جاده ناهموار",
    "Slippery road": "جاده لغزنده",
    "Road narrows on the right": "جاده از سمت راست باریک می‌شود",
    "Road work": "عملیات عمرانی جاده",
    "Traffic signals": "چراغ راهنمایی",
    "Pedestrians": "عبور عابر پیاده",
    "Children crossing": "عبور کودکان",
    "Bicycles crossing": "عبور دوچرخه",
    "Beware of ice/snow": "احتیاط، یخ و برف",
    "Wild animals crossing": "عبور حیوانات وحشی",
    "End of all speed and passing limits": "پایان تمام محدودیت‌های سرعت و سبقت",
    "Turn right ahead": "پیچیدن به راست در جلو",
    "Turn left ahead": "پیچیدن به چپ در جلو",
    "Ahead only": "فقط مستقیم",
    "Go straight or right": "مستقیم یا راست",
    "Go straight or left": "مستقیم یا چپ",
    "Keep right": "از راست حرکت کنید",
    "Keep left": "از چپ حرکت کنید",
    "Roundabout mandatory": "میدان اجباری",
    "End of no passing": "پایان ممنوعیت سبقت",
    "End of no passing by vehicles over 3.5 tons": "پایان ممنوعیت سبقت برای وسایل نقلیه بالای ۳٫۵ تن",
}


# =========================================================
# بارگذاری مدل‌ها
# =========================================================

@st.cache_resource
def load_classifier():
    """The same ResNet18 classifier as streamlit_app.py."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"مدل پیدا نشد:\n{MODEL_PATH.resolve()}")

    checkpoint = torch.load(
        MODEL_PATH, map_location=DEVICE, weights_only=False
    )
    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    image_size = int(checkpoint.get("image_size", 128))

    model = build_model(
        model_name, len(class_names),
        pretrained=False, freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    mean = checkpoint.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("imagenet_std", [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, class_names, transform


@st.cache_resource
def load_detector():
    """YOLOv8n (COCO weights, ~6 MB, downloaded on first run)."""
    return YOLO("yolov8n.pt")


@torch.inference_mode()
def classify_crop(model, crop, transform, class_names):
    tensor = transform(crop).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1)
    confidence, index = probs.max(dim=1)
    index = int(index.item())
    english = class_names[index]
    return english, FA_NAMES.get(english, "ترجمه موجود نیست"), float(confidence.item())


def detect_sign_boxes(detector, image_np, conf_threshold=0.5):
    """Run YOLO and return boxes for sign-like objects.

    Returns a list of (x1, y1, x2, y2, yolo_label, confidence).
    """
    results = detector(image_np, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        class_id = int(box.cls[0].item())
        if class_id not in YOLO_SIGN_CLASSES:
            continue
        confidence = float(box.conf[0].item())
        if confidence < conf_threshold:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        boxes.append((x1, y1, x2, y2, YOLO_SIGN_CLASSES[class_id], confidence))
    return boxes


def process_photo(image, detector, classifier, class_names, transform):
    """Detect all signs in a photo, classify each, show numbered boxes."""
    image_np = np.array(image)

    with st.spinner("در حال تشخیص..."):
        boxes = detect_sign_boxes(detector, image_np)

    if not boxes:
        st.warning(
            "تابلوای پیدا نشد — دوربین رو نزدیک‌تر بگیر یا مطمئن شو "
            "تابلو توی کادره."
        )
        st.image(image, caption="تصویر ورودی", width="stretch")
        return

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for number, (x1, y1, x2, y2, _label, _conf) in enumerate(boxes, start=1):
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
        draw.ellipse(
            [x1 - 14, y1 - 14, x1 + 14, y1 + 14],
            fill=(255, 0, 0),
        )
        draw.text((x1, y1), str(number), fill=(255, 255, 255))

    st.image(annotated, caption="تابلوهای پیدا شده", width="stretch")

    st.divider()
    st.subheader("🎯 نتیجه‌ها")
    for number, (x1, y1, x2, y2, _label, _conf) in enumerate(boxes, start=1):
        crop = image.crop((x1, y1, x2, y2))
        english, persian, confidence = classify_crop(
            classifier, crop, transform, class_names
        )
        st.write(f"**{number}. {english}**")
        st.caption(f"{persian} — {confidence * 100:.2f}%")


# =========================================================
# انتخاب دوربین و حالت
# =========================================================

try:
    classifier, class_names, transform = load_classifier()
    detector = load_detector()
except Exception as error:
    st.error("❌ خطا در بارگذاری مدل")
    st.code(str(error))
    st.stop()

st.write(
    "پیش‌نمایش دوربین رو ببین، بعد «عکس بگیر» رو بزن؛ مدل دور هر "
    "تابلو خط می‌کشه و زیر عکس می‌نویسه هر کدوم چیه."
)


class PhotoCapture(VideoProcessorBase):
    def __init__(self):
        self.last_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="rgb24")
        self.last_frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="rgb24")


ctx = webrtc_streamer(
    key="photo-capture",
    video_processor_factory=PhotoCapture,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints=video_constraints,
)

if st.button("📸 عکس بگیر"):
    if ctx.video_processor is None or ctx.video_processor.last_frame is None:
        st.warning(
            "هنوز تصویری از دوربین نرسیده — چند ثانیه صبر کن تا "
            "پیش‌نمایش بیاد، بعد دوباره بزن."
        )
    else:
        image = Image.fromarray(ctx.video_processor.last_frame)
        process_photo(image, detector, classifier, class_names, transform)
