from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from ultralytics import YOLO

from traffic_ai.models import build_model


# =========================================================
# تنظیمات
# =========================================================

MODEL_PATH = Path(
    "runs/gtsrb_resnet18/best_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# ترجمه کلاس‌ها
# =========================================================

FA_NAMES = {
    "Speed limit (20 km/h)": "محدودیت سرعت ۲۰ کیلومتر بر ساعت",
    "Speed limit (30 km/h)": "محدودیت سرعت ۳۰ کیلومتر بر ساعت",
    "Speed limit (50 km/h)": "محدودیت سرعت ۵۰ کیلومتر بر ساعت",
    "Speed limit (60 km/h)": "محدودیت سرعت ۶۰ کیلومتر بر ساعت",
    "Speed limit (70 km/h)": "محدودیت سرعت ۷۰ کیلومتر بر ساعت",
    "Speed limit (80 km/h)": "محدودیت سرعت ۸۰ کیلومتر بر ساعت",

    "End speed limit (80 km/h)":
        "پایان محدودیت سرعت ۸۰ کیلومتر بر ساعت",

    "Speed limit (100 km/h)": "محدودیت سرعت ۱۰۰ کیلومتر بر ساعت",
    "Speed limit (120 km/h)": "محدودیت سرعت ۱۲۰ کیلومتر بر ساعت",

    "No passing": "سبقت ممنوع",

    "No passing for vehicles over 3.5 tons":
        "سبقت ممنوع برای وسایل نقلیه بالای ۳٫۵ تن",

    "Right-of-way at the next intersection":
        "حق تقدم در تقاطع بعدی",

    "Right-of-way at next intersection":
        "حق تقدم در تقاطع بعدی",

    "No passing for vehicles over 3.5 t":
        "سبقت ممنوع برای وسایل نقلیه بالای ۳٫۵ تن",

    "Dangerous curve right":
        "پیچ خطرناک به راست",

    "Priority road": "جاده اصلی",
    "Yield": "رعایت حق تقدم",
    "Stop": "ایست",
    "No vehicles": "ورود وسایل نقلیه ممنوع",

    "Vehicles over 3.5 tons prohibited":
        "ورود وسایل نقلیه بالای ۳٫۵ تن ممنوع",

    "No entry": "ورود ممنوع",
    "General caution": "احتیاط",

    "Dangerous curve to the left":
        "پیچ خطرناک به چپ",

    "Dangerous curve to the right":
        "پیچ خطرناک به راست",

    "Double curve": "پیچ‌های متوالی",
    "Bumpy road": "جاده ناهموار",
    "Slippery road": "جاده لغزنده",

    "Road narrows on the right":
        "جاده از سمت راست باریک می‌شود",

    "Road work": "عملیات عمرانی جاده",
    "Traffic signals": "چراغ راهنمایی",

    "Pedestrians": "عبور عابر پیاده",
    "Children crossing": "عبور کودکان",
    "Bicycles crossing": "عبور دوچرخه",

    "Beware of ice/snow":
        "احتیاط، یخ و برف",

    "Wild animals crossing":
        "عبور حیوانات وحشی",

    "End of all speed and passing limits":
        "پایان تمام محدودیت‌های سرعت و سبقت",

    "Turn right ahead":
        "پیچیدن به راست در جلو",

    "Turn left ahead":
        "پیچیدن به چپ در جلو",

    "Ahead only":
        "فقط مستقیم",

    "Go straight or right":
        "مستقیم یا راست",

    "Go straight or left":
        "مستقیم یا چپ",

    "Keep right":
        "از راست حرکت کنید",

    "Keep left":
        "از چپ حرکت کنید",

    "Roundabout mandatory":
        "میدان اجباری",

    "End of no passing":
        "پایان ممنوعیت سبقت",

    "End of no passing by vehicles over 3.5 tons":
        "پایان ممنوعیت سبقت برای وسایل نقلیه بالای ۳٫۵ تن",
}


# =========================================================
# بارگذاری مدل
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"مدل پیدا نشد:\n{MODEL_PATH.resolve()}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]

    image_size = int(
        checkpoint.get("image_size", 128)
    )

    model = build_model(
        model_name,
        len(class_names),
        pretrained=False,
        freeze_backbone=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    mean = checkpoint.get(
        "imagenet_mean",
        [0.485, 0.456, 0.406],
    )

    std = checkpoint.get(
        "imagenet_std",
        [0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
    ])

    return (
        model,
        class_names,
        transform,
        model_name,
    )


# =========================================================
# تشخیص جای تابلوها با YOLO
# =========================================================
# ResNet18 فقط «طبقه‌بندی» می‌کنه (یه عکس برش‌خورده چیه؟).
# پیدا کردن جای تابلوها توی عکس، کار «تشخیص شیء» است و
# با YOLOv8n انجام می‌شه. YOLO آماده (COCO) فقط «ایست» و
# «چراغ راهنمایی» رو به‌عنوان تابلو می‌شناسه؛ برای اینکه
# هر ۴۳ کلاس رو توی عکس پیدا کنه، قدم بعدی فاین‌تیون YOLO
# روی دیتاست تشخیص علائم (GTSDB) است.

# YOLO (COCO) class ids that look like traffic signs.
YOLO_SIGN_CLASSES = {
    9: "Traffic light",
    11: "Stop",
}


@st.cache_resource
def load_detector():
    """YOLOv8n (COCO weights, ~6 MB, downloaded on first run)."""
    return YOLO("yolov8n.pt")


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


# =========================================================
# پیش‌بینی
# =========================================================

@torch.inference_mode()
def predict(
    model,
    image,
    transform,
    class_names,
):

    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    logits = model(tensor)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    confidence, index = probabilities.max(
        dim=1
    )

    index = int(index.item())
    confidence = float(confidence.item())

    # اسم انگلیسی مستقیماً از checkpoint
    english_name = class_names[index]

    # فارسی از همان اسم انگلیسی
    persian_name = FA_NAMES.get(
        english_name,
        "ترجمه موجود نیست",
    )

    return (
        english_name,
        persian_name,
        confidence,
        probabilities[0].cpu(),
    )


# =========================================================
# رابط کاربری
# =========================================================

st.set_page_config(
    page_title="GTSRB ResNet18",
    page_icon="🚦",
    layout="centered",
)

st.title(
    "🚦 تشخیص علائم راهنمایی و رانندگی"
)

st.caption(
    "مدل: ResNet18 — GTSRB | اگه چند تا تابلو توی عکس باشه، همه رو پیدا می‌کنه"
)


# =========================================================
# Load
# =========================================================

try:

    (
        model,
        class_names,
        transform,
        model_name,
    ) = load_model()

except Exception as error:

    st.error(
        "❌ خطا در بارگذاری مدل"
    )

    st.code(
        str(error)
    )

    st.stop()

# YOLO جدا لود می‌شه؛ اگه لود نشد، اپ با طبقه‌بندی کل عکس
# (مثل قبل) کار می‌کنه.
try:
    detector = load_detector()
except Exception:
    detector = None


# =========================================================
# Upload
# =========================================================

uploaded_file = st.file_uploader(
    "📷 تصویر علامت را انتخاب کنید",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
)


def show_single_prediction(image):
    """Classify one (cropped) sign image and show result + top 3."""

    with st.spinner(
        "در حال تشخیص..."
    ):

        (
            english_name,
            persian_name,
            confidence,
            probabilities,
        ) = predict(
            model,
            image,
            transform,
            class_names,
        )

    st.divider()

    st.subheader(
        "🎯 نتیجه"
    )

    # نام انگلیسی دقیق کلاس
    st.markdown(
        f"### 🇬🇧 {english_name}"
    )

    # ترجمه فارسی همان کلاس
    st.markdown(
        f"### 🇮🇷 {persian_name}"
    )

    st.metric(
        "اطمینان مدل",
        f"{confidence * 100:.2f}%",
    )

    # =====================================================
    # Top 3
    # =====================================================

    st.divider()

    st.subheader(
        "🔎 سه پیش‌بینی برتر"
    )

    top_k = min(
        3,
        len(class_names)
    )

    values, indices = torch.topk(
        probabilities,
        k=top_k,
    )

    for rank, (
        value,
        index,
    ) in enumerate(
        zip(
            values.tolist(),
            indices.tolist(),
        ),
        start=1,
    ):

        english = class_names[index]

        persian = FA_NAMES.get(
            english,
            "ترجمه موجود نیست",
        )

        st.write(
            f"**{rank}. {english}**"
        )

        st.caption(
            f"{persian} — "
            f"{value * 100:.2f}%"
        )


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="تصویر ورودی",
        width="stretch",
    )

    # =====================================================
    # چند تابلو؟ — اول با YOLO جای تابلوها رو پیدا کن،
    # بعد هر کدوم رو با ResNet18 طبقه‌بندی کن.
    # =====================================================

    boxes = []

    if detector is not None:
        with st.spinner(
            "در حال پیدا کردن تابلوها..."
        ):
            boxes = detect_sign_boxes(
                detector,
                np.array(image),
            )

    if boxes:
        # دور هر تابلو یه کادر شماره‌دار بکش
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        for number, (
            x1,
            y1,
            x2,
            y2,
            _label,
            _conf,
        ) in enumerate(boxes, start=1):

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(255, 0, 0),
                width=4,
            )

            draw.ellipse(
                [x1 - 14, y1 - 14, x1 + 14, y1 + 14],
                fill=(255, 0, 0),
            )

            draw.text(
                (x1, y1),
                str(number),
                fill=(255, 255, 255),
            )

        st.image(
            annotated,
            caption=f"تابلوهای پیدا شده: {len(boxes)}",
            width="stretch",
        )

        if len(boxes) == 1:
            # یه تابلو — مثل قبل: نتیجه + سه پیش‌بینی برتر
            (x1, y1, x2, y2, _label, _conf) = boxes[0]
            crop = image.crop((x1, y1, x2, y2))
            show_single_prediction(crop)
        else:
            # چند تابلو — هر کدوم چیه؟
            st.divider()

            st.subheader(
                f"🎯 نتیجه‌ها — {len(boxes)} تابلو"
            )

            for number, (
                x1,
                y1,
                x2,
                y2,
                _label,
                _conf,
            ) in enumerate(boxes, start=1):

                crop = image.crop((x1, y1, x2, y2))

                (
                    english_name,
                    persian_name,
                    confidence,
                    _probabilities,
                ) = predict(
                    model,
                    crop,
                    transform,
                    class_names,
                )

                st.write(
                    f"**{number}. {english_name}**"
                )

                st.caption(
                    f"{persian_name} — "
                    f"{confidence * 100:.2f}%"
                )

    else:
        # هیچ کادری پیدا نشد — کل عکس رو مثل قبل طبقه‌بندی کن
        st.caption(
            "کادر تابلو پیدا نشد؛ کل عکس طبقه‌بندی شد."
        )

        show_single_prediction(image)