from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

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
    "مدل: ResNet18 — GTSRB"
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
    # Prediction
    # =====================================================

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