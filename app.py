import os
import time
import cv2
import gdown
import torch
import functools
import numpy as np
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

DEHAZE_MODEL_PATH = "remove_hazy_model_256x256.pth"
DEHAZE_GDRIVE_ID  = "1ji3x-KO19X2yGpT7oaUIpJ5DiCgQg8xS"
YOLO_MODEL_NAME   = "yolov8n.pt"

DRIVING_CLASSES = {
    "person", "bicycle", "car", "motorcycle",
    "bus", "truck", "traffic light", "stop sign",
}

st.set_page_config(
    page_title="NEXTGEN VISION AI",
    page_icon="👁️",
    layout="wide",
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.stApp { background: #050a12; color: #d8ecff; }
.block-container { padding-top: 1.4rem; }
h1, h2, h3 { color: #00d4ff !important; }
[data-testid="stSidebar"] { background: #07111f; }
.metric-card {
    background: #0d1e35; border: 1px solid #1a3a5c;
    border-top: 2px solid #00d4ff; padding: 1rem;
    border-radius: 8px; text-align: center;
}
.metric-val { color: #00ff88; font-size: 1.15rem; font-weight: 800; }
.metric-lbl { color: #7fa1bd; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
.small-note { color: #7fa1bd; font-size: 0.85rem; line-height: 1.6; }
.warning-box {
    background: #21170a; border: 1px solid #ffb800;
    color: #ffdf8a; padding: 0.9rem 1rem;
    border-radius: 8px; margin-bottom: 1rem;
}
.info-box {
    background: #071b2c; border: 1px solid #1a3a5c;
    color: #c8e8ff; padding: 0.9rem 1rem;
    border-radius: 8px; margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_dehaze_model_if_needed():
    if os.path.exists(DEHAZE_MODEL_PATH):
        return True, None
    try:
        url = f"https://drive.google.com/uc?id={DEHAZE_GDRIVE_ID}"
        gdown.download(url, DEHAZE_MODEL_PATH, quiet=False)
        if os.path.exists(DEHAZE_MODEL_PATH):
            return True, None
        return False, "Download finished but file not found."
    except Exception as e:
        return False, str(e)


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3):
        super().__init__()
        self.r = r
        self.eps = eps
        self.boxfilter = nn.AvgPool2d(kernel_size=2*r+1, stride=1, padding=r)

    def forward(self, I, p):
        N        = self.boxfilter(torch.ones(p.size(), device=p.device, dtype=p.dtype))
        mean_I   = self.boxfilter(I) / N
        mean_p   = self.boxfilter(p) / N
        mean_Ip  = self.boxfilter(I * p) / N
        cov_Ip   = mean_Ip - mean_I * mean_p
        mean_II  = self.boxfilter(I * I) / N
        var_I    = mean_II - mean_I * mean_I
        a        = cov_Ip / (var_I + self.eps)
        b        = mean_p - a * mean_I
        mean_a   = self.boxfilter(a) / N
        mean_b   = self.boxfilter(b) / N
        return mean_a * I + mean_b


class DCPDehazeGenerator(nn.Module):
    def __init__(self, win_size=15, r=40, eps=1e-3):
        super().__init__()
        self.guided_filter    = GuidedFilter(r=r, eps=eps)
        self.neighborhood_size = win_size
        self.omega             = 0.95

    def get_dark_channel(self, img, w):
        if len(img.shape) != 4:
            raise NotImplementedError
        img, _ = torch.min(img, dim=1)
        img     = torch.unsqueeze(img, dim=1)
        pad_size = int(np.floor(w / 2))
        pads = [pad_size, pad_size-1, pad_size, pad_size-1] if w % 2 == 0 else [pad_size]*4
        img_min  = F.pad(img, pads, mode="replicate")
        return -F.max_pool2d(-img_min, kernel_size=w, stride=1)

    def atmospheric_light(self, img, dark_img):
        num, chl, height, width = img.shape
        top_num = max(int(0.001 * height * width), 1)
        A = torch.zeros(num, chl, 1, 1, device=img.device, dtype=img.dtype)
        for n in range(num):
            cur_dark = dark_img[n, 0].reshape(height * width)
            _, indices = cur_dark.sort(descending=True)
            for c in range(chl):
                A[n, c, 0, 0] = img[n, c].reshape(height * width)[indices[:top_num]].mean()
        return A

    def forward(self, x):
        if x.shape[1] > 1:
            guidance = 0.2989*x[:,0] + 0.5870*x[:,1] + 0.1140*x[:,2]
        else:
            guidance = x[:,0]
        guidance   = (guidance + 1) / 2
        guidance   = torch.unsqueeze(guidance, 1)
        img_patch  = (x + 1) / 2
        num, chl, h, w = img_patch.shape
        dark_img   = self.get_dark_channel(img_patch, self.neighborhood_size)
        A          = self.atmospheric_light(img_patch, dark_img)
        map_A      = A.repeat(1, 1, h, w).clamp(min=1e-6)
        trans_raw  = (1 - self.omega * self.get_dark_channel(img_patch / map_A, self.neighborhood_size)).clamp(0.05, 1.0)
        T_DCP      = self.guided_filter(guidance, trans_raw).clamp(0.05, 1.0)
        J_DCP      = (img_patch - map_A) / T_DCP.repeat(1, 3, 1, 1) + map_A
        return J_DCP.clamp(0, 1)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self._build(dim, padding_type, norm_layer, use_dropout, use_bias)

    def _build(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        block = []
        for _ in range(2):
            if padding_type == "reflect":
                block += [nn.ReflectionPad2d(1)]; p = 0
            elif padding_type == "replicate":
                block += [nn.ReplicationPad2d(1)]; p = 0
            else:
                p = 1
            block += [nn.Conv2d(dim, dim, 3, 1, p, bias=use_bias), norm_layer(dim)]
            if _ == 0:
                block += [nn.ReLU(True)]
                if use_dropout:
                    block += [nn.Dropout(0.5)]
        return nn.Sequential(*block)

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=9, padding_type="reflect"):
        super().__init__()
        use_bias = (norm_layer == nn.InstanceNorm2d or
                    (hasattr(norm_layer, 'func') and norm_layer.func == nn.InstanceNorm2d))
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7, padding=0, bias=use_bias),
            norm_layer(ngf), nn.ReLU(True),
        ]
        for i in range(2):
            m = 2**i
            model += [nn.Conv2d(ngf*m, ngf*m*2, 3, 2, 1, bias=use_bias),
                      norm_layer(ngf*m*2), nn.ReLU(True)]
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf*4, padding_type, norm_layer, use_dropout, use_bias)]
        for i in range(2):
            m = 2**(2-i)
            model += [nn.ConvTranspose2d(ngf*m, ngf*m//2, 3, 2, 1, output_padding=1, bias=use_bias),
                      norm_layer(ngf*m//2), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, 7, padding=0, bias=use_bias),
                  nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return torch.clamp(self.model(x), -1, 1)


# ============================================================
# MODEL LOADING (cached — loads once for the whole session)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_dehaze_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcp    = DCPDehazeGenerator().to(device).eval()
    resnet = ResnetGenerator(3, 3, norm_layer=nn.InstanceNorm2d).to(device)
    ckpt   = torch.load(DEHAZE_MODEL_PATH, map_location=device)
    if isinstance(ckpt, dict):
        key = next((k for k in ["params","state_dict","model","net_g","generator"] if k in ckpt), None)
        state_dict = ckpt[key] if key else ckpt
    else:
        state_dict = ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = resnet.load_state_dict(state_dict, strict=False)
    resnet.eval()
    return dcp, resnet, device, missing, unexpected


@st.cache_resource(show_spinner=False)
def load_yolo_model():
    if not YOLO_AVAILABLE:
        return None
    return YOLO(YOLO_MODEL_NAME)


# ============================================================
# INFERENCE HELPERS
# ============================================================

def bgr_to_tensor(img_bgr, size=192):
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t   = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return t * 2.0 - 1.0


def tensor_to_bgr(tensor, original_hw):
    out = tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    out = cv2.cvtColor((out * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(out, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_LINEAR)


def dehaze_frame(img_bgr, dcp, resnet, device, strength=1.0, dcp_only=False, size=192):
    h, w = img_bgr.shape[:2]
    x    = bgr_to_tensor(img_bgr, size).to(device)
    with torch.no_grad():
        dcp_out = dcp(x)
        if dcp_only:
            refined = dcp_out
        else:
            refined = (resnet(dcp_out) + 1.0) / 2.0
        result = tensor_to_bgr(refined, (h, w))
    if strength < 1.0:
        result = cv2.addWeighted(img_bgr, 1.0 - strength, result, strength, 0)
    return result


def visibility_score(img_bgr):
    gray      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    contrast  = gray.std()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    score     = min(100, max(0, int(contrast * 1.4 + sharpness**0.5 * 2.0)))
    return score, round(float(contrast), 2), round(float(sharpness), 2)


def detect_objects(img_bgr, yolo, conf=0.35, driving_only=True, ar_style=True):
    results  = yolo(img_bgr, conf=conf, verbose=False)
    result   = results[0]
    annotated = img_bgr.copy()
    detections = []

    if result.boxes is None:
        return annotated, detections

    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf_v = float(box.conf[0])
        name   = yolo.names[cls_id]
        if driving_only and name not in DRIVING_CLASSES:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        detections.append({"class": name, "confidence": round(conf_v, 2), "box": [x1, y1, x2, y2]})

        color = (0, 255, 150)
        if name in ["person", "motorcycle", "bicycle"]:
            color = (0, 80, 255)
        elif name in ["traffic light", "stop sign"]:
            color = (0, 212, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label   = f"{name.upper()} {conf_v:.2f}"
        label_y = max(y1 - 10, 25)
        cv2.rectangle(annotated, (x1, label_y-22), (x1 + max(120, len(label)*11), label_y+4), color, -1)
        cv2.putText(annotated, label, (x1+5, label_y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (5, 10, 18), 2, cv2.LINE_AA)
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(annotated, (cx, cy), 4, color, -1)

    return annotated, detections


def draw_overlay(img, fps=None, inf_time=None, n_det=0, mode="LIVE"):
    out = img.copy()
    cv2.putText(out, f"NEXTGEN VISION AI | {mode}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2, cv2.LINE_AA)
    line2 = f"Objects: {n_det}"
    if fps is not None:       line2 += f" | FPS: {fps:.1f}"
    if inf_time is not None:  line2 += f" | Inference: {inf_time:.2f}s"
    cv2.putText(out, line2, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 212, 255), 2, cv2.LINE_AA)
    return out


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙ System Configuration")

    if not os.path.exists(DEHAZE_MODEL_PATH):
        st.warning("Dehazing model not downloaded.")
        if st.button("Download Model"):
            with st.spinner("Downloading from Google Drive..."):
                ok, err = download_dehaze_model_if_needed()
            if ok:
                st.success("Model downloaded.")
                st.rerun()
            else:
                st.error(err)
    else:
        st.success("Dehazing model ready ✓")

    if YOLO_AVAILABLE:
        st.success("YOLO available ✓")
    else:
        st.error("YOLO not installed")

    st.markdown("---")

    app_mode = st.radio("Mode", ["Image Upload", "Real-Time Camera"], index=0)

    st.markdown("---")
    st.markdown("### Dehazing")
    strength       = st.slider("Enhancement strength", 0.0, 1.0, 1.0, 0.05)
    dcp_only       = st.toggle("DCP only (faster)", value=False)
    inference_size = st.selectbox("Inference size", [128, 192, 256], index=1,
                                   help="Lower = faster. 192 recommended for real-time.")

    st.markdown("---")
    st.markdown("### Object Detection")
    enable_detection    = st.toggle("Enable YOLO", value=True)
    conf_threshold      = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
    only_driving        = st.toggle("Driving classes only", value=True)
    ar_style            = st.toggle("AR-style overlay", value=True)

    st.markdown("---")
    st.markdown("### Real-Time Optimisation")
    dehaze_every = st.slider(
        "Dehaze every N frames",
        min_value=1, max_value=6, value=3,
        help="1 = every frame (slow). 3 = smooth ~8fps. YOLO always runs every frame."
    )
    show_metrics = st.toggle("Show metrics panel", value=True)

    st.markdown("---")
    st.markdown(
        '<div class="small-note">Pipeline: frame → DCP dehazing → ResNet refinement → YOLOv8 → AR overlay</div>',
        unsafe_allow_html=True
    )


# ============================================================
# ENSURE MODEL IS DOWNLOADED
# ============================================================

if not os.path.exists(DEHAZE_MODEL_PATH):
    with st.spinner("Downloading dehazing model..."):
        ok, err = download_dehaze_model_if_needed()
    if ok:
        st.rerun()
    else:
        st.error(f"Could not download model: {err}")
        st.stop()

# Load models
try:
    with st.spinner("Loading dehazing model..."):
        dcp_model, resnet_model, device, missing_keys, unexpected_keys = load_dehaze_models()
except Exception as e:
    st.error(f"Dehazing model failed to load: {e}")
    st.stop()

if enable_detection:
    if not YOLO_AVAILABLE:
        st.error("Install ultralytics: add it to requirements.txt")
        st.stop()
    try:
        with st.spinner("Loading YOLO..."):
            yolo_model = load_yolo_model()
    except Exception as e:
        st.error(f"YOLO failed to load: {e}")
        st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("# 👁️ NEXTGEN VISION AI")
st.markdown("### Real-Time AR Vision Enhancement — Dehazing + Object Detection")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><div class="metric-val">READY</div><div class="metric-lbl">Dehazing Status</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{str(device).upper()}</div><div class="metric-lbl">Compute Device</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{"ON" if enable_detection else "OFF"}</div><div class="metric-lbl">YOLO Detection</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><div class="metric-val">DCP+RESNET+YOLO</div><div class="metric-lbl">Pipeline</div></div>', unsafe_allow_html=True)

st.markdown("")


# ============================================================
# IMAGE UPLOAD MODE
# ============================================================

if app_mode == "Image Upload":
    uploaded = st.file_uploader("Upload a hazy/foggy road image", type=["jpg","jpeg","png"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        st.stop()

    img_bgr = cv2.cvtColor(np.array(Image.open(uploaded).convert("RGB")), cv2.COLOR_RGB2BGR)

    t0         = time.time()
    dehazed    = dehaze_frame(img_bgr, dcp_model, resnet_model, device, strength, dcp_only, inference_size)
    dehaze_t   = time.time() - t0

    t1          = time.time()
    detections  = []
    if enable_detection:
        final, detections = detect_objects(dehazed, yolo_model, conf_threshold, only_driving, ar_style)
    else:
        final = dehazed
    detect_t = time.time() - t1

    final = draw_overlay(final, inf_time=dehaze_t+detect_t, n_det=len(detections), mode="IMAGE")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    with col2:
        st.image(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB), caption="Dehazed", use_container_width=True)
    with col3:
        st.image(cv2.cvtColor(final, cv2.COLOR_BGR2RGB), caption="Final + Detection", use_container_width=True)

    if show_metrics:
        os_, oc, osh = visibility_score(img_bgr)
        es_, ec, esh = visibility_score(dehazed)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original Visibility", f"{os_}%")
        m2.metric("Enhanced Visibility", f"{es_}%")
        m3.metric("Dehazing Time",       f"{dehaze_t:.2f}s")
        m4.metric("Objects Detected",    len(detections))

        st.markdown("#### Detections")
        if enable_detection:
            if detections:
                st.dataframe(detections, use_container_width=True)
            else:
                st.info("No driving-related objects detected.")
        else:
            st.info("YOLO is disabled.")

        with st.expander("Technical metrics"):
            st.json({
                "original_contrast":  oc,  "enhanced_contrast":  ec,
                "original_sharpness": osh, "enhanced_sharpness": esh,
                "dehaze_seconds":     round(dehaze_t, 3),
                "detect_seconds":     round(detect_t, 3),
                "total_seconds":      round(dehaze_t + detect_t, 3),
                "inference_size":     inference_size,
                "yolo_conf":          conf_threshold,
            })


# ============================================================
# REAL-TIME CAMERA MODE
# ============================================================

else:
    st.markdown("## 📹 Real-Time Camera")

    st.markdown("""
    <div class="info-box">
    <b>How frame skipping works here:</b> Dehazing runs every <b>N frames</b> (set in sidebar).
    YOLO runs <b>every frame</b> on the last dehazed result, keeping detections smooth.
    This gives fluid bounding boxes without re-running the heavy dehazing model each frame.
    </div>
    """, unsafe_allow_html=True)

    if not WEBRTC_AVAILABLE:
        st.error("Install streamlit-webrtc and av — add both to requirements.txt")
        st.stop()

    rtc_config = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

    # Snapshot references for the processor to read sidebar values
    # We use a simple container so the processor can read current slider values
    _cfg = {
        "strength":       strength,
        "dcp_only":       dcp_only,
        "size":           inference_size,
        "enable_det":     enable_detection,
        "conf":           conf_threshold,
        "driving_only":   only_driving,
        "ar_style":       ar_style,
        "dehaze_every":   dehaze_every,
    }

    class VisionProcessor(VideoProcessorBase):
        def __init__(self):
            # Load models once into the processor — avoids repeated cache lookups per frame
            self.dcp, self.resnet, self.device, _, _ = load_dehaze_models()
            self.yolo         = load_yolo_model()

            # Frame skip state
            self.frame_count  = 0
            self.last_dehazed = None   # last dehazed frame (reused between dehaze calls)
            self.last_dets    = []     # last detections (reused between dehaze calls)

            # FPS tracking
            self.last_time    = time.time()
            self.fps          = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            cfg = _cfg  # read current sidebar config snapshot

            try:
                # --- DEHAZING (every N frames) ---
                if self.frame_count % cfg["dehaze_every"] == 0 or self.last_dehazed is None:
                    self.last_dehazed = dehaze_frame(
                        img,
                        self.dcp, self.resnet, self.device,
                        cfg["strength"], cfg["dcp_only"], cfg["size"]
                    )

                # --- YOLO (every frame, on last dehazed result) ---
                # We run YOLO on a copy of last_dehazed so the base image stays clean
                if cfg["enable_det"] and self.yolo is not None:
                    final, self.last_dets = detect_objects(
                        self.last_dehazed.copy(),
                        self.yolo,
                        cfg["conf"],
                        cfg["driving_only"],
                        cfg["ar_style"],
                    )
                else:
                    final          = self.last_dehazed.copy()
                    self.last_dets = []

                # --- FPS ---
                now            = time.time()
                dt             = now - self.last_time
                self.last_time = now
                self.fps       = 1.0 / dt if dt > 0 else self.fps

                final = draw_overlay(
                    final,
                    fps=self.fps,
                    n_det=len(self.last_dets),
                    mode=f"LIVE (dehaze/{cfg['dehaze_every']}f)"
                )

            except Exception as e:
                # Failsafe: show original frame with error message
                final = img.copy()
                cv2.putText(final, f"Error: {str(e)[:60]}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(final, format="bgr24")

    webrtc_streamer(
        key="nextgen-vision",
        video_processor_factory=VisionProcessor,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": {
                "width":     {"ideal": 640},
                "height":    {"ideal": 480},
                "frameRate": {"ideal": 10, "max": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )

    st.markdown("""
    <div class="info-box" style="margin-top:1rem">
    <b>Demo tip:</b> Point the camera at a screen showing a foggy road scene.
    Adjust "Dehaze every N frames" in the sidebar to tune the speed vs quality trade-off live.
    </div>
    """, unsafe_allow_html=True)
