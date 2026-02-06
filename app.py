import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tempfile

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

POSE_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

T_WIDTH = 384
T_HEIGHT = 288

# ---------------------------------------------------------------------------
# Fonctions pose / angle / 3D
# ---------------------------------------------------------------------------

def calculate_leg_flexion(keypoints_2d, keypoints_conf, ref_bones=None, conf_threshold=0.5):
    angles = []
    for hip_idx, knee_idx, ankle_idx in [(11, 13, 15), (12, 14, 16)]:
        if hip_idx < len(keypoints_2d) and ankle_idx < len(keypoints_2d):
            if (keypoints_conf[hip_idx] < conf_threshold or
                keypoints_conf[knee_idx] < conf_threshold or
                keypoints_conf[ankle_idx] < conf_threshold):
                continue

            knee_z = 0.0
            if ref_bones and (hip_idx, knee_idx) in ref_bones:
                d_2d = np.linalg.norm(keypoints_2d[hip_idx][:2] - keypoints_2d[knee_idx][:2])
                ref_len = ref_bones[(hip_idx, knee_idx)]
                if d_2d < ref_len:
                    knee_z = np.sqrt(ref_len**2 - d_2d**2)

            p_hip   = np.array([keypoints_2d[hip_idx][0],   keypoints_2d[hip_idx][1],   0.0])
            p_knee  = np.array([keypoints_2d[knee_idx][0],  keypoints_2d[knee_idx][1],  knee_z])
            p_ankle = np.array([keypoints_2d[ankle_idx][0], keypoints_2d[ankle_idx][1], 0.0])

            v1 = p_hip - p_knee
            v2 = p_ankle - p_knee
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cosine = np.dot(v1, v2) / (n1 * n2)
                angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return min(angles) if angles else None


def estimate_3d_from_2d(keypoints_2d, image_width, image_height, ref_bones=None):
    cx, cy = image_width / 2, image_height / 2
    kp3d = np.zeros((len(keypoints_2d), 3))
    kp3d[:, 0] = -(keypoints_2d[:, 0] - cx) / image_width * 2
    kp3d[:, 1] = -(keypoints_2d[:, 1] - cy) / image_height * 2

    z_scale = 3.0
    p2n = 2.0 / image_width

    if len(keypoints_2d) >= 17:
        for i in [5, 6, 11, 12, 15, 16]:
            kp3d[i, 2] = 0.0
        kp3d[0, 2] = 0.05 * z_scale
        for i in [7, 8]:
            ref = 5 if i == 7 else 6
            dx = abs(keypoints_2d[i, 0] - keypoints_2d[ref, 0])
            kp3d[i, 2] = (-dx / image_width) * z_scale
        for i in [9, 10]:
            elbow = 7 if i == 9 else 8
            kp3d[i, 2] = kp3d[elbow, 2] - (0.1 * z_scale)

        for hip_idx, knee_idx in [(11, 13), (12, 14)]:
            if ref_bones and (hip_idx, knee_idx) in ref_bones:
                d_2d = np.linalg.norm(keypoints_2d[hip_idx] - keypoints_2d[knee_idx])
                ref_len = ref_bones[(hip_idx, knee_idx)]
                if d_2d < ref_len:
                    kp3d[knee_idx, 2] = np.sqrt(ref_len**2 - d_2d**2) * p2n
                else:
                    kp3d[knee_idx, 2] = 0.0
            else:
                kp3d[knee_idx, 2] = 0.1 * z_scale
    return kp3d


def draw_3d_skeleton(ax, keypoints_3d, connections):
    ax.clear()
    if len(keypoints_3d) > 12:
        hip_center = (keypoints_3d[11] + keypoints_3d[12]) / 2
        keypoints_3d = keypoints_3d - hip_center

    for s, e in connections:
        if s < len(keypoints_3d) and e < len(keypoints_3d):
            color = 'red' if s > 10 else ('green' if s in [5, 6, 11, 12] else 'blue')
            ax.plot([keypoints_3d[s, 0], keypoints_3d[e, 0]],
                    [keypoints_3d[s, 2], keypoints_3d[e, 2]],
                    [keypoints_3d[s, 1], keypoints_3d[e, 1]],
                    color=color, linewidth=4, alpha=0.9)

    ax.set_zlim([-0.9, 0.9])
    ax.set_ylim([-0.4, 0.4])
    ax.set_xlim([-0.8, 0.8])
    ax.set_box_aspect((1, 1, 1))
    ax.axis('off')
    ax.view_init(elev=0, azim=-90)


def render_3d_to_bgr(ax, fig, kp_3d, target_size):
    """Render 3D skeleton et retourne une image BGR a la taille voulue."""
    draw_3d_skeleton(ax, kp_3d, POSE_CONNECTIONS)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_rgb = buf.reshape((h, w, 4))[:, :, 1:]  # ARGB -> RGB
    return cv2.resize(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), target_size)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model(path):
    return YOLO(path)


def calibrate_bones(kp_2d, kp_conf):
    bones = {}
    for hip, knee, ankle in [(11, 13, 15), (12, 14, 16)]:
        if all(kp_conf[i] > 0.5 for i in [hip, knee, ankle]):
            bones[(hip, knee)]   = np.linalg.norm(kp_2d[hip][:2]  - kp_2d[knee][:2])
            bones[(knee, ankle)] = np.linalg.norm(kp_2d[knee][:2] - kp_2d[ankle][:2])
    return bones if bones else None


def infer_frame(frame, model, ref_bones):
    """Inference pure. Retourne (kp_2d, kp_3d, kp_conf, angle, ref_bones)."""
    h, w = frame.shape[:2]
    results = model(frame, verbose=False, conf=0.5)
    kp_2d = kp_3d = kp_conf = None
    angle = None

    if results[0].keypoints.xy is not None and len(results[0].keypoints.xy) > 0:
        kp_2d   = results[0].keypoints.xy[0].cpu().numpy()
        kp_conf = results[0].keypoints.conf[0].cpu().numpy()

        if ref_bones is None:
            ref_bones = calibrate_bones(kp_2d, kp_conf)

        kp_3d = estimate_3d_from_2d(kp_2d, w, h, ref_bones)
        angle = calculate_leg_flexion(kp_2d, kp_conf, ref_bones)

    return kp_2d, kp_3d, kp_conf, angle, ref_bones


def draw_overlay(frame, kp_2d, angle, counter, stage):
    """Dessine squelette + info directement sur le frame (BGR). Modifie en place."""
    if kp_2d is not None:
        for s, e in POSE_CONNECTIONS:
            if s < len(kp_2d) and e < len(kp_2d):
                cv2.line(frame,
                         tuple(kp_2d[s].astype(int)),
                         tuple(kp_2d[e].astype(int)),
                         (0, 255, 0), 2)

    cv2.rectangle(frame, (0, 0), (170, 85), (0, 0, 0), -1)
    if angle is not None:
        col = (0, 255, 0) if angle < 90 else (255, 255, 255)
        cv2.putText(frame, f"{angle:.0f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    else:
        cv2.putText(frame, "-- deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    cv2.putText(frame, f"Count: {counter}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, stage, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


def update_counter_state(angle):
    if angle is None:
        return
    if angle < 90 and st.session_state.stage == "UP":
        st.session_state.stage = "DOWN"
        st.session_state.min_angle = angle
    if st.session_state.stage == "DOWN":
        st.session_state.min_angle = min(st.session_state.min_angle, angle)
    if st.session_state.stage == "DOWN" and angle > 120 and st.session_state.min_angle < 70:
        st.session_state.stage = "UP"
        st.session_state.counter += 1
        st.session_state.min_angle = 180


def update_counter_local(angle, counter, stage, min_angle):
    """Version sans session_state pour le traitement video offline."""
    if angle is None:
        return counter, stage, min_angle
    if angle < 90 and stage == "UP":
        stage = "DOWN"
        min_angle = angle
    if stage == "DOWN":
        min_angle = min(min_angle, angle)
    if stage == "DOWN" and angle > 120 and min_angle < 70:
        stage = "UP"
        counter += 1
        min_angle = 180
    return counter, stage, min_angle


def reset_state():
    st.session_state.counter   = 0
    st.session_state.stage     = "UP"
    st.session_state.min_angle = 180
    st.session_state.ref_bones = None


def open_video_writer(path, fourcc_str, fps, size):
    """Tente d'ouvrir un VideoWriter. Retourne (writer, True) ou (None, False)."""
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    out = cv2.VideoWriter(path, fourcc, fps, size)
    if out.isOpened():
        return out, True
    out.release()
    return None, False

# ---------------------------------------------------------------------------
# Mode Webcam  -- une seule image composite par frame, zero widget reconstruit
# ---------------------------------------------------------------------------

def run_webcam(model):
    run = st.toggle("Activer la webcam")
    frame_ph = st.empty()

    if not run:
        frame_ph.info("Activez la webcam pour commencer")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la webcam")
        return

    fig = plt.figure(figsize=(4, 4))
    ax  = fig.add_subplot(111, projection='3d')

    # Image 3D persistante (mise a jour toutes les N frames)
    canvas_3d = np.ones((T_HEIGHT, T_WIDTH, 3), dtype=np.uint8) * 255
    n = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_s = cv2.resize(frame, (T_WIDTH, T_HEIGHT))
        frame_s = cv2.flip(frame_s, 1)

        kp_2d, kp_3d, _, angle, st.session_state.ref_bones = infer_frame(
            frame_s, model, st.session_state.ref_bones)
        update_counter_state(angle)

        # Overlay 2D
        canvas_2d = frame_s.copy()
        draw_overlay(canvas_2d, kp_2d, angle,
                     st.session_state.counter, st.session_state.stage)

        # Rendu 3D toutes les 10 frames seulement
        if kp_3d is not None and n % 10 == 0:
            canvas_3d = render_3d_to_bgr(ax, fig, kp_3d, (T_WIDTH, T_HEIGHT))

        # Une seule image envoyee a Streamlit
        combined = np.hstack([canvas_2d, canvas_3d])
        frame_ph.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB),
                       use_container_width=True)
        n += 1

    cap.release()
    plt.close(fig)

# ---------------------------------------------------------------------------
# Mode Video  -- traitement offline puis lecture fluide avec st.video()
# ---------------------------------------------------------------------------

def run_video(model):
    uploaded = st.file_uploader("Charger une video", type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.info("Chargez une video pour lancer l'analyse")
        return

    # Sauvegarder l'upload
    in_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_file.write(uploaded.read())
    in_file.close()

    cap = cv2.VideoCapture(in_file.name)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    t_h    = int(h_orig * T_WIDTH / w_orig)
    out_size = (T_WIDTH * 2, t_h)  # [canvas_2d | canvas_3d]

    # Fichier de sortie
    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_file.name
    out_file.close()

    # Essayer plusieurs codecs pour compatibilite navigateur
    writer = None
    for codec in ['avc1', 'H264', 'mp4v']:
        writer, ok = open_video_writer(out_path, codec, fps, out_size)
        if ok:
            break
    if writer is None or not writer.isOpened():
        st.error("Impossible de creer le fichier video de sortie (aucun codec disponible)")
        cap.release()
        return

    fig = plt.figure(figsize=(4, 4))
    ax  = fig.add_subplot(111, projection='3d')

    ref_bones = None
    counter, stage, min_angle = 0, "UP", 180
    canvas_3d = np.ones((t_h, T_WIDTH, 3), dtype=np.uint8) * 255

    status_text = st.empty()
    progress    = st.progress(0)
    n = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_s = cv2.resize(frame, (T_WIDTH, t_h))
        kp_2d, kp_3d, _, angle, ref_bones = infer_frame(frame_s, model, ref_bones)
        counter, stage, min_angle = update_counter_local(angle, counter, stage, min_angle)

        # Overlay 2D
        canvas_2d = frame_s.copy()
        draw_overlay(canvas_2d, kp_2d, angle, counter, stage)

        # Rendu 3D toutes les 5 frames
        if kp_3d is not None and n % 5 == 0:
            canvas_3d = render_3d_to_bgr(ax, fig, kp_3d, (T_WIDTH, t_h))

        writer.write(np.hstack([canvas_2d, canvas_3d]))

        n += 1
        if n % 10 == 0 and total > 0:
            pct = min(n / total, 1.0)
            progress.progress(pct)
            status_text.text(f"Traitement : {n}/{total} frames  --  {counter} squats")

    cap.release()
    writer.release()
    plt.close(fig)

    progress.progress(1.0)
    status_text.empty()

    # Lecture du resultat
    with open(out_path, "rb") as f:
        video_bytes = f.read()

    os.unlink(in_file.name)
    os.unlink(out_path)

    st.success(f"Analyse terminee -- {counter} squats detectes")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.video(video_bytes)
    with col2:
        st.metric("Squats", counter)
        st.download_button("Telecharger la video", video_bytes,
                           file_name="squat_analysis.mp4", mime="video/mp4")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Squat Counter AI", layout="wide")
    st.title("Squat Counter AI")

    for k, v in {"counter": 0, "stage": "UP", "min_angle": 180, "ref_bones": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v

    mode = st.radio("Source", ["Webcam", "Video"], horizontal=True)

    if st.button("Reset compteur"):
        reset_state()
        st.rerun()

    model = load_model("yolo26n-pose.pt")

    if mode == "Webcam":
        run_webcam(model)
    else:
        run_video(model)


if __name__ == "__main__":
    main()
