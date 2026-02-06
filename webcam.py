import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

POSE_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def calculate_leg_flexion(keypoints_2d, keypoints_conf, ref_bones=None, conf_threshold=0.5):
    angles = []
    LEG_JOINTS = [(11, 13, 15), (12, 14, 16)]
    for hip_idx, knee_idx, ankle_idx in LEG_JOINTS:
        if hip_idx < len(keypoints_2d) and ankle_idx < len(keypoints_2d):
            if (keypoints_conf[hip_idx] < conf_threshold or
                keypoints_conf[knee_idx] < conf_threshold or
                keypoints_conf[ankle_idx] < conf_threshold):
                continue

            # Estimation profondeur genou par conservation longueur d'os
            knee_z = 0.0
            if ref_bones and (hip_idx, knee_idx) in ref_bones:
                d_2d = np.linalg.norm(keypoints_2d[hip_idx][:2] - keypoints_2d[knee_idx][:2])
                ref_len = ref_bones[(hip_idx, knee_idx)]
                if d_2d < ref_len:
                    knee_z = np.sqrt(ref_len**2 - d_2d**2)

            # Points 3D en espace pixel (X, Y pixel + Z estimé)
            p_hip = np.array([keypoints_2d[hip_idx][0], keypoints_2d[hip_idx][1], 0.0])
            p_knee = np.array([keypoints_2d[knee_idx][0], keypoints_2d[knee_idx][1], knee_z])
            p_ankle = np.array([keypoints_2d[ankle_idx][0], keypoints_2d[ankle_idx][1], 0.0])

            v1 = p_hip - p_knee
            v2 = p_ankle - p_knee
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            if norm_v1 > 0 and norm_v2 > 0:
                cosine = np.dot(v1, v2) / (norm_v1 * norm_v2)
                angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                angles.append(angle)
    return min(angles) if angles else None

def estimate_3d_from_2d(keypoints_2d, image_width, image_height, ref_bones=None):
    center_x, center_y = image_width / 2, image_height / 2
    keypoints_3d = np.zeros((len(keypoints_2d), 3))

    keypoints_3d[:, 0] = -(keypoints_2d[:, 0] - center_x) / image_width * 2
    keypoints_3d[:, 1] = -(keypoints_2d[:, 1] - center_y) / image_height * 2

    z_scale = 3.0
    pixel_to_norm = 2.0 / image_width

    if len(keypoints_2d) >= 17:
        for i in [5, 6, 11, 12, 15, 16]:
            keypoints_3d[i, 2] = 0.0
        keypoints_3d[0, 2] = 0.05 * z_scale
        for i in [7, 8]:
            ref = 5 if i == 7 else 6
            dx = abs(keypoints_2d[i, 0] - keypoints_2d[ref, 0])
            keypoints_3d[i, 2] = (-dx / image_width) * z_scale
        for i in [9, 10]:
            elbow = 7 if i == 9 else 8
            keypoints_3d[i, 2] = keypoints_3d[elbow, 2] - (0.1 * z_scale)

        # Genoux : profondeur par conservation longueur d'os
        for hip_idx, knee_idx in [(11, 13), (12, 14)]:
            if ref_bones and (hip_idx, knee_idx) in ref_bones:
                d_2d = np.linalg.norm(keypoints_2d[hip_idx] - keypoints_2d[knee_idx])
                ref_len = ref_bones[(hip_idx, knee_idx)]
                if d_2d < ref_len:
                    keypoints_3d[knee_idx, 2] = np.sqrt(ref_len**2 - d_2d**2) * pixel_to_norm
                else:
                    keypoints_3d[knee_idx, 2] = 0.0
            else:
                keypoints_3d[knee_idx, 2] = 0.1 * z_scale

    return keypoints_3d

def draw_3d_skeleton(ax, keypoints_3d, connections):
    ax.clear()
    
    if len(keypoints_3d) > 12:
        hip_center = (keypoints_3d[11] + keypoints_3d[12]) / 2
        keypoints_3d = keypoints_3d - hip_center

    for start, end in connections:
        if start < len(keypoints_3d) and end < len(keypoints_3d):
            color = 'red' if start > 10 else ('green' if start in [5, 6, 11, 12] else 'blue')
            ax.plot([keypoints_3d[start, 0], keypoints_3d[end, 0]],
                    [keypoints_3d[start, 2], keypoints_3d[end, 2]],
                    [keypoints_3d[start, 1], keypoints_3d[end, 1]], 
                    color=color, linewidth=4, alpha=0.9)

    ax.set_zlim([-0.9, 0.9]) 
    depth_zoom = 0.4 
    ax.set_ylim([-depth_zoom, depth_zoom])
    ax.set_xlim([-0.8, 0.8])
    ax.set_box_aspect((1, 1, 1))
    ax.axis('off')
    ax.view_init(elev=0, azim=-90)

def start_webcam_analysis():
    print("Chargement modèle...")
    model = YOLO('yolo26n-pose.pt')
    
    # 0 pour la webcam par défaut
    cap = cv2.VideoCapture(0)
    
    # Taille de traitement
    t_width = 384
    t_height = 288 # 4:3 ratio standard webcam
    
    fig = plt.figure(figsize=(5, 5))
    ax_3d = fig.add_subplot(111, projection='3d')
    
    # Variables Compteur
    counter = 0
    stage = "UP" # UP ou DOWN
    min_angle_in_down = 180  # Angle min atteint pendant la descente
    ref_bones = None  # Longueurs d'os calibrées (première frame debout)
    
    print("Démarrage Webcam (Appuyez sur 'q' pour quitter)...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_s = cv2.resize(frame, (t_width, t_height))
        # Flip horizontal pour effet miroir naturel
        frame_s = cv2.flip(frame_s, 1)
        
        results = model(frame_s, verbose=False, conf=0.5)
        
        kp_2d, kp_3d = None, None
        flexion_angle = None

        if results[0].keypoints.xy is not None and len(results[0].keypoints.xy) > 0:
            kp_2d = results[0].keypoints.xy[0].cpu().numpy()
            kp_conf = results[0].keypoints.conf[0].cpu().numpy()

            # Calibration des longueurs d'os (première frame avec jambes visibles)
            if ref_bones is None:
                bones = {}
                for hip_idx, knee_idx, ankle_idx in [(11, 13, 15), (12, 14, 16)]:
                    if (kp_conf[hip_idx] > 0.5 and kp_conf[knee_idx] > 0.5 and kp_conf[ankle_idx] > 0.5):
                        bones[(hip_idx, knee_idx)] = np.linalg.norm(kp_2d[hip_idx][:2] - kp_2d[knee_idx][:2])
                        bones[(knee_idx, ankle_idx)] = np.linalg.norm(kp_2d[knee_idx][:2] - kp_2d[ankle_idx][:2])
                if bones:
                    ref_bones = bones
                    print("Calibration os OK - tenez-vous debout pour calibrer")

            kp_3d = estimate_3d_from_2d(kp_2d, t_width, t_height, ref_bones)
            flexion_angle = calculate_leg_flexion(kp_2d, kp_conf, ref_bones)

            # --- LOGIQUE COMPTEUR (seulement si les jambes sont visibles) ---
            if flexion_angle is not None:
                # Phase DOWN : angle descend sous 90° (on s'abaisse)
                if flexion_angle < 90 and stage == "UP":
                    stage = "DOWN"
                    min_angle_in_down = flexion_angle

                # Pendant la phase DOWN, on garde le min atteint
                if stage == "DOWN":
                    min_angle_in_down = min(min_angle_in_down, flexion_angle)

                # Phase UP : retour au-dessus de 120° après être descendu sous 70°
                if stage == "DOWN" and flexion_angle > 120 and min_angle_in_down < 70:
                    stage = "UP"
                    counter += 1
                    min_angle_in_down = 180  # Reset
                    print(f"Squat validé ! Total: {counter}")
        
        # Rendu 3D
        canvas_3d = np.ones((t_height, t_width, 3), dtype=np.uint8) * 255
        if kp_3d is not None:
            draw_3d_skeleton(ax_3d, kp_3d, POSE_CONNECTIONS)
            fig.canvas.draw()
            img_plot = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            img_plot = img_plot.reshape((h, w, 4))  # ARGB a 4 canaux
            img_plot = img_plot[:, :, 1:]  # Enlever le canal alpha, garder RGB
            canvas_3d = cv2.resize(cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR), (t_width, t_height))

        # Rendu 2D
        canvas_2d = frame_s.copy()
        if kp_2d is not None:
            for s, e in POSE_CONNECTIONS:
                if s < len(kp_2d) and e < len(kp_2d):
                    cv2.line(canvas_2d, tuple(kp_2d[s].astype(int)), tuple(kp_2d[e].astype(int)), (0,255,0), 2)

        # Affichage UI
        # Boite info
        cv2.rectangle(canvas_2d, (0,0), (150, 85), (0,0,0), -1)
        
        # Angle
        if flexion_angle is not None:
            color_angle = (0, 255, 0) if flexion_angle < 90 else (255, 255, 255)
            cv2.putText(canvas_2d, f"{flexion_angle:.0f} deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_angle, 2)
        else:
            cv2.putText(canvas_2d, "-- deg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        
        # Compteur
        cv2.putText(canvas_2d, f"Count: {counter}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Assemblage
        combined = np.hstack([canvas_2d, canvas_3d])
        
        cv2.imshow('Squat Counter AI', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    plt.close()

if __name__ == "__main__":
    start_webcam_analysis()