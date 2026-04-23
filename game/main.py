import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import keyboard as kbd  # Menggantikan pynput
import time

# ========================================================
# ⚙️ PENGATURAN KALIBRASI ZONA NETRAL (DEADZONE)
# ========================================================
BATAS_ATAS  = 0.20  
BATAS_BAWAH = 0.65  
BATAS_KIRI  = 0.35  
BATAS_KANAN = 0.65  

# --- KONFIGURASI MODEL ---
model_path = 'pose_landmarker_full.task' 
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

current_key = None

def handle_keyboard(key):
    """Fungsi eksekusi tombol menggunakan library 'keyboard' (Level OS)"""
    global current_key
    if current_key != key:
        # 1. Lepas tombol sebelumnya
        if current_key:
            kbd.release(current_key)
        
        # 2. Tekan tombol baru
        if key:
            kbd.press(key)
            print(f"[ACTION] Menekan Tombol: {key.upper()}")
        else:
            print("[ACTION] Kembali ke Zona Netral (Berhenti)")
        
        current_key = key

def print_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if not result.pose_landmarks:
        handle_keyboard(None)
        return

    landmarks = result.pose_landmarks[0]
    nose = landmarks[0]         

    action = "DIAM"
    
    if nose.y < BATAS_ATAS:
        handle_keyboard('up')          # Panah Atas
        action = "ATAS (UP)"
    elif nose.y > BATAS_BAWAH:
        handle_keyboard('down')        # Panah Bawah
        action = "BAWAH (DOWN)"
    elif nose.x < BATAS_KIRI:
        handle_keyboard('left')        # Panah Kiri
        action = "KIRI (LEFT)"
    elif nose.x > BATAS_KANAN:
        handle_keyboard('right')       # Panah Kanan
        action = "KANAN (RIGHT)"
    else:
        handle_keyboard(None)          # Diam

    print(f"Hidung X: {nose.x:.2f}, Y: {nose.y:.2f} | Action: {action}")

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# --- MAIN LOOP KAMERA ---
with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        h, w, _ = frame.shape
        x1, y1 = int(w * BATAS_KIRI), int(h * BATAS_ATAS)
        x2, y2 = int(w * BATAS_KANAN), int(h * BATAS_BAWAH)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "ZONA NETRAL", (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        status_text = f"Tombol Aktif: {current_key.upper() if current_key else 'NONE'}"
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow('Pose Game Controller - Oxigen', frame)

        if cv2.waitKey(1) & 0xFF == 27: 
            break

    # Lepas semua tombol saat program keluar (PENTING!)
    kbd.release_all()
    cap.release()
    cv2.destroyAllWindows()