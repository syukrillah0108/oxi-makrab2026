import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import keyboard as kbd 
import time

# ========================================================
# ⚙️ PENGATURAN KALIBRASI ZONA NETRAL (DEADZONE)
# ========================================================
BATAS_ATAS  = 0.25  
BATAS_BAWAH = 0.60  
BATAS_KIRI  = 0.40  
BATAS_KANAN = 0.60  

# --- KONFIGURASI MODEL ---
model_path = 'pose_landmarker_full.task' 
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

# State tombol agar tidak spam
current_move_key = None
is_space_pressed = False

def handle_controls(move_key, space_trigger):
    global current_move_key, is_space_pressed
    
    # 1. LOGIKA PERGERAKAN (Hidung)
    if current_move_key != move_key:
        if current_move_key:
            kbd.release(current_move_key)
        if move_key:
            kbd.press(move_key)
            print(f"[MOVE] {move_key.upper()}")
        current_move_key = move_key

    # 2. LOGIKA LOMPAT (Kelingking Kanan ditekuk)
    if space_trigger and not is_space_pressed:
        kbd.press('space')
        is_space_pressed = True
        print("[ACTION] SPACE DOWN")
    elif not space_trigger and is_space_pressed:
        kbd.release('space')
        is_space_pressed = False
        print("[ACTION] SPACE UP")

def print_result(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if not result.pose_landmarks:
        handle_controls(None, False)
        return

    landmarks = result.pose_landmarks[0]
    
    # Titik Hidung (Arah)
    nose = landmarks[0] 
    
    # Titik Kelingking Kanan (16: Wrist, 20: Pinky Tip)
    # Kita cek jarak ujung kelingking ke pergelangan tangan
    pinky_tip = landmarks[20]
    wrist = landmarks[16]
    dist_pinky = ((pinky_tip.x - wrist.x)**2 + (pinky_tip.y - wrist.y)**2)**0.5

    move_key = None
    # Logika Hidung
    if nose.y < BATAS_ATAS:
        move_key = 'up'
    elif nose.y > BATAS_BAWAH:
        move_key = 'down'
    elif nose.x < BATAS_KIRI:
        move_key = 'left'
    elif nose.x > BATAS_KANAN:
        move_key = 'right'

    # Logika Kelingking (Jika ditekuk/jarak kecil, tekan spasi)
    # Kalibrasi: 0.05 biasanya jarak saat ditekuk
    space_active = dist_pinky < 0.06 

    handle_controls(move_key, space_active)

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result
)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
print("Windows Pose Controller Aktif! Gunakan Hidung untuk Arah, Tekuk Kelingking untuk Spasi.")

with PoseLandmarker.create_from_options(options) as landmarker:
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
        
        # Gambar UI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        status = f"MOVE: {str(current_move_key).upper()} | SPACE: {'ACTIVE' if is_space_pressed else 'OFF'}"
        cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Windows Pose Game - Oxigen', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    kbd.release_all()
    cap.release()
    cv2.destroyAllWindows()