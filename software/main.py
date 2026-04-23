import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import math

# --- KONFIGURASI MODEL HAND ---
model_path = 'hand_landmarker.task' 
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Ambang batas jepitan (Makin kecil makin harus rapat jarinya)
PINCH_THRESHOLD = 0.04 

# Variabel Global AI
mouse_data = {
    'index_x': 0.0,
    'index_y': 0.0,
    'is_pinching': False,
    'has_hand': False
}

def calculate_dist(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def mp_asyn_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global mouse_data
    if not result.hand_landmarks:
        mouse_data['has_hand'] = False
        return
    
    mouse_data['has_hand'] = True
    landmarks = result.hand_landmarks[0]
    
    # Ambil posisi ujung telunjuk sebagai titik kursor utama
    index_tip = landmarks[8] 
    mouse_data['index_x'] = index_tip.x
    mouse_data['index_y'] = index_tip.y
    
    # Deteksi Jepitan (Jempol + Telunjuk)
    thumb_tip = landmarks[4]
    mouse_data['is_pinching'] = calculate_dist(thumb_tip, index_tip) < PINCH_THRESHOLD

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=mp_asyn_callback
)

# ==========================================
# ELEMEN GUI VIRTUAL (KOTAK DRAG & DROP)
# ==========================================
box_x, box_y = 250, 200  # Posisi awal kotak
box_size = 150           # Ukuran kotak
is_dragging = False      # Status apakah kotak sedang dijepit

cap = cv2.VideoCapture(0)
print("Virtual Sandbox Aktif! Coba sentuh dan cubit kotaknya!")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1) 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        h, w, _ = frame.shape
        cursor_inside_box = False
        
        # --- LOGIKA VIRTUAL DRAG & DROP ---
        if mouse_data['has_hand']:
            # Konversi kursor dari AI (0.0-1.0) ke resolusi piksel layar kamera
            cx = int(mouse_data['index_x'] * w)
            cy = int(mouse_data['index_y'] * h)
            
            # Cek tabrakan: Apakah kursor jari berada di dalam area kotak?
            cursor_inside_box = (box_x < cx < box_x + box_size) and (box_y < cy < box_y + box_size)

            if mouse_data['is_pinching']:
                if cursor_inside_box and not is_dragging:
                    is_dragging = True # Mulai menjepit kotak
                
                if is_dragging:
                    # Geser posisi kotak mengikuti jari (jari selalu di tengah kotak)
                    box_x = cx - (box_size // 2)
                    box_y = cy - (box_size // 2)
            else:
                is_dragging = False # Lepaskan jepitan
                
            # Gambar Kursor Tangan (Lingkaran)
            cursor_color = (0, 0, 255) if mouse_data['is_pinching'] else (255, 255, 255)
            cv2.circle(frame, (cx, cy), 12, cursor_color, -1)

        # --- WARNA KOTAK VIRTUAL ---
        if is_dragging:
            box_color = (0, 255, 0)     # HIJAU: Saat dijepit & digeser
            text = "DIGESER!"
        elif cursor_inside_box:
            box_color = (0, 255, 255)   # KUNING: Saat jari di atas kotak (Hover) tapi belum dicubit
            text = "CUBIT SAYA"
        else:
            box_color = (255, 0, 0)     # BIRU: Saat diam tidak disentuh
            text = "KOTAK UI"

        # Gambar Kotak dan Teks
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), box_color, -1)
        
        # Penyesuaian posisi teks agar ke tengah
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx = box_x + (box_size - text_size[0]) // 2
        ty = box_y + (box_size + text_size[1]) // 2
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        cv2.imshow('AI Virtual Sandbox - Oxigen Demo', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()