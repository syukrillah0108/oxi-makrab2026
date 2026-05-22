import cv2
import mediapipe as mp
import pyautogui
import time
import math

# --- KONFIGURASI ---
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 
SCREEN_W, SCREEN_H = pyautogui.size()

SMOOTHING = 5
prev_x, prev_y = 0, 0

# --- AMBANG BATAS ---
PINCH_THRESH = 0.035     # Jarak Jempol-Telunjuk untuk Klik
PINKY_BENT_THRESH = 0.05 # Jarak Kelingking ke telapak untuk Drag
DOUBLE_CLICK_GAP = 0.35  

model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mouse_data = {'tx': 0, 'ty': 0, 'd_pinch': 1.0, 'd_pinky': 1.0, 'has_hand': False}

def calculate_dist_3d(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def mp_callback(result, output_image, timestamp_ms):
    global mouse_data
    if not result.hand_landmarks:
        mouse_data['has_hand'] = False
        return
    mouse_data['has_hand'] = True
    lm = result.hand_landmarks[0]
    
    # 1. POSISI KURSOR (Pangkal Telunjuk)
    ref = lm[5]
    tx = max(0, min(1, (ref.x - 0.2) / 0.6))
    ty = max(0, min(1, (ref.y - 0.2) / 0.6))
    mouse_data['tx'], mouse_data['ty'] = int(tx * SCREEN_W), int(ty * SCREEN_H)
    
    # 2. JARAK PINCH (Jempol 4 & Telunjuk 8)
    mouse_data['d_pinch'] = calculate_dist_3d(lm[4], lm[8])
    
    # 3. JARAK KELINGKING (Ujung Kelingking 20 ke Pangkal Telunjuk 5 atau Telapak 0)
    # Kita gunakan jarak ujung kelingking ke pergelangan tangan (0) untuk deteksi tekukan
    mouse_data['d_pinky'] = calculate_dist_3d(lm[20], lm[0])

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=mp_callback
)

cap = cv2.VideoCapture(0)

# State Variables
is_dragging = False
is_pinching = False
last_click_time = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(time.time()*1000))

        if mouse_data['has_hand']:
            # SMOOTHING
            curr_x = prev_x + (mouse_data['tx'] - prev_x) / SMOOTHING
            curr_y = prev_y + (mouse_data['ty'] - prev_y) / SMOOTHING
            pyautogui.moveTo(int(curr_x), int(curr_y))
            prev_x, prev_y = curr_x, curr_y
            
            now = time.time()

            # --- LOGIKA 1: KLIK KIRI & DOUBLE CLICK (JEMPOL-TELUNJUK) ---
            if mouse_data['d_pinch'] < PINCH_THRESH:
                if not is_pinching:
                    if (now - last_click_time) < DOUBLE_CLICK_GAP:
                        pyautogui.doubleClick()
                        print("ACTION: DOUBLE CLICK")
                    else:
                        pyautogui.click()
                        print("ACTION: SINGLE CLICK")
                    is_pinching = True
                    last_click_time = now
            elif mouse_data['d_pinch'] > (PINCH_THRESH + 0.03):
                is_pinching = False

            # --- LOGIKA 2: DRAG & DROP (KELINGKING TEKUK) ---
            # Kita bandingkan jarak kelingking saat ini dengan ambang batas.
            # Normalnya kelingking lurus jaraknya jauh (> 0.15), kalau ditekuk jadi pendek (< 0.1)
            if not is_dragging and mouse_data['d_pinky'] < 0.12: # Nilai 0.12 bisa disesuaikan
                pyautogui.mouseDown(button='left')
                is_dragging = True
                print("MODE: DRAGGING (PINKY BENT)")
            elif is_dragging and mouse_data['d_pinky'] > 0.16:
                pyautogui.mouseUp(button='left')
                is_dragging = False
                print("MODE: RELEASE (PINKY STRAIGHT)")

            # UI FEEDBACK
            cv2.putText(frame, f"Pinky Dist: {mouse_data['d_pinky']:.2f}", (10, 30), 1, 1, (255,255,255), 1)
            status = "DRAGGING" if is_dragging else "READY"
            color = (0, 255, 0) if is_dragging else (0, 0, 255)
            cv2.circle(frame, (50, 80), 15, color, -1)
            cv2.putText(frame, status, (80, 90), 1, 1.5, color, 2)

        cv2.imshow('AI Mouse Pinky-Drag - Oxigen Software', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()