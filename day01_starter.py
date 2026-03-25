import cv2
import mediapipe as mp
import vlc
import sys
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: No webcam found.")
    sys.exit(1)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Music settings
MUSIC_FILE = "music.mp3"
if not os.path.exists(MUSIC_FILE):
    print(f"ERROR: '{MUSIC_FILE}' not found!")
    sys.exit(1)

player = vlc.MediaPlayer(MUSIC_FILE)
player.audio_set_volume(100)

# Eye landmark indices
LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_EYE_OUTER = 33
LEFT_EYE_INNER = 133
RIGHT_EYE_OUTER = 362
RIGHT_EYE_INNER = 263

is_playing = False
looking_at_camera = False

# ============================================
# TODO #1: Set your gaze threshold
GAZE_THRESHOLD = 0.02  # <-- ADJUST THIS VALUE
# ============================================

def calculate_iris_position(face_landmarks, iris_idx, eye_outer_idx, eye_inner_idx):
    iris = face_landmarks.landmark[iris_idx]
    eye_outer = face_landmarks.landmark[eye_outer_idx]
    eye_inner = face_landmarks.landmark[eye_inner_idx]
    
    iris_x = iris.x
    eye_outer_x = eye_outer.x
    eye_inner_x = eye_inner.x
    
    eye_width = abs(eye_inner_x - eye_outer_x)
    eye_center = (eye_outer_x + eye_inner_x) / 2
    iris_offset = abs(iris_x - eye_center) / eye_width
    
    return iris_offset

print("=== Gaze Music Controller ===")
print("Look at camera → Music plays")
print("Look away → Music pauses")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate gaze
            left_gaze = calculate_iris_position(face_landmarks, LEFT_IRIS, LEFT_EYE_OUTER, LEFT_EYE_INNER)
            right_gaze = calculate_iris_position(face_landmarks, RIGHT_IRIS, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
            gaze_offset = (left_gaze + right_gaze) / 2
            
            # ============================================
            # TODO #2: Music trigger logic
            # ============================================
            if gaze_offset < GAZE_THRESHOLD:
                if not looking_at_camera:
                    looking_at_camera = True
                if not is_playing:
                    player.play()
                    is_playing = True
                cv2.putText(frame, "LOOKING AT CAMERA - MUSIC PLAYING", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                if looking_at_camera:
                    looking_at_camera = False
                if is_playing:
                    player.pause()
                    is_playing = False
                cv2.putText(frame, "LOOKING AWAY - MUSIC PAUSED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Gaze: {gaze_offset:.3f} (Threshold: {GAZE_THRESHOLD})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face mesh
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS
            )
    
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Gaze Music Controller', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if is_playing:
    player.stop()
cap.release()
cv2.destroyAllWindows()
print("Done!")