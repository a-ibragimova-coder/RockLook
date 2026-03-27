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

# Eye landmark indices for VERTICAL gaze (up/down)
# For refined_landmarks=True
LEFT_IRIS = 468
RIGHT_IRIS = 473
# Eye lids for vertical tracking
LEFT_EYE_UPPER = 159   # Upper eyelid
LEFT_EYE_LOWER = 145   # Lower eyelid  
RIGHT_EYE_UPPER = 386  # Upper eyelid
RIGHT_EYE_LOWER = 374  # Lower eyelid

is_playing = False
looking_at_camera = False

# ============================================
# TODO #1: Set your gaze threshold
# Lower = looking straight, Higher = looking away
GAZE_THRESHOLD = 0.03  # <-- ADJUST THIS VALUE
# ============================================

def calculate_vertical_gaze(face_landmarks, iris_idx, eye_upper_idx, eye_lower_idx):
    """Calculate vertical gaze position (0 = center, higher = looking down/up)"""
    iris = face_landmarks.landmark[iris_idx]
    eye_upper = face_landmarks.landmark[eye_upper_idx]
    eye_lower = face_landmarks.landmark[eye_lower_idx]
    
    iris_y = iris.y
    eye_upper_y = eye_upper.y
    eye_lower_y = eye_lower.y
    
    # Calculate eye height
    eye_height = abs(eye_lower_y - eye_upper_y)
    
    # Calculate where iris is relative to eye center
    eye_center_y = (eye_upper_y + eye_lower_y) / 2
    iris_offset = abs(iris_y - eye_center_y) / eye_height
    
    return iris_offset  # Returns value between 0 and 0.5

print("=== VERTICAL GAZE Music Controller ===")
print("Look directly at camera → Music plays")
print("Look up/down → Music pauses")
print("Press 'q' to quit\n")

# Test audio
import winsound
winsound.Beep(440, 200)
print("✓ Audio ready\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate vertical gaze for both eyes
            left_gaze = calculate_vertical_gaze(face_landmarks, LEFT_IRIS, LEFT_EYE_UPPER, LEFT_EYE_LOWER)
            right_gaze = calculate_vertical_gaze(face_landmarks, RIGHT_IRIS, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER)
            
            # Average gaze offset (how far from center vertically)
            gaze_offset = (left_gaze + right_gaze) / 2
            
            # Show individual gaze values for debugging
            cv2.putText(frame, f"Left Vertical Gaze: {left_gaze:.3f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Right Vertical Gaze: {right_gaze:.3f}", (10, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # ============================================
            # TODO #2: Music trigger logic based on vertical gaze
            # ============================================
            if gaze_offset < GAZE_THRESHOLD:
                if not looking_at_camera:
                    looking_at_camera = True
                    print(f"✓ Looking at camera! (vertical offset: {gaze_offset:.3f})")
                if not is_playing:
                    player.play()
                    is_playing = True
                    print("🎵 Music playing")
                cv2.putText(frame, "LOOKING AT CAMERA - MUSIC PLAYING", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                if looking_at_camera:
                    looking_at_camera = False
                    print(f"✗ Looking away (vertical offset: {gaze_offset:.3f})")
                if is_playing:
                    player.pause()
                    is_playing = False
                    print("⏸ Music paused")
                cv2.putText(frame, "LOOKING AWAY - MUSIC PAUSED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show gaze value on screen
            cv2.putText(frame, f"Vertical Gaze Offset: {gaze_offset:.3f} (Threshold: {GAZE_THRESHOLD})", 
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw face mesh
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS
            )
            
            # Draw circles on irises
            h, w, _ = frame.shape
            left_iris_point = face_landmarks.landmark[LEFT_IRIS]
            right_iris_point = face_landmarks.landmark[RIGHT_IRIS]
            cv2.circle(frame, (int(left_iris_point.x * w), int(left_iris_point.y * h)), 3, (0, 255, 255), -1)
            cv2.circle(frame, (int(right_iris_point.x * w), int(right_iris_point.y * h)), 3, (0, 255, 255), -1)
    
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Vertical Gaze Music Controller', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if is_playing:
    player.stop()
cap.release()
cv2.destroyAllWindows()
print("Done!")