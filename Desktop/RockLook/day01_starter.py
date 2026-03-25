import cv2
import mediapipe as mp
import vlc
import sys
import os
import time

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
music_loaded = False
is_playing = False
face_detected = False

# Check if music file exists
if not os.path.exists(MUSIC_FILE):
    print(f"✗ ERROR: '{MUSIC_FILE}' not found!")
    print(f"   Current folder: {os.getcwd()}")
    print("   Place an MP3 file named 'music.mp3' in this folder")
    sys.exit(1)

print(f"✓ Loaded: {MUSIC_FILE}")

# Initialize VLC player
player = vlc.MediaPlayer(MUSIC_FILE)
player.audio_set_volume(100)  # Max volume

# Test with a beep to confirm audio
import winsound
print("Testing audio...")
winsound.Beep(440, 200)
print("✓ Audio working!\n")

print("=== Face Music Controller ===")
print("Show your face → Music plays")
print("Look away → Music pauses")
print("Press 'q' to quit")
print("Click on the camera window first!\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    # Check if face detected
    if results.multi_face_landmarks:
        if not face_detected:
            print("✓ Face detected - Playing music")
            face_detected = True
            
            # Start or resume music
            if not is_playing:
                player.play()
                is_playing = True
        
        # Draw face mesh
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS
            )
        
        cv2.putText(frame, "FACE DETECTED - MUSIC PLAYING", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        if face_detected:
            print("✗ No face - Pausing music")
            face_detected = False
            
            # Pause music
            if is_playing:
                player.pause()
                is_playing = False
        
        cv2.putText(frame, "NO FACE - MUSIC PAUSED", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Show instructions
    cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display
    cv2.imshow('Face Music Controller', frame)
    
    # Handle quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("\nCleaning up...")
if is_playing:
    player.stop()
cap.release()
cv2.destroyAllWindows()
print("Done!")