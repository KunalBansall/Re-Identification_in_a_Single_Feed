# YOLOv11 player detection logic will go here 

import torch
import ultralytics.nn.tasks

# Allowlist required classes for PyTorch 2.6+ safe loading
try:
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
    import torch.nn.modules.container
    torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
    # Try both possible Conv import paths
    from ultralytics.nn.modules import Conv as Conv1
    torch.serialization.add_safe_globals([Conv1])
    import ultralytics.nn.modules.conv
    torch.serialization.add_safe_globals([ultralytics.nn.modules.conv.Conv])
except AttributeError:
    # For older torch versions, this function may not exist
    pass
except Exception as e:
    print(f"Warning during allowlisting: {e}")

from ultralytics import YOLO
import cv2


def detect_players(video_path: str, model_path: str):
    # Select device
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    model = YOLO(model_path)
    model.conf = 0.15  # Lower detection threshold
    cap = cv2.VideoCapture(video_path)

    all_frames = []
    all_detections = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1

        # Optional: Enhance frame contrast with CLAHE
        # Uncomment the following lines to enable CLAHE
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        # enhanced = clahe.apply(gray)
        # frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Run detection on the selected device
        results = model(frame, device=device, verbose=False)[0]
        player_boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            if cls == 0:  # Class 0 is Player
                player_boxes.append([x1, y1, x2, y2, conf, 'player'])

        all_frames.append(frame)
        all_detections.append(player_boxes)
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    return all_frames, all_detections 