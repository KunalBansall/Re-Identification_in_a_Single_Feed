# Entry point for player re-identification project 
from detector import detect_players
from tracker import track_players
import cv2
import os

def save_video(frames, output_path="outputs/output.mp4", fps=30):
    if not frames:
        print("No frames to save.")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Output video saved to {output_path}")

def main():
    # Paths
    video_path = "videos/15sec_input_720p.mp4"
    model_path = "models/best.pt"

    print("Running detection...")
    frames, detections = detect_players(video_path, model_path)

    print("Running tracking...")
    tracked_frames = track_players(frames, detections)

    print("Saving final output...")
    os.makedirs("outputs", exist_ok=True)
    save_video(tracked_frames)

if __name__ == "__main__":
    main() 