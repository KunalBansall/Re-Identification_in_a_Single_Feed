# DeepSORT tracking and re-ID logic will go here 

import cv2
import csv
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.3)

def track_players(frames, detections):
    tracked_frames = []
    tracking_data = []

    for frame_idx, (frame, dets) in enumerate(zip(frames, detections)):
        valid_dets = []
        for d in dets:
            if (
                isinstance(d, (list, tuple))
                and len(d) == 6  # Expecting 6 values: x1, y1, x2, y2, conf, label
                and all(isinstance(x, (int, float)) for x in d[:5])
                and isinstance(d[5], str)
            ):
                bbox = [int(d[0]), int(d[1]), int(d[2]), int(d[3])]
                confidence = float(d[4])
                class_name = d[5]
                valid_dets.append((bbox, confidence, class_name))
            # else:
            #     print("Malformed detection skipped:", d)
        converted_dets = valid_dets

        # Additional debug print before tracking
        if converted_dets:
            for det in converted_dets:
                if not (isinstance(det, tuple) and len(det) == 3 and isinstance(det[0], (list, tuple))):
                    print("❌ Invalid detection format before tracking:", det)

        # Only call update_tracks if all detections are valid
        try:
            tracks = tracker.update_tracks(converted_dets, frame=frame)
        except Exception as e:
            print("Error in update_tracks with converted_dets:", converted_dets)
            print("Exception:", e)
            tracks = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())

            # Draw on frame
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {track_id}", (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # ✅ Save tracking info
            tracking_data.append([frame_idx, track_id, l, t, r, b])

        tracked_frames.append(frame)

    # ✅ Save CSV at the end
    csv_path = "outputs/tracking_data.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2"])
        writer.writerows(tracking_data)
    print(f"Tracking data exported to {csv_path}")

    return tracked_frames 