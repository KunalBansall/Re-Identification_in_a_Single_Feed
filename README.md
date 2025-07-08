# 🧠 Player Re-Identification from Single Camera Feed

This project performs **player detection and re-identification** using a single camera video feed. It leverages a YOLOv5-based custom model and tracking logic to identify and follow players across frames, producing an annotated output video.

## 🚀 Features

- Player detection using a pretrained YOLO model (`best.pt`)
- Multi-object tracking with unique IDs per player
- Output: annotated video + tracking CSV
- Lightweight, works on CPU

---

## 🧩 Project Structure

```
player_reid_single_feed/
├── .gitignore
├── README.md
├── detector.py
├── main.py
├── requirements.txt
├── tracker.py
├── __pycache__/
│   ├── detector.cpython-39.pyc
│   └── tracker.cpython-39.pyc
├── models/
│   ├── .gitkeep
│   └── best.pt
├── outputs/
│   ├── .gitkeep
│   ├── output.mp4
│   └── tracking_data.csv
├── videos/
│   ├── .gitkeep
│   └── 15sec_input_720p.mp4
├── venv/
```

---

## 📥 Download Model and Videos

- **Download the YOLO model (`best.pt`) from:**  
  [Google Drive - best.pt](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
  
  Place the downloaded `best.pt` file in the `models/` directory.

- **Download the input videos from:**  
  [Google Drive - Videos Folder](https://drive.google.com/drive/folders/1Nx6H_n0UUI6L-6i8WknXd4Cv2c3VjZTP)
  
  Place the required video (e.g., `15sec_input_720p.mp4`) in the `videos/` directory.

---

## ⚙️ Getting Started

### 1. Clone this repo

```bash
git clone https://github.com/KunalBansall/Re-Identification_in_a_Single_Feed
cd player_reid_single_feed
```

### 2. Setup environment

```bash
python -m venv venv
venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Download model and videos
- See the [Download Model and Videos](#download-model-and-videos) section above.

### 4. Run detection + tracking

```bash
python main.py
```

### 📦 Output
- 🎥 Annotated video: `outputs/output.mp4`
- 📄 Tracking CSV: `outputs/tracking_data.csv`
- 🔗 [Download Output Video 1 (Google Drive)](https://drive.google.com/file/d/1rkQKvJI6m0Rvu2uvBweqWN2X8JXVdAAO/view?usp=drive_link)
- 🔗 [Download Output Video 2 (broadway.mp4, Google Drive)](https://drive.google.com/file/d/11rNgXWLi3YRQU7Hdgo6J6EK4IXX0rMCC/view?usp=sharing)
- 📁 [All Output Files (Google Drive Folder)](https://drive.google.com/drive/folders/1Y6Lp6IW_UaisUY5CoYGuxuhgRniZXgfp?usp=sharing)

#### CSV Format:
```
frame,track_id,x1,y1,x2,y2
```

---

## 📄 Project Report
- 📑 [PDF Explanation Project Report (Google Docs)](https://docs.google.com/document/d/1r6xkbLBA1VqJoMnjVwHXuCnWONkwFkh4ySN0dCJg-S4/edit?usp=sharing)

---

## 📉 Limitations
- Only a few players (~3–4) tracked in the demo video
- Misses distant or occluded players due to model limitations
- The `best.pt` model was provided; not retrained for this project

---

## ✅ Conclusion
The system successfully detects and tracks players using a pretrained model, fulfilling the assignment objectives. Output is generated and stored correctly for evaluation.

---

## 🙌 Credits
- YOLOv5 by Ultralytics
- Tracking code inspired by DeepSORT
- Assignment support and model provided by medi@liat.ai