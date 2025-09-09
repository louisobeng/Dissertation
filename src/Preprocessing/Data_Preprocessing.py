import os
import shutil
import cv2
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
from datetime import datetime
import random
import numpy as np
import gc
import wandb

#  CONFIG 
GENDER_CONF_THRESHOLD = 0.90
EMOTION_CONF_THRESHOLD = 0.90
LIVE_STATS_INTERVAL = 100
random.seed(42)

# INPUT & OUTPUT PATHS 
input_root = "/Users/mct/Documents/GitHub/dissertation/CK+/dataset"
output_base_dir = "/Users/mct/Documents/GitHub/dissertation/CK+/processed"

split_dirs = {
    "train": os.path.join(output_base_dir, "train"),
    "val": os.path.join(output_base_dir, "val"),
    "test": os.path.join(output_base_dir, "test")
}
emotion_dir = os.path.join(output_base_dir, "emotions")
failed_dir = os.path.join(output_base_dir, "Failed")
log_file = os.path.join(output_base_dir, "processed_files.log")
partial_csv = os.path.join(output_base_dir, "partial_results.csv")

#  SPLIT RATIOS 
split_ratios = {"train": 0.8, "val": 0.15, "test": 0.05}

#  CIRCUMPLEX VA MAPPING 
circumplex_va = {
    "angry": (-0.7, 0.8),
    "disgust": (-0.6, 0.5),
    "fear": (-0.7, 0.9),
    "happy": (0.8, 0.6),
    "sad": (-0.8, 0.3),
    "surprise": (0.4, 0.9),
    "neutral": (0.0, 0.0)
}

#  HELPERS 
def compute_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[..., 2].mean()

def compute_sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_face_geometry(region):
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    x, y = max(x, 0), max(y, 0)
    area = w * h
    aspect_ratio = round(w / h, 2) if h > 0 else 0
    return x, y, w, h, area, aspect_ratio

def save_failure_entry(filename, reason, image_path=None, split="unknown", emotion="unknown"):
    if image_path:
        image = cv2.imread(image_path)
        if image is not None:
            os.makedirs(failed_dir, exist_ok=True)
            cv2.imwrite(os.path.join(failed_dir, filename), image)

    return {
        "filename": filename,
        "dataset": split,
        "true_emotion": emotion,
        "gender_pred": reason,
        "gender": reason,
        "dominant_emotion": reason,
        "gender_emotion_class": reason,
        **{emo: None for emo in circumplex_va},
        "face_x": None,
        "face_y": None,
        "face_w": None,
        "face_h": None,
        "face_area": None,
        "face_aspect_ratio": None,
        "brightness": None,
        "sharpness": None,
        "valence": None,
        "arousal": None
    }

def process_image(filepath, split, emotion):
    filename = os.path.basename(filepath)
    print(f"🔍 Processing: {filename}")
    try:
        analysis = DeepFace.analyze(
            img_path=filepath,
            actions=["emotion", "gender"],
            detector_backend="mtcnn",
            enforce_detection=True
        )[0]

        dominant_emotion = analysis["dominant_emotion"]
        emotion_conf = analysis["emotion"][dominant_emotion]
        gender = analysis["dominant_gender"]
        gender_conf = analysis["gender"][gender]

        if gender_conf < GENDER_CONF_THRESHOLD or emotion_conf < EMOTION_CONF_THRESHOLD:
            return save_failure_entry(filename, "low_conf", filepath, split, emotion)

        region = analysis["region"]
        x, y, w, h, area, aspect_ratio = extract_face_geometry(region)

        image = cv2.imread(filepath)
        face_crop = image[y:y+h, x:x+w]
        if face_crop.size == 0:
            return save_failure_entry(filename, "empty_crop", filepath, split, emotion)

        # Save cropped face
        dest_dir = os.path.join(split_dirs[split])
        os.makedirs(dest_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dest_dir, f"face_{filename}"), face_crop)

        # Save to gender × emotion folder
        gender_dir = os.path.join(emotion_dir, dominant_emotion, gender)
        os.makedirs(gender_dir, exist_ok=True)
        cv2.imwrite(os.path.join(gender_dir, filename), face_crop)

        brightness = compute_brightness(face_crop)
        sharpness = compute_sharpness(face_crop)

        emotion_probs = analysis["emotion"]
        valence = arousal = total_prob = 0.0
        for emo, prob in emotion_probs.items():
            if emo in circumplex_va:
                v, a = circumplex_va[emo]
                valence += v * prob
                arousal += a * prob
                total_prob += prob
        if total_prob > 0:
            valence /= total_prob
            arousal /= total_prob

        return {
            "filename": filename,
            "dataset": split,
            "true_emotion": emotion,
            "gender_pred": str(analysis["gender"]),
            "gender": gender,
            "dominant_emotion": dominant_emotion,
            "gender_emotion_class": f"{dominant_emotion}_{gender}",
            **{emo: emotion_probs.get(emo, 0) for emo in circumplex_va},
            "face_x": x,
            "face_y": y,
            "face_w": w,
            "face_h": h,
            "face_area": area,
            "face_aspect_ratio": aspect_ratio,
            "brightness": brightness,
            "sharpness": sharpness,
            "valence": valence,
            "arousal": arousal
        }

    except Exception as e:
        print(f"Failed: {filename} → {type(e).__name__}: {e}")
        return save_failure_entry(filename, "error", filepath, split, emotion)

def main():
    wandb.init(
        project="ckplus-processing",
        name="ckplus-run",
        config={
            "gender_conf_threshold": GENDER_CONF_THRESHOLD,
            "emotion_conf_threshold": EMOTION_CONF_THRESHOLD,
            "live_stats_interval": LIVE_STATS_INTERVAL
        }
    )

    # COLLECT AND SPLIT ALL FILES 
    all_image_tuples = []
    for emotion in os.listdir(input_root):
        emo_dir = os.path.join(input_root, emotion)
        if not os.path.isdir(emo_dir):
            continue
        files = [os.path.join(emo_dir, f) for f in os.listdir(emo_dir) if f.lower().endswith((".jpg", ".png"))]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * split_ratios["train"])
        n_val = int(n * split_ratios["val"])
        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]
        all_image_tuples += [(f, "train", emotion) for f in train_files]
        all_image_tuples += [(f, "val", emotion) for f in val_files]
        all_image_tuples += [(f, "test", emotion) for f in test_files]

    print(f" Total images to process: {len(all_image_tuples)}")

    processed_files = set()
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            processed_files = set(line.strip() for line in f)

    counter = success_count = fail_count = 0
    emotion_counter = {emo: 0 for emo in circumplex_va}

    for path, split, emotion in tqdm(all_image_tuples):
        filename = os.path.basename(path)
        if filename in processed_files:
            continue

        result = process_image(path, split, emotion)

        with open(log_file, "a") as f:
            f.write(filename + "\n")

        if result:
            header_needed = not os.path.exists(partial_csv)
            pd.DataFrame([result]).to_csv(partial_csv, mode="a", header=header_needed, index=False)

            if result["dataset"] != "failed":
                success_count += 1
                if result["dominant_emotion"] in emotion_counter:
                    emotion_counter[result["dominant_emotion"]] += 1
            else:
                fail_count += 1

        counter += 1
        if counter % LIVE_STATS_INTERVAL == 0:
            wandb.log({
                "processed_images": counter,
                "success_count": success_count,
                "fail_count": fail_count,
                **emotion_counter
            })

        gc.collect()

    #  FINAL CSV EXPORT 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_csv = os.path.join(output_base_dir, f"CKPLUS_Features_{timestamp}.csv")
    if os.path.exists(partial_csv):
        df = pd.read_csv(partial_csv)
        df.to_csv(final_csv, index=False)
        print(f" Final CSV saved to: {final_csv}")
    else:
        print("No data saved.")

    wandb.finish()

if __name__ == "__main__":
    main()