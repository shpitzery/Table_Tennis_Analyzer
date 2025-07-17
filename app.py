import cv2
import csv
import numpy as np
from ultralytics import YOLO
import easyocr
import logging
import time

import analysis_utils as utils

# --- SCRIPT START ---
script_start_time = time.time()

# Silence Ultralytics progress lines
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# --- INITIALIZATION ---
print("Initializing EasyOCR... (This may take a moment on first run as models are downloaded)")

reader = easyocr.Reader(['en'])

print("EasyOCR initialized successfully.")

print("Loading YOLO model...")
model = YOLO('yolo11n-pose.pt')
print("YOLO model loaded successfully.")

# Load digit templates
digit_templates = utils.create_digit_templates()

# --- CONFIGURATION ---
START_FRAME = 1000
END_FRAME = 0  # 0 means full video
INPUT_PATH = 'input.mp4'
OUTPUT_PATH = 'output_with_detections.mp4'
POSITIONS_CSV_PATH = 'player_positions.csv'
SCORES_CSV_PATH = 'frame_scores.csv'

# --- VIDEO AND CSV SETUP ---
cap = cv2.VideoCapture(INPUT_PATH)

# 50 fps ==> 10 sec = 500 frames ==> 1 min = 3000 frames
# ==> 84750 frames = 28.25 min for the whole video (84750/3000)
fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Creates a VideoWriter object, which takes frames and writes them into a new video file.
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

with open(POSITIONS_CSV_PATH, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y'])

print(f"Video has {total_frames} frames")
if END_FRAME == 0:
    END_FRAME = total_frames

# --- MAIN PROCESSING LOOP ---
all_x, all_y = [], []
first_frame = None
frame_idx = START_FRAME
last_top_score = 0
last_bottom_score = 0
frame_scores = {}

cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

i = 0
invalid_frames = 0
while cap.isOpened() and frame_idx <= END_FRAME:
    ret, frame = cap.read()
    if not ret:
        break

    # results = model(frame)
    # result = results[0]
    
    # if not utils.is_valid_frame(result, width, height):
    #     invalid_frames += 1
    #     frame_idx += 1
    #     continue


    # --- SCORE READING ---
    if frame_idx % 500 == 0:
        
        try:
            top_games, top_pts, bottom_games, bottom_pts = utils.read_score(frame, reader, digit_templates, frame_idx)
            new_top_score = top_games * 10 + top_pts
            new_bottom_score = bottom_games * 10 + bottom_pts

            if new_top_score >= last_top_score:
                last_top_score = new_top_score
            if new_bottom_score >= last_bottom_score:
                last_bottom_score = new_bottom_score
        except Exception as e:
            print(f"[Frame {frame_idx}] OCR failed: {e}")
        
        frame_scores[frame_idx] = [last_top_score, last_bottom_score]
        
        total_seconds = frame_idx / fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        
        print(f"[Frame {frame_idx}, time {minutes:02d}:{seconds:02d}] Top: {last_top_score} | Bottom: {last_bottom_score}")

    # --- POSE DETECTION ---
    
    # processes the image (which in NumPy array format) and performs object detection.
    results = model(frame)
    
    # extracts the first (and only - passing a single frame to the model) Results object.
    result = results[0]
    
    
    ##########################################
    
    # For debugging
    # if i == 0:
        # print(f"results_len:\n {len(results)}", '\n')
        # print(f"result.boxes:\n {result.boxes}", '\n')
        # print(f"result.keypoints:\n {result.keypoints}", '\n')
        # print(f"result.keypoints.xy:\n {result.keypoints.xy}", '\n')


        # for box in result.boxes:
        #     x1, y1, x2, y2 = map(float, box.xyxy[0])
        #     print(f"Bounding box coordinates:\n x1 = {x1}\n y1 = {y1}\n x2 = {x2}\n y2 = {y2}", '\n')
        
        # print("can q")
        # i = 1

    ##########################################
    
    if not utils.is_valid_frame(result, width, height):
        invalid_frames += 1
        frame_idx += 1
        continue

    if first_frame is None:
        first_frame = frame.copy()

    # --- DRAWING AND DATA EXTRACTION ---
    player_centers = []
    for person in result.boxes:
        x1, y1, x2, y2 = map(int, person.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        player_centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

    # Skeleton drawing
    if result.keypoints is not None:
        for person_keypoints in result.keypoints.xy:
            
            # skeleton according to the standard keypoint indexing that YOLO pose models often follow (e.g., 0: Nose, 1: Left Eye, etc.)
            skeleton = [
                (0, 1, (0, 255, 0)), (1, 2, (0, 255, 0)), (2, 3, (0, 255, 0)), (3, 4, (0, 255, 0)),
                (1, 5, (255, 0, 0)), (5, 7, (255, 0, 0)),
                (2, 6, (255, 0, 0)), (6, 8, (255, 0, 0)),
                (5, 11, (255, 0, 255)), (6, 12, (255, 0, 255)),
                (11, 12, (255, 0, 255)),
                (11, 13, (255, 165, 0)), (13, 15, (255, 165, 0)),
                (12, 14, (255, 165, 0)), (14, 16, (255, 165, 0))
            ]
            for start_idx, end_idx, color in skeleton:
                
                # check if the YOLO model would detect all keypoints (17 (x,y) pairs) accurately.
                if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                    
                    x1, y1 = person_keypoints[start_idx]
                    x2, y2 = person_keypoints[end_idx]
                    if (x1 > 0 and y1 > 0) and (x2 > 0 and y2 > 0):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            for x, y in person_keypoints:
                # make the joints visible by drawing a small red dot at the location of each detected keypoint
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    # keeps only those player centers whose Y-coordinate is in the bottom half of the frame
    player_centers = sorted([pc for pc in player_centers if pc[1] > height * 0.5], key=lambda p: p[0])

    # print(f"\nPlayer centers:\n {player_centers}", '\n')
    
    
    # checks for edge cases or an issue with detection.
    # Player centers example: [(522, 592), (1444, 567)] 
    p1x, p1y = player_centers[0] if len(player_centers) > 0 else ('', '')
    p2x, p2y = player_centers[1] if len(player_centers) > 1 else ('', '')
    
    # append new data to the end of the file without overwriting existing content.
    with open(POSITIONS_CSV_PATH, mode='a', newline='') as csv_file:
        
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([frame_idx, p1x, p1y, p2x, p2y])

    # print(len(player_centers), '\n')
    for cx, cy in player_centers:
        all_x.append(cx)
        all_y.append(cy)

    out.write(frame)
    frame_idx += 1

# print(f"all_x: {all_x}", '\n')
# print(f"all_y: {all_y}", '\n')

# --- POST-PROCESSING AND OUTPUT ---
print(f"\nInvalid frames skipped: {invalid_frames}\n")
print("Processing complete. Generating outputs...")

# Save scores
with open(SCORES_CSV_PATH, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "top_score", "bottom_score"])
    for frame_num, scores in sorted(frame_scores.items()):
        writer.writerow([frame_num, scores[0], scores[1]])

# Generate visuals
utils.generate_heatmap(first_frame, all_x, all_y, width, height)
utils.generate_score_plot(frame_scores)

# --- CLEANUP ---
cap.release()
out.release()
cv2.destroyAllWindows()
print("All tasks finished.")

# --- EXECUTION TIME ---
script_end_time = time.time()
total_seconds = script_end_time - script_start_time
mins = int(total_seconds // 60)
secs = int(total_seconds % 60)
print(f"\nTotal execution time: {mins} minutes and {secs} seconds.")