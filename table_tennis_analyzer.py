import cv2
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
import easyocr
import os

reader = easyocr.Reader(['en'])  # English OCR

# Silence Ultralytics progress lines
import logging
logging.getLogger('ultralytics').setLevel(logging.WARNING)

# Create template digits function
def create_digit_templates():
    # Only create templates if they don't exist
    if not os.path.exists("templates"):
        os.makedirs("templates")
        print("Created templates directory. Please add template images of each digit.")
        return {}
    
    templates = {}
    for i in range(10):
        template_path = f"templates/{i}.png"
        if os.path.exists(template_path):
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                templates[i] = template
    
    return templates

# Load digit templates
digit_templates = create_digit_templates()

def read_score(frame):
    h, w = frame.shape[:2]
    
    # Define all crops with more precise coordinates
    crops = {
        "p1_games": frame[int(h * 0.865):int(h * 0.90),  int(w * 0.235):int(w * 0.255)],
        "p1_points": frame[int(h * 0.865):int(h * 0.90),  int(w * 0.265):int(w * 0.285)],
        "p2_games": frame[int(h * 0.91):int(h * 0.945),  int(w * 0.235):int(w * 0.255)],
        "p2_points": frame[int(h * 0.91):int(h * 0.945),  int(w * 0.265):int(w * 0.285)],
    }
    
    # Debug: save crops to inspect them visually
    # for name, crop in crops.items():
    #     cv2.imwrite(f"debug_{name}.png", crop)
    
    # Enhanced preprocessing function
    def preprocess_crop(crop, crop_name):
        # Scale up by 4x for even better detail
        resized = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Save grayscale for debugging
        # cv2.imwrite(f"debug_gray_{crop_name}.png", gray)
        
        # 1. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        # 2. Simple binary threshold with multiple values
        _, binary_low = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        _, binary_med = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, binary_high = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # 3. Edge enhancement
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        enhanced = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        _, enhanced_binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        
        # 4. Invert and apply morphological operations (for thin digits like "1")
        # Create both regular and inverted versions
        _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Create a kernel for dilation
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary_inv, kernel, iterations=1)
        
        # Create another version with more aggressive dilation for very thin digits
        thick_kernel = np.ones((3, 3), np.uint8)
        very_dilated = cv2.dilate(binary_inv, thick_kernel, iterations=2)
        
        # Save all versions for debugging
        # cv2.imwrite(f"debug_adaptive_{crop_name}.png", adaptive)
        # cv2.imwrite(f"debug_binary_low_{crop_name}.png", binary_low)
        # cv2.imwrite(f"debug_binary_med_{crop_name}.png", binary_med)
        # cv2.imwrite(f"debug_binary_high_{crop_name}.png", binary_high)
        # cv2.imwrite(f"debug_enhanced_{crop_name}.png", enhanced_binary)
        # cv2.imwrite(f"debug_inverted_{crop_name}.png", binary_inv)
        # cv2.imwrite(f"debug_dilated_{crop_name}.png", dilated)
        # cv2.imwrite(f"debug_very_dilated_{crop_name}.png", very_dilated)
        
        # Return multiple versions to try with OCR
        return [adaptive, binary_low, binary_med, binary_high, enhanced_binary, binary_inv, dilated, very_dilated]
    
    # Improved OCR function with confidence
    def get_digit(crop, crop_name):
        processed_versions = preprocess_crop(crop, crop_name)
        best_digit = 0
        best_confidence = 0
        best_method = "none"
        
        # Try OCR on all preprocessing versions with EasyOCR
        for i, proc in enumerate(processed_versions):
            # Try with different OCR configurations
            text = reader.readtext(proc, detail=1, 
                                  allowlist='0123456789',
                                  paragraph=False,
                                  width_ths=0.7,
                                  height_ths=0.7)
            
            # Log what was found for debugging
            # print(f"OCR results for {crop_name} (version {i}): {text}")
            
            if text:
                # Find the result with highest confidence
                for t in text:
                    bbox, result_text, confidence = t
                    if result_text.isdigit() and confidence > best_confidence:
                        best_digit = int(result_text)
                        best_confidence = confidence
                        best_method = f"easyocr_v{i}"
        
        # If OCR methods failed or had low confidence, try template matching
        if best_confidence < 0.4 and digit_templates:
            for i, proc in enumerate(processed_versions):
                best_match_val = 0
                best_match_digit = None
                
                # Try each template
                for digit, template in digit_templates.items():
                    # Resize template to match the expected digit size
                    h, w = proc.shape[:2]
                    resized_template = cv2.resize(template, (w, h))
                    
                    # Perform template matching
                    result = cv2.matchTemplate(proc, resized_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    # print(f"Template match for {crop_name} digit {digit}: {max_val:.3f}")
                    
                    if max_val > best_match_val:
                        best_match_val = max_val
                        best_match_digit = digit
                
                # If we found a good match
                if best_match_val > 0.6 and best_match_val > best_confidence:
                    best_digit = best_match_digit
                    best_confidence = best_match_val
                    best_method = f"template_v{i}"
        
        # print(f"Best digit for {crop_name}: {best_digit} (confidence: {best_confidence:.2f}, method: {best_method})")
        return best_digit
    
    # Extract all scores with improved debugging
    top_games = get_digit(crops["p1_games"], "p1_games")
    top_points = get_digit(crops["p1_points"], "p1_points")
    bottom_games = get_digit(crops["p2_games"], "p2_games")
    bottom_points= get_digit(crops["p2_points"], "p2_points")
    
    return top_games, top_points, bottom_games, bottom_points

# Rest of your script remains the same...
# Colormap and heat‑map helper
colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
          (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
HEATMAP_CM = LinearSegmentedColormap.from_list('player_heat', colors)

def is_valid_frame(result, frame_w, frame_h):
    frame_area = frame_w * frame_h
    candidate_boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if cy > frame_h * 0.35:                
            if ((x2 - x1) * (y2 - y1)) / frame_area <= 0.25:
                candidate_boxes.append(box)
    return len(candidate_boxes) >= 2

# Constants
START_FRAME = 0
END_FRAME = 20000  # 0 means full video

# Load the YOLOv11n-pose model
model = YOLO('yolo11n-pose.pt')

# Open video
input_path = 'input.mp4'
output_path = 'output_with_detections.mp4'

csv_file = open('player_positions.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'player1_position_x', 'player1_position_y', 'player2_position_x', 'player2_position_y'])

cap = cv2.VideoCapture(input_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video has {total_frames} frames")

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'mp4v' for .mp4 output
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# If END_FRAME=0, use full video
if END_FRAME == 0:
    END_FRAME = total_frames

# Inform the user how long this run will be
frames_to_process = END_FRAME - START_FRAME + 1
run_seconds = frames_to_process / fps if fps else 0
# Pretty duration "m:ss seconds"
total_sec = int(round(run_seconds))
mins, secs = divmod(total_sec, 60)
duration_readable = f"{mins}:{secs:02d} seconds"

print(
    f"Run for {frames_to_process:,} frames "
    f"({duration_readable} of video at {fps:.2f} fps)."
)

all_x, all_y = [], []
first_frame = None  # We will capture the first valid rally frame for the background

cap = cv2.VideoCapture(input_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)  # Jump directly to START_FRAME

frame_idx = START_FRAME
top_score = 0
bottom_score = 0
frame_scores = {}
last_top_score = 0
last_bottom_score = 0


while cap.isOpened():
    ret, frame = cap.read() # Read a frame. A frame is a numpy array.
    # Stop once we've passed END_FRAME
    if frame_idx > END_FRAME:
        break
    if not ret:
        break  # No more frames

    if frame_idx % 500 == 0:
        try:
            top_games, top_pts, bottom_games, bottom_pts = read_score(frame)

            # Calculate new scores
            new_top_score = top_games * 10 + top_pts
            new_bottom_score = bottom_games * 10 + bottom_pts

            # Only update if OCR result makes sense (score doesn’t decrease)
            if new_top_score >= last_top_score:
                last_top_score = new_top_score
            else:
                print(f"[Frame {frame_idx}] ⚠️ Ignoring top score {new_top_score}, lower than last ({last_top_score})")

            if new_bottom_score >= last_bottom_score:
                last_bottom_score = new_bottom_score
            else:
                print(f"[Frame {frame_idx}] ⚠️ Ignoring bottom score {new_bottom_score}, lower than last ({last_bottom_score})")

        except Exception as e:
            print(f"[Frame {frame_idx}] ⚠️ OCR failed: {e}")

        # Save (last known good) scores for the current frame
        frame_scores[frame_idx] = [last_top_score, last_bottom_score]

        print(f"[Frame {frame_idx}] Top player ➜ {last_top_score} | Bottom player ➜ {last_bottom_score}")

        # print(f"[Frame {frame_idx}] Top player: gamesWon={top_games}, pts={top_pts} ➜ {top_score} | "
        #       f"Bottom player: gamesWon={bottom_games}, pts={bottom_pts} ➜ {bottom_score}")

    results = model(frame) 
    result = results[0] 

    # --------  FRAME QUALITY FILTER  --------
    if not is_valid_frame(result, width, height):
        frame_idx += 1
        continue

    # Save a clean copy of the very first VALID frame for the heat‑map background
    if first_frame is None:
        first_frame = frame.copy()

    # Draw detections and skeletons
    for person in result.boxes:
        x1, y1, x2, y2 = map(int, person.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if result.keypoints is not None:
        for person_keypoints in result.keypoints.xy:
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
                if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                    x1, y1 = person_keypoints[start_idx]
                    x2, y2 = person_keypoints[end_idx]
                    if (x1 > 0 and y1 > 0) and (x2 > 0 and y2 > 0):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            for x, y in person_keypoints:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    # --------  CSV & ACCUMULATORS  --------
    player_centers = []
    for person in result.boxes:
        x1, y1, x2, y2 = map(int, person.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        player_centers.append((cx, cy))

    # keep only centres in lower half (exclude referee)
    player_centers = [(cx, cy) for (cx, cy) in player_centers if cy > height * 0.50]

    player_centers.sort(key=lambda p: p[0])  # left‑to‑right
    p1x, p1y = player_centers[0] if len(player_centers) > 0 else ('', '')
    p2x, p2y = player_centers[1] if len(player_centers) > 1 else ('', '')
    csv_writer.writerow([frame_idx, p1x, p1y, p2x, p2y])

    # store positions for heat‑map
    for cx, cy in player_centers:
        all_x.append(cx)
        all_y.append(cy)

    # --------  OUTPUT VIDEO  --------
    out.write(frame)
    frame_idx += 1

# Save frame-wise scores
with open("frame_scores.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "top_score", "bottom_score"])
    for frame, (top, bottom) in sorted(frame_scores.items()):
        writer.writerow([frame, top, bottom])

# ----------  HEAT‑MAP VISUALISATION  ----------
if first_frame is not None and all_x:
    xs = np.array(all_x)
    ys = np.array(all_y)

    # use ~50 px cells and milder blur
    bin_size = 50
    heatmap, _, _ = np.histogram2d(
        ys, xs,
        bins=[height // bin_size, width  // bin_size],
        range=[[0, height], [0, width]]
    )
    heatmap = gaussian_filter(heatmap, sigma=2)
    alpha_mask = np.clip(heatmap / heatmap.max(), 0, 1)

    # Normalise for alpha masking
    #heatmap_norm = heatmap / (heatmap.max() if heatmap.max() else 1.0)

    plt.figure(figsize=(width / 100, height / 100), dpi=100)

    # Show the background frame exactly as‑is
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))

    # Overlay the heat‑map: pixels with zero density are fully transparent;
    # alpha increases with density.
    im = plt.imshow(
        heatmap,
        cmap=HEATMAP_CM,
        alpha=alpha_mask,
        extent=[0, width, height, 0],
        interpolation='nearest'
    )

    # Side color‑bar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Density')

    plt.title('Combined Player Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')

    plt.savefig('player_position_heatmap.png', bbox_inches='tight')
    plt.close()


# ----------  PLOT: Calculated Score Over Time  ----------
frames = sorted(frame_scores.keys())
top_scores = [frame_scores[f][0] for f in frames]
bottom_scores = [frame_scores[f][1] for f in frames]

plt.figure(figsize=(12, 8))
plt.plot(frames, top_scores, label="Top Player", marker="o")
plt.plot(frames, bottom_scores, label="Bottom Player", marker="o")
plt.xlabel("Frame")
plt.ylabel("Score")
plt.title("Calculated Score Over Time")
plt.yticks(range(0, 32))  # Y axis: 0 to 31
plt.xticks(frames)         # Show only sampled frame points (e.g., every 500)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("calculatedScore.png")
plt.close()

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
csv_file.close()