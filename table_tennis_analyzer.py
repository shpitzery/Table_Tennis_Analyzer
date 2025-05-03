import cv2
import csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

# Colormap and heat‑map helper
colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
          (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
HEATMAP_CM = LinearSegmentedColormap.from_list('player_heat', colors)

def is_valid_frame(result, frame_w, frame_h):
    """
    Accepts frames where *at least* the two athletes at the bottom half are visible.
    Referee (usually higher in the frame) is ignored.

    Heuristics
    ----------
    1. Keep only detections whose vertical centre (cy) is in the lower half of the frame.
    2. Among those, require at least 2 boxes (the two players).
    3. Each player's box should not occupy >20 % of the frame area (filters out zoom‑ins).

    Returns
    -------
    bool
    """
    frame_area = frame_w * frame_h
    candidate_boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if cy > frame_h * 0.35:                  # lower half → likely a player
            if ((x2 - x1) * (y2 - y1)) / frame_area <= 0.25:
                candidate_boxes.append(box)
    return len(candidate_boxes) >= 2

# Constants
START_FRAME = 0
END_FRAME = 10000  # 0 means full video

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
    f"({duration_readable} of video at {fps:.2f} fps)."
)

all_x, all_y = [], []
first_frame = None  # We will capture the first valid rally frame for the background

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read() # Read a frame. A frame is a numpy array.
    # Stop once we've passed END_FRAME
    if frame_idx > END_FRAME:
        break
    if not ret:
        break  # No more frames

    if frame_idx < START_FRAME or frame_idx > END_FRAME:
        frame_idx += 1
        continue

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

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
csv_file.close()