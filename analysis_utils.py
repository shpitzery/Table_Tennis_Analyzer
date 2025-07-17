import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# Colormap and heat-map helper
colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
          (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
HEATMAP_CM = LinearSegmentedColormap.from_list('player_heat', colors)

# --- CONSTANTS ---
# Defines the bounding box for the referee's expected position.
# Used for both validating frames and drawing debug rectangles.
REFEREE_BOX = {
    'x1': 870, 'y1': 80, 
    'x2': 1070, 'y2': 380
}

def create_digit_templates():
    """Creates and loads digit templates from the 'templates' directory."""
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


def read_score(frame, reader, digit_templates, frame_idx):
    """
    Reads the score from a given frame using OCR and template matching.
    
    Args:
        frame: The video frame (NumPy array).
        reader: The initialized easyocr.Reader instance.
        digit_templates: A dictionary of loaded digit templates.
        
    Returns:
        A tuple of four integers: (top_games, top_points, bottom_games, bottom_points).
    """
    h, w = frame.shape[:2]
    
    crops = {
        "p1_games": frame[int(h * 0.865):int(h * 0.90),  int(w * 0.235):int(w * 0.255)],
        "p1_points": frame[int(h * 0.865):int(h * 0.90),  int(w * 0.265):int(w * 0.285)],
        "p2_games": frame[int(h * 0.91):int(h * 0.945),  int(w * 0.235):int(w * 0.255)],
        "p2_points": frame[int(h * 0.91):int(h * 0.945),  int(w * 0.265):int(w * 0.285)],
    }

    # --- Debugging block to visualize crop locations ---
    debug_dir = "debug_rects"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)

    debug_frame = frame.copy()
    # Player 1 boxes (Green)
    cv2.rectangle(debug_frame, (int(w * 0.235), int(h * 0.865)), (int(w * 0.255), int(h * 0.90)), (0, 255, 0), 2)
    cv2.rectangle(debug_frame, (int(w * 0.265), int(h * 0.865)), (int(w * 0.285), int(h * 0.90)), (0, 255, 0), 2)
    # Player 2 boxes (Red)
    cv2.rectangle(debug_frame, (int(w * 0.235), int(h * 0.91)), (int(w * 0.255), int(h * 0.945)), (0, 0, 255), 2)
    cv2.rectangle(debug_frame, (int(w * 0.265), int(h * 0.91)), (int(w * 0.285), int(h * 0.945)), (0, 0, 255), 2)
    
    # Referee box (Blue)
    cv2.rectangle(debug_frame, (REFEREE_BOX['x1'], REFEREE_BOX['y1']), (REFEREE_BOX['x2'], REFEREE_BOX['y2']), (255, 0, 0), 2)
    
    cv2.imwrite(os.path.join(debug_dir, f"debug_rect_{frame_idx}.png"), debug_frame)
    # cv2.imwrite("debug_crop_locations.png", debug_frame)

    # debug_dir = "debug_images"
    # if not os.path.exists(debug_dir):
    #     os.makedirs(debug_dir)
    # for name, crop_img in crops.items():
    #     if crop_img.size > 0:
    #         cv2.imwrite(os.path.join(debug_dir, f"crop_{name}.png"), crop_img)
    
    # --- End Debugging block ---
    
    def preprocess_crop(crop, crop_name):
        
        if crop.size == 0:
            return [] # Return empty list if crop is empty
        
        resized = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        _, binary_low = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        _, binary_med = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        _, binary_high = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)
        enhanced = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        _, enhanced_binary = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
        _, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2, 2), np.uint8)
        
        dilated = cv2.dilate(binary_inv, kernel, iterations=1)
        thick_kernel = np.ones((3, 3), np.uint8)
        very_dilated = cv2.dilate(binary_inv, thick_kernel, iterations=2)
        
        return [adaptive, binary_low, binary_med, binary_high, enhanced_binary, binary_inv, dilated, very_dilated]


    def get_digit(crop, crop_name):
        processed_versions = preprocess_crop(crop, crop_name)
        if not processed_versions:
            return 0 # Return default if crop was empty
        best_digit = 0
        best_confidence = 0
        
        for i, proc in enumerate(processed_versions):
            text = reader.readtext(proc, detail=1, allowlist='0123456789', paragraph=False, width_ths=0.7, height_ths=0.7)
            
            if text:
                for _, result_text, confidence in text:
                    if result_text.isdigit() and confidence > best_confidence:
                        best_digit = int(result_text)
                        best_confidence = confidence
        
        
        if best_confidence < 0.4 and digit_templates:
            for i, proc in enumerate(processed_versions):
                best_match_val = 0
                best_match_digit = None
                for digit, template in digit_templates.items():
                    h, w = proc.shape[:2]
                    if h > 0 and w > 0:
                        resized_template = cv2.resize(template, (w, h))
                        result = cv2.matchTemplate(proc, resized_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        if max_val > best_match_val:
                            best_match_val = max_val
                            best_match_digit = digit
                if best_match_val > 0.6 and best_match_val > best_confidence:
                    best_digit = best_match_digit
                    best_confidence = best_match_val
        return best_digit


    top_games = get_digit(crops["p1_games"], "p1_games")
    top_points = get_digit(crops["p1_points"], "p1_points")
    bottom_games = get_digit(crops["p2_games"], "p2_games")
    bottom_points= get_digit(crops["p2_points"], "p2_points")
    
    return top_games, top_points, bottom_games, bottom_points


def is_valid_frame(result, frame_w, frame_h):
    # frame_area = frame_w * frame_h
    # candidate_boxes = []
    # for box in result.boxes:
    #     x1, y1, x2, y2 = map(float, box.xyxy[0])
    #     cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    #     if cy > frame_h * 0.35:                
    #         if ((x2 - x1) * (y2 - y1)) / frame_area <= 0.25:
    #             candidate_boxes.append(box)
    # return len(candidate_boxes) >= 2
    
    
    """
    Checks if a frame is valid by ensuring it contains at least two players
    and a referee in a predefined location.
    """
    referee_box = REFEREE_BOX

    player_candidate_boxes = []
    referee_found = False
    frame_area = frame_w * frame_h

    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # Check 1: Is the center of this detected person inside the referee_box?
        if (referee_box['x1'] < cx < referee_box['x2']) and (referee_box['y1'] < cy < referee_box['y2']):
            referee_found = True
            # This box is the referee, no need to check if it's a player.
            # We can skip to the next detected box.
            continue

        # Check 2: If not the referee, is it a potential player?
        # Checks if the person is in the bottom 65% of the frame.
        if cy > frame_h * 0.35:
            # Checks if the person's bounding box is not excessively large (<= 25% of frame area).
            if ((x2 - x1) * (y2 - y1)) / frame_area <= 0.25:
                player_candidate_boxes.append(box)

    # A frame is valid if we found a referee AND at least two players.
    return referee_found and len(player_candidate_boxes) >= 2


def generate_heatmap(first_frame, all_x, all_y, width, height):
    """Generates and saves a player position heatmap."""
    if first_frame is None or not all_x:
        print("Skipping heatmap generation: no valid frames or player positions found.")
        return

    xs = np.array(all_x)
    ys = np.array(all_y)
    bin_size = 30
    
    heatmap, _, _ = np.histogram2d(ys, xs, bins=[height // bin_size, width // bin_size], range=[[0, height], [0, width]])
    
    heatmap = gaussian_filter(heatmap, sigma=2)
    alpha_mask = np.clip(heatmap / heatmap.max(), 0, 1)
    

    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    im = plt.imshow(heatmap, cmap=HEATMAP_CM, alpha=alpha_mask, extent=[0, width, height, 0], interpolation='nearest')
    
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Density')
    plt.title('Combined Player Position Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.savefig('player_position_heatmap.png', bbox_inches='tight')
    plt.close()
    print("Generated player position heatmap.")


def generate_score_plot(frame_scores):
    """Generates and saves a plot of the scores over time."""
    if not frame_scores:
        print("Skipping score plot generation: no scores were recorded.")
        return

    # Filter to get only the frames where the score actually changed.
    change_points = {}
    last_top_score = -1
    last_bottom_score = -1
    all_frames = sorted(frame_scores.keys())

    for frame in all_frames:
        top_score, bottom_score = frame_scores[frame]
        if top_score != last_top_score or bottom_score != last_bottom_score:
            change_points[frame] = [top_score, bottom_score]
            last_top_score = top_score
            last_bottom_score = bottom_score
            
    # Ensure the plot line extends to the last frame checked.
    if all_frames and all_frames[-1] not in change_points:
        last_frame = all_frames[-1]
        change_points[last_frame] = frame_scores[last_frame]

    # Create the lists for plotting from the filtered data
    frames = sorted(change_points.keys())
    top_scores = [change_points[f][0] for f in frames]
    bottom_scores = [change_points[f][1] for f in frames]

    plt.figure(figsize=(12, 8))
    plt.plot(frames, top_scores, label="Top Player", marker="o", drawstyle="steps-post")
    plt.plot(frames, bottom_scores, label="Bottom Player", marker="o", drawstyle="steps-post")
    plt.xlabel("Frame")
    plt.ylabel("Score")
    plt.title("Calculated Score Over Time")
    plt.yticks(range(0, max(max(top_scores, default=0), max(bottom_scores, default=0)) + 2))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("calculatedScore.png")
    plt.close()
    print("Generated score plot.")
