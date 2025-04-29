import cv2
from ultralytics import YOLO

# Constants
START_FRAME = 0
END_FRAME = 200  # 0 means full video

# Load the YOLOv11n-pose model
model = YOLO('yolo11n-pose.pt')

# Open video
input_path = 'input.mp4'
output_path = 'output_with_detections.mp4'

cap = cv2.VideoCapture(input_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use 'mp4v' for .mp4 output
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# If END_FRAME=0, use full video
if END_FRAME == 0:
    END_FRAME = total_frames

# Start reading frames
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # No more frames

    if frame_idx >= START_FRAME and frame_idx <= END_FRAME:
        results = model(frame)

        # Get the predictions
        for result in results:
            for person in result.boxes:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, person.xyxy[0])  # bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw keypoints
            if result.keypoints is not None:
                for person_keypoints in result.keypoints.xy:
                    # Define skeleton connections: (start_idx, end_idx, color)
                    skeleton = [
                        (0, 1, (0, 255, 0)), (1, 2, (0, 255, 0)), (2, 3, (0, 255, 0)), (3, 4, (0, 255, 0)),  # head
                        (1, 5, (255, 0, 0)), (5, 7, (255, 0, 0)),  # left arm
                        (2, 6, (255, 0, 0)), (6, 8, (255, 0, 0)),  # right arm
                        (5, 11, (255, 0, 255)), (6, 12, (255, 0, 255)),  # shoulders to hips
                        (11, 12, (255, 0, 255)),  # hips
                        (11, 13, (255, 165, 0)), (13, 15, (255, 165, 0)),  # left leg
                        (12, 14, (255, 165, 0)), (14, 16, (255, 165, 0))   # right leg
                    ]

                    # Draw skeleton
                    for start_idx, end_idx, color in skeleton:
                        if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                            x1, y1 = person_keypoints[start_idx]
                            x2, y2 = person_keypoints[end_idx]
                            # Check if both points have valid coordinates
                            if (x1 > 0 and y1 > 0) and (x2 > 0 and y2 > 0):
                                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # Draw joints (optional, small circles)
                    for x, y in person_keypoints:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    if frame_idx > END_FRAME:
        break

    # Write the frame
    out.write(frame)

    frame_idx += 1

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()