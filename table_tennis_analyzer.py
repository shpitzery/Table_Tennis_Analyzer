import cv2
from ultralytics import YOLO

# Load the YOLOv11n-pose model
model = YOLO('yolo11n-pose.pt')

# Load your image
image_path = 'table-tennis.png'
image = cv2.imread(image_path)

# Inference
results = model(image)

# Get the predictions
for result in results:
    for person in result.boxes:
        # Draw bounding box
        x1, y1, x2, y2 = map(int, person.xyxy[0])  # bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw keypoints
    if result.keypoints is not None:
        for person_keypoints in result.keypoints.xy:
            for keypoint in person_keypoints:
                x, y = keypoint
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

# Save the result image
cv2.imwrite('result-pic.png', image)

# Show the image
# cv2.imshow('Pose Detection', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()