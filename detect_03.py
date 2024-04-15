import cv2
import numpy as np
import argparse

# Parse argument for image path
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Đường dẫn đến ảnh cần phát hiện đối tượng")
args = vars(ap.parse_args())

# Load the image
image = cv2.imread(args["image"])
if image is None:
    print("Không thể đọc ảnh")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce image noise if it is required
gray = cv2.GaussianBlur(gray, (7, 7), 0)  # Làm mịn hơn
_, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)  # Áp dụng threshold

# Detect edges using Canny
edged = cv2.Canny(gray, 70, 200)
edged = cv2.dilate(edged, None, iterations=1)
cv2.imshow('Canny Edges', edged)
cv2.waitKey(0)
# Find contours
contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around detected objects
for contour in contours:
    # Calculate the bounding rectangle for each contour
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Objects', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

