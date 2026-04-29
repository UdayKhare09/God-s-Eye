import cv2
import sys

url = sys.argv[1]
print(f"Trying to open {url}...")
cap = cv2.VideoCapture(url)
if cap.isOpened():
    print("Success!")
    cap.release()
else:
    print("Failed!")
