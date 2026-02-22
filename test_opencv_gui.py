import cv2
import numpy as np

# Create a simple test image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(img, 'OpenCV GUI Test', (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.putText(img, 'Press any key to close', (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

try:
    cv2.imshow('Test Window', img)
    print("Window opened successfully! Press any key in the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("OpenCV GUI is working!")
except Exception as e:
    print(f"OpenCV GUI Error: {e}")
    print("GUI support is not available. Use the video saving method instead.")
