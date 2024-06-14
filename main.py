import cv2
import numpy as np

# Define the HSV range for the red color
lower_red = np.array([0, 0, 230])
upper_red = np.array([0, 0, 255])
morph_size = 5
element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * morph_size + 1, 2 * morph_size + 1),
                                    (morph_size, morph_size))
operation = cv2.MORPH_OPEN


def create_red_dot_mask(frame):
    mask = cv2.inRange(frame, lower_red, upper_red)

    # dst = cv2.morphologyEx(mask, operation, element)
    dilatation_dst = cv2.dilate(mask, element)
    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(dilatation_dst, (15, 15), 0)

    _, thresh = cv2.threshold(blurred_mask, 25, 255, cv2.THRESH_BINARY)
    return thresh


# Function to detect and localize the red blinking dot
def detect_red_blinking_dot(frame, prev_frame):
    # Create a mask for red color
    mask = cv2.inRange(frame, lower_red, upper_red)
    morph_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * morph_size + 1, 2 * morph_size + 1),
                                        (morph_size, morph_size))
    operation = cv2.MORPH_OPEN
    # dst = cv2.morphologyEx(mask, operation, element)
    dilatation_dst = cv2.dilate(mask, element)
    # Apply Gaussian blur
    blurred_mask = cv2.GaussianBlur(dilatation_dst, (15, 15), 0)

    # Compare current frame with previous frame for blinking detection
    if prev_frame is not None:

        _, thresh = cv2.threshold(blurred_mask, 25, 255, cv2.THRESH_BINARY)
    else:
        thresh = blurred_mask

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over contours to find the largest one
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 100:  # Adjust threshold value as needed
            # Get the coordinates of the bounding box
            x, y, w, h = cv2.boundingRect(max_contour)
            return x, y, w, h, thresh

    return None, None, None, None, thresh


def main(cap):
    prev_frame = None
    # Create a named window that is resizable
    cv2.namedWindow('CCTV Feed', cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # x, y, w, h, prev_frame = detect_red_blinking_dot(frame, prev_frame)
        mask = create_red_dot_mask(frame)

        # if x is not None and y is not None:
        #     # Draw a rectangle around the detected red dot
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Inpaint the frame using the mask
        inpainted_frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        # Display the frame
        cv2.imshow('CCTV Feed', inpainted_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = r"C:\Users\Mantra\Downloads\APRIL Technical Assessment.mp4"
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)  # or use 0 for webcam

    main(cap)
