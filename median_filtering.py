import cv2
import numpy as np
from collections import deque


class VideoRestorer:
    def __init__(self, video_path, buffer_size=30, lower_red=np.array([0, 0, 180]),
                 upper_red=np.array([40, 40, 255])):
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.lower_red = lower_red
        self.upper_red = upper_red
        self.frame_buffer = deque(maxlen=buffer_size)

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        morph_size = 9
        self.element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * morph_size + 1, 2 * morph_size + 1),
                                                 (morph_size, morph_size))
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('restored_video.mp4', fourcc, self.fps, (self.frame_width, self.frame_height))

    def add_gaussian_blur(self, frame, kernel_size):
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

    def add_salt_and_pepper_noise(self, frame, amount=0.004):
        row, col, ch = frame.shape
        s_vs_p = 0.5
        out = np.copy(frame)

        # Salt mode
        num_salt = np.ceil(amount * frame.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in frame.shape]
        out[coords[0], coords[1], :] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * frame.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in frame.shape]
        out[coords[0], coords[1], :] = 0
        return out

    def add_gaussian_noise(self, frame):
        row, col, ch = frame.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = frame + gauss * 255
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def add_poisson_noise(self, frame):
        vals = len(np.unique(frame))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(frame * vals) / float(vals)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def adjust_illumination(self, frame, alpha=1.5, beta=0):
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    def create_red_dot_mask(self, frame):

        mask = cv2.inRange(frame, self.lower_red, self.upper_red)


        dilatation_dst = cv2.dilate(mask, self.element)
        # Apply Gaussian blur
        blurred_mask = cv2.GaussianBlur(dilatation_dst, (15, 15), 0)

        _, thresh = cv2.threshold(blurred_mask, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over contours to find the largest one
        if contours:
            max_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(max_contour) > 100:
                mask = np.zeros_like(mask)
                cv2.drawContours(mask, [max_contour], -1, 255, thickness=cv2.FILLED)
                return mask, True

        return thresh, False

    def inpaint_frame(self, frame, mask):
        return cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    def process_video(self):
        cv2.namedWindow('Restored CCTV Feed', cv2.WINDOW_NORMAL)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # frame = self.add_gaussian_blur(frame, kernel_size=9)
            # frame = self.add_salt_and_pepper_noise(frame)
            # frame = self.adjust_illumination(frame, alpha=0.7, beta=30)
            # Create mask for the red dot
            mask, valid_mask = self.create_red_dot_mask(frame)

            if valid_mask:
                # Inpaint the frame using the mask
                inpainted_frame = self.inpaint_frame(frame, mask)
            else:
                half_frame = cv2.resize(frame, (self.frame_width // 2, self.frame_height // 2),
                                        interpolation=cv2.INTER_CUBIC)
                self.frame_buffer.append(half_frame)

            # Ensure the buffer is full before processing
            if len(self.frame_buffer) == self.buffer_size:

                combined_frame = frame.copy()
                if valid_mask:
                    # Compute the median frame
                    median_frame = np.median(np.array(self.frame_buffer), axis=0).astype(dtype=np.uint8)
                    full_frame = cv2.resize(median_frame, (self.frame_width, self.frame_height),
                                            interpolation=cv2.INTER_CUBIC)
                    # Combine inpainted and median frames
                    combined_frame[mask > 0] = cv2.addWeighted(inpainted_frame[mask > 0], 0.2, full_frame[mask > 0], 0.8
                                                               , 0)

                # Write the restored frame to the output video
                self.out.write(combined_frame)

                # Display the restored frame
                cv2.imshow('Restored CCTV Feed', combined_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release video capture and writer, and close windows
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()


# Main function
def main():
    video_path = "cctv_feed.mp4"
    restorer = VideoRestorer(video_path)
    restorer.process_video()


if __name__ == "__main__":
    main()
