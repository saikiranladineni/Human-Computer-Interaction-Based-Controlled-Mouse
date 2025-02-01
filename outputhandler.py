import cv2
import numpy as np

def draw_preview(frame, inference_results, preview_flags):
    if not preview_flags:
        return frame

    preview_frame = frame.copy()
    
    # Face Detection
    if 'ff' in preview_flags:
        x_min, y_min, x_max, y_max = inference_results["face_coords"][0]
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Landmark Detection
    if 'fl' in preview_flags:
        for eye in inference_results["eye_coords"]:
            x_min, y_min, x_max, y_max = eye
            cv2.rectangle(preview_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    return preview_frame

def save_video(output_path, frame):
    cv2.imwrite(output_path, frame)
