import time
import logging
import cv2
import os
import numpy as np
from args_parser import build_argparser
from model_loader import load_models
from input_handler import get_input_source
from inference import run_inference
from mouse_controller import move_mouse
from output_handler import draw_preview, save_video
from mouse_controller import MouseController

def main():
    args = build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO)

    model_paths = {
        'FaceDetectionModel': args.faceDetectionModel,
        'LandmarkRegressionModel': args.landmarkRegressionModel,
        'HeadPoseEstimationModel': args.headPoseEstimationModel,
        'GazeEstimationModel': args.gazeEstimationModel,
    }

    models = load_models(model_paths, args.device, args.prob_threshold)
    feeder = get_input_source(args.input)
    mouse_controller = MouseController('medium', 'fast')

    feeder.load_data()
    frame_count = 0
    for ret, frame in feeder.next_batch():
        if not ret:
            break
        
        frame_count += 1
        results = run_inference(models, frame)
        if not results:
            continue

        preview_frame = draw_preview(frame, results, args.previewFlags)
        cv2.imshow('Preview', preview_frame)
        save_video(os.path.join(args.output_path, "output_video.mp4"), frame)

        move_mouse(mouse_controller, results["mouse_coords"], frame_count)

        if cv2.waitKey(60) == 27:
            break

    cv2.destroyAllWindows()
    feeder.close()

if __name__ == '__main__':
    main()
