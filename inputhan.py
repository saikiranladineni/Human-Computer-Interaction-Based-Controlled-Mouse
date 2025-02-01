import os
import logging
from input_feeder import InputFeeder

def get_input_source(input_filename):
    logger = logging.getLogger("InputHandler")

    if input_filename.lower() == 'cam':
        return InputFeeder(input_type='cam')

    if not os.path.isfile(input_filename):
        logger.error("Specified video file not found")
        exit(1)

    return InputFeeder(input_type='video', input_file=input_filename)
