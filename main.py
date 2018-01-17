import logging
import os
from time import time, sleep
from threading import Thread

import cv2
from flask import Flask, redirect, flash, send_file, abort, request
from werkzeug.utils import secure_filename

from frame import Frame
from model_manager import ModelManager

app = Flask(__name__)
logging.basicConfig(
    level="INFO",
    format="[%(levelname)s %(asctime)s path:%(pathname)s line:%(lineno)s] %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S"
)


def get_local_filename(file_name):
    """
    Method creates local path for file in storage.
    :param file_name: [str] file name
    :return: [str] file path in storage
    """
    path = os.path.abspath(os.path.join('storage', file_name))
    return path


def video_processor(input_file, output_file, file_type, threshold_person, threshold_face):
    """
    Main method. Process video or image. Detects persons at frames/image, detects face at person, 
    writes bounding box in person and his face.
    :param input_file: [str] path to a video or image file
    :param output_file: [str] path to save processed video/image file
    :param file_type: [str ('video' or 'image')] type of file
    :param threshold_person: [float] confidence level (Threshold) for person detection algorithm
    :param threshold_face: [float] confidence level (Threshold) for face detection algorithm
    :return: [None]
    """
    logging.log(logging.INFO, 'Processing {} from {} to {}'.format(file_type, input_file, output_file))
    global model_manager

    if file_type == 'video':
        # Create a VideoCapture
        cap = cv2.VideoCapture(input_file)

        # Initialise a counter
        frame_number = 0
        # save parameters of video
        video_width = int(cap.get(3))
        video_height = int(cap.get(4))
        frame_rate = int(cap.get(5))
        # Create a VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(output_file, fourcc, frame_rate, (video_width, video_height))

        start_fps = time()
        # Going through video frame by frame
        logging.log(logging.INFO, 'Start detection process loop for {}.'.format(os.path.split(input_file)[-1]))
        while cap.isOpened():

            start = time()
            logging.log(logging.INFO, "Analysing capture {}".format(frame_number))
            frame_number += 1

            # Get next frame of input video, create Frame object
            is_retrieved, image = cap.read()
            if not is_retrieved:
                break
            current_frame = Frame(image)

            model_manager.put_all_predictions_into_frame(current_frame, threshold_person, threshold_face)
            current_frame.draw_all_labeled_bounding_boxes()

            if frame_number % 5 == 0:
                logging.log(logging.INFO, '{0:3.3f} FPS \r'.format(
                    frame_number / (time() - start_fps)))

            writer.write(current_frame.image)

            finish = time() - start
            logging.log(logging.INFO, "Analysing ended in %f" % finish)

        writer.release()
        cap.release()
    elif file_type == 'image':
        current_frame = Frame(cv2.imread(input_file))
        model_manager.put_all_predictions_into_frame(current_frame, threshold_person, threshold_face)
        current_frame.draw_all_labeled_bounding_boxes()
        cv2.imwrite(output_file, current_frame.image)


def start_delete_old_files_daemon(live_time=60):
    """
    Creates util thread that deleting files that "live time" is greater than given.
    :param live_time: [int] maximum live time of file
    :return: [None]
    """

    def delete_old_files():
        storage_dir = os.path.abspath('storage')
        while True:
            logging.log(logging.INFO, 'Checking and deleting old files.')
            for root, dirs, files in os.walk(storage_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    file_live_time = time() - os.path.getmtime(file_path)
                    if live_time <= file_live_time:
                        logging.log(logging.INFO, 'Deleting {} with live time {} file.'.format(file_name,
                                                                                               file_live_time))
                        os.remove(file_path)
            sleep(live_time // 2)

    logging.log(logging.INFO, 'Starting old file monitoring.')
    Thread(target=delete_old_files).start()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    Method uploads file to local storage and runs person detection algorithm.
    :return: [str] URL to a processed video/image
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file_to_process = request.files['file']
        file_name = secure_filename(file_to_process.filename)
        file_type = request.form.get('file_type')
        logging.log(logging.INFO, 'Uploading {} file to server storage'.format(file_name))

        threshold_person = float(request.form.get('threshold_person') or 0.5)
        threshold_face = float(request.form.get('threshold_face') or 0.1)

        input_path = get_local_filename(os.path.join('input_files', file_name))
        output_path = get_local_filename(os.path.join('output_files', 'processed_{}'.format(file_name)))
        if not os.path.exists(os.path.dirname(input_path)):
            os.makedirs(os.path.dirname(input_path))
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        file_to_process.save(input_path)

        logging.log(logging.INFO,
                    'Starting to process {} {} file with {} person threshold and {} face threshold'
                    .format(
                        file_name,
                        file_type,
                        threshold_person,
                        threshold_face
                    ))
        video_processor(input_path,
                        output_path,
                        file_type,
                        threshold_person,
                        threshold_face)

        download_url = '{}processed/{}'.format(request.base_url, file_name)
        logging.log(logging.INFO, 'File {} uploaded and processed successfully.'.format(file_name))
        return '''
        <!doctype html>
        <title>Processed file</title>
        <h1>Processed file</h1>
        <a href="{}" download>{}</a>
        '''.format(download_url, file_name)
    return '''
    <!doctype html>
    <title>Person detection</title>
    <h1>Upload video/image file for person detection process.</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <select name="file_type">
           <option value="image">Image</option>
           <option value="video">Video</option>
         </select>
         <input type=submit value=Upload>
      <p>Threshold for person detection  </p><input type="text" name="threshold_person">
      <p>Threshold for face detection    </p><input type="text" name="threshold_face">
    </form>
    '''


@app.route('/processed/<path:path>', methods=['GET'])
def return_processed_file(path):
    """
    Methods give processed video/image file by URL.
    :param path: [str] file name
    :return: [obj] requested file (if exists)
    """
    logging.log(logging.INFO, 'Retrieving {} file from server storage by url.'.format(path))
    try:
        file_path = get_local_filename(os.path.join('output_files', 'processed_{}'.format(path)))
        resp = send_file(file_path)
    except:
        logging.log(logging.ERROR, 'File {} not found.'.format(path))
        abort(500)
    return resp


if __name__ == "__main__":
    start_delete_old_files_daemon(live_time=60*30)
    # Create manager objects
    model_manager = ModelManager()
    model_manager.build_graph()
    logging.log(logging.INFO, 'Starting person detection server.')
    app.run(host='0.0.0.0')
