# Person Detector

Detects people/faces on a video or image.
We use different frameworks for solving this task:
- Person detection - [MobileNet from Tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
- Face detection [dlib python API](https://github.com/davisking/dlib).

## Dependencies

- Python 3.5.2
- Keras 2.0.8
- Tensorflow 1.3.0
- scipy, numpy, Pandas, tqdm, tables, h5py
- dlib
- OpenCV 3.1.0
- Flask

All dependencies are listed in the requirements.txt file for convenience.
All dependencies are build in docker image person_detector_image:latest
## Run

To run person detector you need to go `http://<your_host>:5000/` (or `http://localhost:5000/`).

After that, processed video will be available by 'http://<your_host>/processed/<your_file_name>'
(will print URL after detection process)

`threshold_person` - confidence level for person detection algorithm;
`threshold_face` - confidence level for face detection algorithm;
