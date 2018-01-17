import os
import sys
import logging

import numpy as np
import tensorflow as tf
from utils import label_map_util


class PersonDetector:
    """Class that contains all about person detection process on Frame."""

    def __init__(self):
        self.category_index = None
        self.det_classes = ['person']
        self.detection_graph = None
        self.detect = None

    def build_graph(self):
        """Builds Tensorflow graph, load model and labels."""
        # Download the mobile net model and ckpts.
        sys.path.append(".")
        MODEL_NAME = 'model_data'

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join(MODEL_NAME, 'mscoco_label_map.pbtxt')
        NUM_CLASSES = 90

        # Load Tensorflow model into memory
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Load lebel_map
        self.__load_label(PATH_TO_LABELS, NUM_CLASSES, use_disp_name=True)
        # Cut label map
        self.detect = self.__classes_to_detect()
        # print(self.category_index)

        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as self.sess:
                # Definite input and output Tensors for detection_graph
                self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def predict_persons(self, frame, threshold):
        """
            Predicts person in frame with threshold level of confidence
            Returns list with top-left, bottom-right coordinates and list with labels, confidence in %
        """
        logging.log(logging.INFO, "Starting predictions.")

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(frame, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})

        # Select classes to detect
        scores, classes = self.__det_selection(classes, scores)

        # Find detected boxes coordinates
        return self.__boxes_coordinates(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            min_score_thresh=threshold,
        )

    def __boxes_coordinates(self,
                            image,
                            boxes,
                            classes,
                            scores,
                            max_boxes_to_draw=20,
                            min_score_thresh=.5):
        """
          This function groups boxes that correspond to the same location
          and creates a display string for each detection and overlays these
          on the image.

          Args:
            image: uint8 numpy array with shape (img_height, img_width, 3)
            boxes: a numpy array of shape [N, 4]
            classes: a numpy array of shape [N]
            scores: a numpy array of shape [N] or None.  If scores=None, then
              this function assumes that the boxes to be plotted are groundtruth
              boxes and plot all boxes as black with no classes or scores.
            category_index: a dict containing category dictionaries (each holding
              category index `id` and category name `name`) keyed by category indices.
            use_normalized_coordinates: whether boxes is to be interpreted as
              normalized coordinates or not.
            max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
              all boxes.
            min_score_thresh: minimum score threshold for a box to be visualized
        """

        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        number_boxes = min(max_boxes_to_draw, boxes.shape[0])
        person_boxes = []
        person_labels = []
        for i in range(number_boxes):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                ymin, xmin, ymax, xmax = box

                im_height, im_width, _ = image.shape
                left, right, top, bottom = [int(z) for z in (xmin * im_width, xmax * im_width,
                                                             ymin * im_height, ymax * im_height)]

                if classes[i] in self.category_index.keys():
                    class_name = self.category_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                display_str = '{}: {}% '.format(class_name,
                                                int(100 * scores[i]))
                # print([(top, left),(bottom, right),display_str])
                person_boxes.append([(top, left),(bottom, right),display_str])
                person_labels.append(display_str)
        return person_boxes, person_labels

    def __classes_to_detect(self):
        """
            Find classes from det in cat and returns new classes mask
        """
        category_index_p = {}
        for i in np.linspace(1, len(self.category_index), len(self.category_index)):
            try:
                if self.category_index[i]['name'] in self.det_classes:
                    en_value = self.category_index[i]
                    category_index_p[i] = en_value
            except:
                continue
        return np.array(list(category_index_p.keys()))

    def __load_label(self, path, num_c, use_disp_name=True):
        """
            Loads labels
        """
        label_map = label_map_util.load_labelmap(path)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_c,
                                                                    use_display_name=use_disp_name)
        self.category_index = label_map_util.create_category_index(categories)

    def __det_selection(self, classes, scores):
        """
            Function that find detection classes at processed matrix
        """
        null_values = np.where(np.in1d(classes, self.detect) == False)
        classes[:, null_values] = 0
        scores[:, null_values] = 0

        return scores, classes