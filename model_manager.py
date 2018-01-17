from face_detector import FaceDetector
from person_detector import PersonDetector


class ModelManager:
    """Class, that manages models of use."""

    def __init__(self):
        self.models = {
            'person_detector': PersonDetector(),
            'face_detector': FaceDetector(),
        }

    def build_graph(self):
        """
        Builds tensorflow graph of pre-trained model.
        :return: [None]
        """
        self.models['person_detector'].build_graph()

    def put_all_predictions_into_frame(self, frame, threshold_person, threshold_face):
        """
        Creates a pipe, that sets all Frame's fields to detected/predicted values.
        
        :param frame: [Frame obj] Frame of video or image
        :param threshold_person: [float] threshold of person detection algorithm
        :param threshold_face: [float] threshold of face detection algorithm
        :return: [None]
        """
        img = frame.image
        face_boxes = []

        # start prediction pipe
        person_boxes, person_labels = self.models['person_detector'].predict_persons(img, threshold_person)
        frame.set_person_boxes(person_boxes)
        frame.set_person_labels(person_labels)

        for person in frame.crop_picture('person'):
            face_boxes.append(self.models['face_detector'].detect_face_of_person(person, threshold_face))
        frame.set_face_boxes(face_boxes)

    def get_person_boxes(self, frame, threshold_person):
        """
        Method only sets person_boxes and person_labels in Frame;
        :param frame: 
        :param threshold_person: 
        :return: 
        """
        img = frame.image

        person_boxes, person_labels = self.models['person_detector'].predict_persons(img, threshold_person)
        frame.set_person_boxes(person_boxes)
        frame.set_person_labels(person_labels)
