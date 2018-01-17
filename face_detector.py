import dlib
import cv2
import numpy as np
import logging


class FaceDetector:
    """
    Class, that uses dlib face detector to detect faces.
    Method detect_face_of_person takes a cropped picture of a person and finds a face on it;
    returns coordinates of a face on image.
    Method detect_faces_from_image takes a picture/frame and finds all faces on it(above threshold);
    returns an array of coordinates of faces on image.
    """
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def detect_face_of_person(self, person, threshold=0):
        (top, left), _ = person[1]

        # Change colorspace of image
        new_image = cv2.cvtColor(person[0], cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(new_image)

        # Run face detector, use only most accurate face or return NaN
        dets, scores, idx = self.face_detector.run(new_image, 1, -1)
        try:
            max_score = max(scores)
            if max_score > threshold:
                for i, d in enumerate(dets):
                    logging.info("Detection {}, score: {}, face_type:{}".format(
                        d, scores[i], idx[i]))
                    if scores[i] == max_score:
                        max_i = i

                coord = [dets[max_i].left(), dets[max_i].top(), dets[max_i].right() + 1, dets[max_i].bottom() + 1]

                for i, item in enumerate(coord):
                    print(item)
                    if item < 0:
                        coord[i] = 0

                x1, y1, x2, y2 = coord
                face_box = [(x1+left, y1+top), (x2+left, y2+top)]
            else:
                face_box = 'NaN'
        except:
            face_box = 'NaN'
        return face_box

    def detect_faces_from_image(self, image, threshold=0):
        face_box = []

        # Change colorspace of image
        new_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = np.shape(new_image)

        # Run face detector, use only most accurate face or return NaN
        dets, scores, idx = self.face_detector.run(new_image, 1, -1)
        try:
            for i, d in enumerate(dets):
                print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))
                if scores[i] > threshold:
                    coord = [d.left(), d.top(), d.right() + 1, d.bottom() + 1]

                    for i, item in enumerate(coord):
                        if item < 0:
                            coord[i] = 0

                    x1, y1, x2, y2 = coord

                    face_box.append([(x1, y1), (x2, y2)])
                else:
                    face_box.append('NaN')
        except:
            face_box.append('NaN')
        return face_box
