import cv2


class Frame:
    """
    Class Frame stores image, information on bounding boxes and labels of faces and persons.
    Methods set_* are used for filling class attributes.
    Method crop_picture is used for cropping persons and faces from picture,
    and takes a string argument to signify operation type.
    Method draw_all_bounding_boxes draws labeled bboxes for persons, and, if present, for faces.
    Method draw_face_bounding_boxes draws labeled bboxes for detected faces.
    """
    def __init__(self, image):
        self.image = image
        self.person_boxes = []
        self.person_labels = []
        self.face_boxes = []

    def set_person_boxes(self, person_boxes):
        self.person_boxes = person_boxes

    def set_face_boxes(self, face_boxes):
        self.face_boxes = face_boxes

    def set_person_labels(self, person_labels):
        self.person_labels = person_labels

    def crop_picture(self, flag):
        cropped = []
        if flag == 'person':
            for i in range(len(self.person_boxes)):
                box = self.person_boxes[i]
                cropped.append([self.image[int(box[0][0]):int(box[1][0]), int(box[0][1]):int(box[1][1])],
                                box[:2]])
        elif flag == 'face':
            for i in range(len(self.face_boxes)):
                box = self.face_boxes[i]
                if box != 'NaN':
                    cropped.append(cv2.resize(self.image[int(box[0][1]):int(box[1][1]) + 1,
                                              int(box[0][0]):int(box[1][0]) + 1, :], (64, 64)))
                else:
                    cropped.append('NaN')
        return cropped

    def draw_all_labeled_bounding_boxes(self, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=0.6, thickness=1):
        try:
            for i in range(len(self.face_boxes)):
                if self.face_boxes[i][0] == 'N':
                    continue
                x1, y1 = self.face_boxes[i][0]
                x2, y2 = self.face_boxes[i][1]
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        finally:
            for i in range(len(self.person_boxes)):
                y1, x1 = self.person_boxes[i][0]
                y2, x2 = self.person_boxes[i][1]
                cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = self.person_labels[i]
                size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                cv2.rectangle(self.image, (x1, y1 - size[1]), (x1 + size[0], y1), (255, 0, 0), cv2.FILLED)
                cv2.putText(self.image, label, (x1, y1), font, font_scale, (255, 255, 255), thickness)

    def draw_face_bounding_boxes(self):

        for i in range(len(self.face_boxes)):
            if self.face_boxes[i][0] == 'N':
                continue
            x1, y1 = self.face_boxes[i][0]
            x2, y2 = self.face_boxes[i][1]
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
