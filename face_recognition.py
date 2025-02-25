import face_recognition
import cv2
import os

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def add_known_face(self, image_path, name):
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)

    def recognize_faces(self, image_path):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        return image

# Example usage
if __name__ == "__main__":
    face_recognition_system = FaceRecognitionSystem()
    face_recognition_system.add_known_face("known_faces/alice.jpg", "Alice")
    face_recognition_system.add_known_face("known_faces/bob.jpg", "Bob")

    image_path = "test_images/test1.jpg"
    recognized_image = face_recognition_system.recognize_faces(image_path)

    cv2.imshow("Recognized Faces", recognized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()