import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

attendance_path = 'ImagesAttendance'
image_list = []
class_names = []
file_list = os.listdir(attendance_path)
print(file_list)
for file_name in file_list:
    current_image = cv2.imread(f'{attendance_path}/{file_name}')
    image_list.append(current_image)
    class_names.append(os.path.splitext(file_name)[0])
print(class_names)

def encode_faces(image_list):
    encoded_faces = []
    for image in image_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(image)[0]
        encoded_faces.append(face_encoding)
    return encoded_faces

def record_attendance(person_name):
    with open('C:\\Users\\mazen\\PycharmProjects\\Face_Detection\\Attendance.csv', 'r+') as file:
        data_lines = file.readlines()
        recorded_names = []
        for line in data_lines:
            entry = line.split(',')
            recorded_names.append(entry[0])
        if person_name not in recorded_names:
            current_time = datetime.now()
            time_string = current_time.strftime('%H:%M:%S')
            file.writelines(f'\n{person_name},{time_string}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def capture_screen(bbox=(300,300,690+300,530+300)):
#     screen_capture = np.array(ImageGrab.grab(bbox))
#     screen_capture = cv2.cvtColor(screen_capture, cv2.COLOR_RGB2BGR)
#     return screen_capture

known_encodings = encode_faces(image_list)
print('Encoding Complete')

video_capture = cv2.VideoCapture(0)

while True:
    success, frame = video_capture.read()
    # frame = capture_screen()
    resized_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    current_frame_faces = face_recognition.face_locations(resized_frame)
    current_frame_encodings = face_recognition.face_encodings(resized_frame, current_frame_faces)

    for face_encoding, face_location in zip(current_frame_encodings, current_frame_faces):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        # print(face_distances)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            detected_name = class_names[best_match_index].upper()
            # print(detected_name)
            top, right, bottom, left = face_location
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, detected_name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            record_attendance(detected_name)

    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)
