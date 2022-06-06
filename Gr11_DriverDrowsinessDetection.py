import argparse
import os
import time
from threading import Thread
import cv2
import dlib
import imutils
import numpy as np
import smtplib
from imutils import face_utils
from imutils.video import VideoStream
from pygame import mixer
from scipy.spatial import distance as dist
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from geopy.geocoders import Nominatim
import geocoder

mixer.init()
sound = mixer.Sound('alarm.wav')

next = 0
def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying


    while alarm_status:
        print('call')
        s = 'espeak "()"'.format(msg)
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

def head_bend(shape):
    nose_length = shape[27:30]
    nose_length1 = abs(np.mean(nose_length,axis = 0))
    return nose_length1[1]

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 40
YAWN_THRESH = 30
HEAD_THRESH = 200
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0




Nomi_locator = Nominatim(user_agent="My App")
my_location= geocoder.ip('me')
latitude= my_location.geojson['features'][0]['properties']['lat']
longitude = my_location.geojson['features'][0]['properties']['lng']
location = Nomi_locator.reverse(f"{latitude}, {longitude}")
mail_content = '''Hello Receiver !!!  Driver is Feeling Drowsy. Driver is advised to stop the vehicle and take a break !!!''',latitude,',',longitude

print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #rects = detector(gray, 0)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                    minNeighbors=5, minSize=(30, 30),
                                    flags=cv2.CASCADE_SCALE_IMAGE)


    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        nose_dst = head_bend(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        nose = shape[27:30]
        cv2.drawContours(frame, [nose], -1, (0, 255, 0), 1)


        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                alarm_status = True
                sound.play()
                next += 1
                if next > 3:
                    sender_address = 'chiragpavesha@sjcem.edu.in'
                    sender_pass = 'Pavesha@09*'
                    receiver_address = 'chiragpavesha01@gmail.com'

                    message = MIMEMultipart()
                    message['From'] = sender_address
                    message['To'] = receiver_address

                    message['Subject'] = 'Drowsiness Detected in your Vehicle.'

                    message.attach(MIMEText(mail_content, 'plain'))

                    session = smtplib.SMTP('smtp.gmail.com', 587)
                    session.starttls()
                    session.login(sender_address, sender_pass)
                    text = message.as_string()
                    session.sendmail(sender_address, receiver_address, text)
                    session.quit()
                    print('Mail Sent Successfully to Concerned Person')
                t = Thread(target=alarm, args=('Stay awake',))
                t.deamon = True
                t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            alarm_status = False

        if (distance > YAWN_THRESH):
            sound.play()
            cv2.putText(frame, "Yawn Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=alarm, args=('take a break',))
                t.deamon = True
                t.start()
        else:
            alarm_status2 = False

        print(nose_dst)

        if (nose_dst > HEAD_THRESH):
            sound.play()
            cv2.putText(frame, "Head tilt Alert", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm_status2 == False and saying == False:
                alarm_status2 = True
                t = Thread(target=alarm, args=('Wake Up',))
                t.deamon = True
                t.start()
        else:
            alarm_status2 = False



        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
