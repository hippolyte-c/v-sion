from unittest import result
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
import streamlit as st
# from streamlit_webrtc import webrtc_streamer
from PIL import Image

import string
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    #mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh,rh])


# actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
actions = np.array(['U','V','W','Y','Z'])

#actions = np.array(['hello', 'thanks', 'iloveyou'])
#actions = np.array(['A','B','bye','C','D','E','F','G','H','hello','I','iloveyou','J','K','L','M','N','Nothing','O','P','Q','R','S','T','thanks','U','V','W','X','Y','Z'])


model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(15,258)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))


colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

    return output_frame

def get_hand_bounding_rect(minX, minY, maxX, maxY, w, h):
    minX *= w
    maxX *= w
    minY *= h
    maxY *= h

    maxSize = max(maxX - minX, maxY - minY)/2
    x = (minX + maxX) / 2
    y = (minY + maxY) / 2

    return (int(x - maxSize), int(y - maxSize)), (int(x + maxSize), int(y + maxSize))

def draw_label(frame, pt1, label, trust):
    label = "Prediction: {} ({:.2f}%)".format(label, trust*100)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = 1
    text_thickness = 2
    text_padding = 10

    col_g = lerp(trust, 0.7, 1, 0, 255)
    col_r = 255 - col_g
    color_pred = (0, col_g, col_r)

    (text_w, text_h), text_baseline = cv2.getTextSize(label, text_font, text_size, text_thickness)
    text_baseline -= 2
    text_origin = (pt1[0], pt1[1] + text_padding - text_baseline)
    text_end = (pt1[0] + text_w + text_padding * 2, pt1[1] - text_h - text_padding - text_baseline)

    cv2.rectangle(frame, text_origin, text_end, color_pred, cv2.FILLED)
    cv2.putText(frame, label, (pt1[0] + text_padding, pt1[1] - text_baseline), text_font, text_size, (0, 0, 0), text_thickness, cv2.LINE_AA)

# Linear interpolation
def lerp(n, start1, stop1, start2, stop2):
    return (n - start1) / (stop1 - start1) * (stop2 - start2) + start2


# @st.cache
def hands_detection():
    model.load_weights('action_UVWYZ_full.h5')
        # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.98
    
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(-1)

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            pt1 = (0,0)
            pt2 = (0,0)
            # Read feed
            ret, frame = cap.read()
            h, w, c = frame.shape

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            #print(results)


            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            resultss = hands.process(image)
            hand_landmarks = resultss.multi_hand_landmarks

            # Draw landmarks
            draw_landmarks(image, results)

            # If a hand is detected on screen
            #print('test', hand_landmarks)
            if hand_landmarks:
                # Loop through each hand
                for hlm in hand_landmarks:

                    minX = minY = minZ = float('inf')
                    maxX = maxY = maxZ = float('-inf')

                    # Loop through each landmark
                    for lm in hlm.landmark:
                        minX = max(min(minX, lm.x), 0)
                        maxX = max(maxX, lm.x)
                        minY = max(min(minY, lm.y), 0)
                        maxY = max(maxY, lm.y)
                        minZ = max(min(minZ, lm.z), 0)
                        maxZ = max(maxZ, lm.z)

                    # Calculate the hand bounding box
                    pt1, pt2 = get_hand_bounding_rect(minX, minY, maxX, maxY, w, h)
                    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)



            sequence.append(keypoints)
            sequence = sequence[-15:]


            if len(sequence) == 15:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #print(res)
                predictions.append(np.argmax(res))
                predicted_class = actions[np.argmax(res)]
                prediction_trust = res[np.argmax(res)].item()
                #print("predicted_class",predicted_class,"prediction_trust",prediction_trust)
                draw_label(image, pt1, predicted_class, prediction_trust)


            #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)



            #cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if(sentence):
                st.markdown(sentence)


            FRAME_WINDOW.image(image)
            # Show to screen
            #cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break

        cap.release()
        cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()


def main():

    st.set_page_config(page_title="V-sion")

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>

    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    st.sidebar.image(Image.open('./lettres/logo.png'), use_column_width=True)
    choix_table = st.sidebar.radio(' ', ('V-sion', 'Entraînez-vous', 'L\'alphabet', 'Notre équipe'))

    if choix_table == 'V-sion':
        st.title("V-sion")

        st.markdown("Avec V-sion, nous voulons vous aider à apprendre la langue des signes française depuis où vous le souhaitez, quand vous le souhaitez.")
        st.markdown("Notre solution s'appuye sur le Computer Vision pour détecter les mouvements de vos mains et les interpreter en langue des signes.")
        st.header("Une solution simple, efficace et flexible.")
        st.markdown(" ")
        st.markdown(" ")

        col1, col2, col3 = st.columns(3)
        col1.metric("Précision", "99%")
        col2.metric("Alphabet", "26 lettres")
        # col3.metric("Humidity", "86%", "4%")

    elif choix_table == 'Entraînez-vous':

        st.title("Entraînez-vous")
        # webrtc_streamer(key="loopback")
        hands_detection()

        st.info("Suite de lettres à reproduire")

        # un = ["a", "b", "c", "d", "e"]
        # deux = ["f", "g", "h", "i", "j"]
        # total = [un, deux]
        #
        # if 'suite' not in st.session_state:
        #     st.session_state.suite = []
        #
        # def x():
        #     st.session_state.suite.append(total[random.randint(0,1)])
        #
        #     for e in st.session_state.suite:
        #         st.text(e)
        #
        # st.button("a", on_click=x())

    elif choix_table == 'L\'alphabet':

        st.title("L'alphabet")
        st.text("Apprenez la Langue des Signes Françaises grâce aux lettres imagées")

        k = 0
        for i in range(1,3): # number of rows in your table! = 2
            cols = st.columns(3) # number of columns in each row! = 2
            # first column of the ith row
            cols[0].image(Image.open('./lettres/' + list(string.ascii_lowercase)[k] + '.png'), use_column_width=True, caption='Lettre '+list(string.ascii_uppercase)[k])
            cols[1].image(Image.open('./lettres/' + list(string.ascii_lowercase)[k+1] + '.png'), use_column_width=True, caption='Lettre '+list(string.ascii_uppercase)[k+1])
            cols[2].image(Image.open('./lettres/' + list(string.ascii_lowercase)[k+2] + '.png'), use_column_width=True, caption='Lettre '+list(string.ascii_uppercase)[k+2])
            k+=3
        # st.image(Image.open('C:\\Users\\hippo\\Desktop\\PFE_V-sion\\lettres\\a.png'), caption='Lettre A')

    elif choix_table == 'Notre équipe':

        st.title("Notre équipe")
        st.text("Nous sommes 5 étudiants")
        st.header("Nous contacter")
        st.text("Contact ?")

if __name__ == '__main__':
    main()
