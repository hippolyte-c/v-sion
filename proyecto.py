from unittest import result
import cv2
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from PIL import Image

import string
import random
import glob

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


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


def load_model(lettre, text, lettres_from_modeles):


    x = random.choice(lettres_from_modeles)

    text.empty()

    all = [x for x in list(x) if x != lettre]
    choix = random.choice(all)

    text.info("Lettre à reproduire : " + choix)

    actions = np.array(list(x))

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(15,258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    print(x)
    model.load_weights('./modeles/action_' + x + '_full.h5')

    return model, actions, choix, 1


def hands_detection():

    lettres_from_modeles = []

    for file in glob.glob("./modeles/*.h5"):
        lettres_from_modeles.append(file.split("_")[1])

    modele = 0
    actions = 0
        # 1. New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.98

    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    lettre = ""

    text = st.empty()

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while run:

            if(not modele):
                model, actions, lettre, modele = load_model(lettre, text, lettres_from_modeles)

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
                                # st.markdown(sentence[-1])

                        else:
                            sentence.append(actions[np.argmax(res)])
                            # st.markdown(sentence[-1])
                        if(sentence[-1] == lettre):
                            modele = 0

                if len(sentence) > 5:
                    sentence = sentence[-5:]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(image)

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
    st.markdown(hide_streamlit_style,unsafe_allow_html=True)

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
        col2.metric("Alphabet", "26 lettres", "Bientôt des mots")
        # col3.metric("Humidity", "86%", "4%")

    elif choix_table == 'Entraînez-vous':

        st.title("Entraînez-vous")
        hands_detection()


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

    elif choix_table == 'Notre équipe':

        st.title("Notre équipe")
        st.markdown("Nous sommes 5 étudiants ingénieurs de l'ECE qui portont nos connaissances en informatique et en intelligence artificielle pour faciliter l'accès à l'apprentissage.")
        st.header("Nous contacter")
        st.markdown("Vous pouvez nous contacter à l'adresse mail v-sion@gmail.com")

if __name__ == '__main__':
    main()
