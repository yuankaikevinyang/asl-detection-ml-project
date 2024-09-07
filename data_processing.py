import os

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle # to extract things from this python file


#Objects that detect and draw landmarks on the hands
#mp stands for MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


#Our main directory
data_directory = './asl_alphabet_train'


data = [] #for the coords of the hand landmarks
labels = [] #categories/classes for the images


for dir_ in os.listdir(data_directory): #iterating through each directory in the main training directory



    for img_path in  os.listdir(os.path.join(data_directory, dir_))[:1]: #iterating through each image in the subdirectories

        data_aux=[]

        img = cv2.imread(os.path.join(data_directory, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb) #function comes from mediapipe
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks: #extracting landmarks
                for i in range(len(hand_landmarks.landmark)):
                    #print(hand_landmarks.landmark[i]) #gives x, y, and z coords

                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
                #Drawing on the Image
                # mp_drawing.draw_landmarks(
                #     img_rgb,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )
            data.append(data_aux)
            labels.append(dir_) #from the most outer for-loop (Should have 29)


f = open('data.pickle', 'wb') #write and binary

pickle.dump({'data': data, 'labels': labels}, f) # a dictionary

f.close()