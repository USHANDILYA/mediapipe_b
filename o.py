# Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.
import tensorflow as tf
import mediapipe as mp
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
mp_holistic=mp.solutions.holistic #holisitic model to make detections
mp_drawing=mp.solutions.drawing_utils # drawing utilities to draw detections
#function to read detections and draw them
def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #since opencv reads in bgr format but for mediapipe we need to read in egb format
    image.flags.writeable=False #image is not writeable
    results=model.process(image) #detecting using mediapipe
    image.flags.writeable=True #image is now writeable
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
#we grab our landmarks and render them onto the image
#we will style our landmarks
def draw_styled_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),#colour landmarks
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)) #colour connections
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),#colour landmarks
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)) #colour connections
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),#colour landmarks
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)) #colour connections
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(80,110,10),thickness=1,circle_radius=1),#colour landmarks
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness=1,circle_radius=1)) #colour connections


#we will write all 4 keypoint extractions in 1 function
def extract_keypoints(results):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face,lh,rh]) #here we concatenate all of our keypoints

#path for exported data i.e numpy array
DATA_PATH=os.path.join('MP_Data')
#actions that we try to detect
#actions=np.array(['hello','thanks','iloveyou'])
actions=np.array(['hello'])
#we will be using 30 videos worth of data
no_sequences=30
#we will be using 30 frames to classify the actions i.e videos are going to be 30 frames in length
sequence_length=30
os.chdir(r'C:\Users\UTKARSH\Desktop\data science\dl\22 aug')
for action in actions: #loop through our actions
    for sequence in range(no_sequences): #loop through 30 different videos
        try:
            os.makedirs(os.path.join(DATA_PATH,action,str(sequence))) #make subdirectories
        except:
            pass    





video_directory = r'C:\Users\UTKARSH\Desktop\data science\dl\22 aug\Waving_Hand_Right'  # Replace with the path to your video directory
video_files = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith('.mp4')]
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences): #our sequence is 1 video and each video has 30 videos in length
            for frame_num in range(sequence_length):
                video_file = video_files[sequence]  # Ensure that each sequence corresponds to one video
                cap = cv2.VideoCapture(video_file)
                
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video file {video_file}.")
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, f'COLLECTING FRAMES FOR {action} VIDEO NUMBER {sequence}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'COLLECTING FRAMES FOR {action} VIDEO NUMBER {sequence}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))  # Path structure updated
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, keypoints)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical #to convert into one hot encoding

#creating label map/array to represent our actions
label_map={label:num for num,label in enumerate(actions)}
#structure the keypoints 
#we will create a big array that will contain all data
#effectively we will have 30 arrays with 30 frames in each arrray with 1662 keypoints in each(in video its 90 arrays beacause they have 3 classes)
sequences,labels=[],[] #sequences is feature data/x_data and labels is labeldata/y_data
#we will be using our features and train a model to detect relationship between our labels
for action in actions:
    for sequence in range(no_sequences):
        window=[] #all different frames for that particular sequence
        for frame_num in range(sequence_length): #looping through each frame
            res=np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num))) #to load that frame
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])    
        #sequences array will have 90 videos and in it each will have 30 frames each
y=to_categorical(labels).astype(int)
#here we have converted our labels into one hot encoded representation
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05) #test percentage is 5 percent of our data

from tensorflow.keras.models import Sequential #to build sequential neural network
from tensorflow.keras.layers import LSTM,Dense # lstm layer that gives temporal component for action detetection
from tensorflow.keras.callbacks import TensorBoard #trace and monitor our model

#create a log directory and setup tensorboard callbacks
log_dir=os.path.join('Logs')
tb_callback=TensorBoard(log_dir=log_dir)
#tensorboard is a webapp offered as a part of tensorflow package that helps to monitor our neural network training 
model=Sequential() #instantiating model by sequential api
#now we will add 3 lstm layers
model.add(LSTM(64,return_sequences=True,activation='relu',input_shape=(30,1662))) 
#64 lstm units and return sequences is true for next layer to read those sequences and input shape is 30 frames per prediction multiplied by 1662 values
#this means each video is 30 frames with 1662 keypoints
model.add(LSTM(128,return_sequences=True,activation='relu'))
model.add(LSTM(64,return_sequences=False,activation='relu')) #next layer is dense layer so we dont return the sequences
model.add(Dense(64,activation='relu')) #dense layers are fully connected neurons
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax')) #this is actions layer
#return output of our model is 3 neural network units
#softmax returns values within probability range 0-1 with sum=1
#this we can use to preprocess and extract our actions
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])
# categorical_crossentropy is necessary loss fn. for multiclass classification model
model.fit(X_train,y_train,epochs=50,callbacks=[tb_callback])
model.save('action.h5')

        
    