Formal Description:
The wrapper class is a Python-based implementation that applies the trained RES-NET 18 model to detect the facial emotions portrayed by a person in a given video file. The program captures a set of frames from the video file that represent various emotions portrayed by the person's face. These frames are then passed through the model that predicts the emotion labels and associated confidence scores for each frame.

Requirements:
●	torch
●	torchvision
●	scikit-learn
●	tqdm

Steps to run:
Run the following command through the command prompt or terminal, after navigating to the project’s directory.

python wrapper.py <video_name>

Provide any video that contains a person’s face as the subject of the video (eg, A zoom video conference)  as an argument to the above command.
 
What the wrapper class does:
Running the wrapper class will output the following:
1.	A set of images, which are the video frames captured across different parts of the video, along with the emotions shown by the person predicted by the model.
2.	A quantification of the person’s emotions predicted  across the video, highlighting the top 2 frequent emotions portrayed by the person.

How it works:
1.	As a first step , a set of frames is captured from the video file that is uploaded as the argument , through the save_i_keyframes method. These frames correspond to various scenes / emotions portrayed by the face of the person in the video. This requires that the system in which we are running this to be installed with the ffmpeg package and it should be accessible from the command prompt. ‘gt(scene,0.00xx)’ value in the command can be tweaked, if you are getting too many frames or too little frames.
2.	The frames will then be sent to the trained prediction model, to detect the facial emotions in each of these frames , by calling the showPredictions() method.
3.	The above method would do the required image transformations, like cropping, resizing and normalizing, in order to match  the preprocessing done during the training phase through the transform_image()method. Once the preprocessing is done, The transformed image tensor will be sent as the input to the model through the get_prediction()method.The model will output the prediction emotion label out of the seven emotion classes (neutral, happy, sad, angry , disgust and surprise) , along with the confidence score.
4.	The keyframes will be plotted, along with the predicted emotions and the model’s confidence score. This plot will be stored under a file named after the video name, along with the keyframes.
5.	The program will also compute the overall percentage of each and every emotion captured through the keyframes, and print out the top two frequent emotions predicted by the model along with the percentage of detection.

Note: To train the model again run the command `python main.py` and after training is done run the command `python wrapper.py <video_name>`.

