# Face-Recognition
Python 3.6 script to recognize faces from a database using Opencv 3.3.0

This script will recognize the faces using a Live CAM of your Laptop/Desktop using LBPHFaceRecognizer of OPENCV.
<br/>
Dataset will be created by taking required number of images ( user controlled ) from CAM and assign a unique ID to them. <br/>

Trainner will train the LBPH Face Recognizer (Inbuild recognizer of Opencv) and will create a .yml file. <br/>

Detection will be done by detecting the face from the frames ( CAM Video Recorder) and using HAARCASCADE. Identity of detected face will be predicted by LBPH Face Recognizer and it will send back the unique ID of the recognized person.<br/>
