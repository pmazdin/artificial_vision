# Face authorization

Authorization via face detection in python using some Machine Learning approaches and image processing

## Dependencies

```
source setup_env.sh
python FaceApp.py
```


## Description

The aim in our project is to give a user a valid face authorization tool. It should be user friendly and intuitive.
Therefore, there is a GUI giving the instructions to follow. There are two different perspectives:

### The user's perspective:
    Choose in a GUI between options: 
    	- Train the model
    		- The program will ask the user to move his/her head in a specific way (e.g. straight/up/down/left/right)
    	- Authorize using the trained model
    		- The program will try to detect if your straight looking face is the authorized one and if it's a real person by a blinking test
    		- If yes:
                    - Ask the user to follow given head movements
                        - If passed, authorization successful
    	- Load the model from the system
    	- Save current configuration to a system
    	- there is a live stream underneath (test if webcam is working)
    	- there is a stream with some techniqued applied (head pose, orientation, etc.)

                

### From the technical perspective:
    Area of interest detection
    - Feature extraction 
    	- PCA and SVM in order to test face recognition when standing straight (many image samples takend and compared within a database)
    	- SIFT to test left and right head orientations and how many features matched among them
    - Blink detection (OpenCV)
    - Head orientation detection (NN)

