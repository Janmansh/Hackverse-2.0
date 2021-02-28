# Hand Gesture to Emoji
## Dependencies

* Python 3, [OpenCV](https://opencv.org/), [Tensorflow](https://www.tensorflow.org/)
* To install the required packages, run `pip install -r requirements.txt`.

## Basic Usage

The repository is currently compatible with `tensorflow-2.0` and makes use of the Keras API using the `tensorflow.keras` library.

* First, clone the repository and enter the folder

```bash
git clone https://github.com/Janmansh/Hackverse-2.0.git
cd Hackverse-2.0
```


```bash
cd src
python gesture.py --mode display
```

* The folder structure is of the form:  
  src:
  * `gesture.py` (file)
  * `aGest.xml` (haar cascade file)
  * `model.h5` (file)

* This implementation by default detects gestures on all hands in the webcam feed. With a simple 4-layer CNN, the test accuracy reached 58.2% in 50 epochs.




## Algorithm

* First, the **haar cascade** method is used to detect hands in each frame of the webcam feed.

* The region of image containing the face is resized to **90x70** and is passed as input to the CNN.

* The network outputs a list of **softmax scores** for the seven classes of emotions.

* The emotion with maximum score is displayed on the screen.

# Video Chat Application
* Made using NodeJs and WebRTC. 
## Required packages
* npm
* Express
* Socket.io
* peerjs
* uuid
## Steps to run
* Run  `peerjs --port 3001` on terminal for establising peer to peer connection. <br/>
* Run `npm run devStart` for starting server

# Work to be Done
* The video app needs to have the ML pipeline so that we can see the emoji's on live interaction.

# Team ( VECTOR )

- Gaurav Singh
- Aniket Agrawal
- Janmansh Agarwal
