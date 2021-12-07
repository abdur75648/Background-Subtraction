# Background-Subtraction #
- - - - 
### OpenCV implementation of background subtraction/foreground detection in Python ###

Background subtraction is a well-known technique for extracting the foreground objects in images
or videos. The main aim of background subtraction is to separate moving object foreground from
the background in a video, which makes the subsequent video processing tasks easier and more
efficient. Usually, the foreground object masks are obtained by performing a subtraction between
the current frame and a background model. The background model contains the characteristics of
the background or static part of a scene.

This repo uses OpenCV to perform background subtraction to find the foreground object masks for 4 different scene conditions:

This repo uses OpenCV to perform background subtraction to find the foreground object masks for 4 different scene conditions:
1. ***Baseline:*** This category contains simple videos. The camera is fixed and steady. The background is static, and there is no change in illumination conditions between the video frames. Handling the baseline data is the first step towards building a robust background subtraction system.
2. ***Illumination Changes:*** In this category, the lighting conditions may vary between the frames of a video. Changes in lighting conditions can also introduce shadows of the foreground objects, which need to be ignored.
3. ***Camera Shake (jitter):*** In this category, the camera shakes due to the vibration of the mount or the unsteady hand of a photographer
4. ***Dynamic Scenes:*** In this category, the background of the video is not static anymore. The change in the background in this case is due to movements of background objects.

### Dataset ###
The dataset can be downloaded at [ChangeDetection.Net](http://changedetection.net)

### Using This Repo ###
1. Download & put the data in their particular folders (like baseline,illumination, jitter & moving_bg)
2. Run the background subtraction using:
  ` python main.py -i datasetname/input -o datasetname/result -c X -e datasetname/eval_frames.txt `,
  where i and o are paths for the input folder and target folder for predicted masks recpectively,
  eval frames.txt file contains the starting and ending frame that will be used for evaluation,
  and c is category name ("b" for baseline, "i" for illumination, "j" for camera jitter, and "d": for dynamic background)
