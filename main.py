""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args


def baseline_bgs(args):
    with open(args.eval_frames,"r") as eval_frames_file:
        eval_interval = str(eval_frames_file.readline()).split(' ')
        eval_start = int(eval_interval[0])
        eval_end = int(eval_interval[1])
    # Create an OpenCV BackgroundSubtractor object to generate the foreground mask
    #BackgroundSubtractor = cv2.createBackgroundSubtractorMOG2()
    BackgroundSubtractor = cv2.createBackgroundSubtractorKNN(history = 2000,dist2Threshold = 300.0) # Better than MOG2

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    
    for dirs,subdirs,files in os.walk(args.inp_path):
        for file in tqdm(files):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                file_number = int(str(file)[2:][:-4])
                input_image = os.path.join(args.inp_path,file)
                frame = cv2.imread(input_image)
                #frame = cv2.blur(frame,(7,7))              # No Effect
                #frame = cv2.GaussianBlur(frame,(3,3),0)    # No Effect
                
                fgmask = BackgroundSubtractor.apply(frame)
                fgmask_blurred = cv2.GaussianBlur(fgmask,(11,11),0)
                fgmask_blurred = cv2.medianBlur(fgmask_blurred,11)
                # Can use Adaptive Thresholding below
                ret,fgmask_thresholded = cv2.threshold(fgmask_blurred,150,255,cv2.THRESH_BINARY)

                # The contour area threshold technique used below -> NOT Effective in this case
                """
                contours, hierarchy = cv2.findContours(fgmask_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in contours:
                    if cv2.contourArea(cnt)>=30:
                        cv2.fillPoly(fgmask_thresholded, pts =[cnt], color=(255,255,255))
                        cv2.fillPoly(frame, pts =[cnt], color=(0,255,0))
                        
                    if cv2.contourArea(cnt)<30:
                        cv2.fillPoly(fgmask_thresholded, pts =[cnt], color=(0,0,0))
                        cv2.fillPoly(frame, pts =[cnt], color=(0,0,255))
                """
                
                cv2.imshow('original',frame)                            # For Visualisation
                cv2.imshow('BackgroundSubtractor', fgmask_thresholded)  # For Visualisation
                if  file_number >= eval_start and file_number<= eval_end :
                    cv2.imwrite(str(os.path.join(args.out_path,"gt"+file[:-4][2:]))+".png",fgmask_thresholded)
                #print(file)
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    print("Done")
                    break
        # Closes all the frames
        cv2.destroyAllWindows()
    pass


def illumination_bgs(args):
    def normalize(arr):
        arr = arr.astype('float')
        # Do not touch the alpha channel
        for i in range(3):
            minval = arr[...,i].min()
            maxval = arr[...,i].max()
            if minval != maxval:
                arr[...,i] -= minval
                arr[...,i] *= (255.0/(maxval-minval))
        return arr
    with open(args.eval_frames,"r") as eval_frames_file:
        eval_interval = str(eval_frames_file.readline()).split(' ')
        eval_start = int(eval_interval[0])
        eval_end = int(eval_interval[1])
    # Create an OpenCV BackgroundSubtractor object to generate the foreground mask
    BackgroundSubtractor = cv2.createBackgroundSubtractorMOG2(history = 60,varThreshold = 10) # Better than KNN
    # BackgroundSubtractor = cv2.createBackgroundSubtractorKNN(history = 20,dist2Threshold = 300.0)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    
    for dirs,subdirs,files in os.walk(args.inp_path):
        for file in tqdm(files):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                file_number = int(str(file)[2:][:-4])
                # if  file_number >= eval_start and file_number<= eval_end :
                input_image = os.path.join(args.inp_path,file)
                frame = cv2.imread(input_image)
                
                frame = normalize(frame).astype('uint8') # Normalize -> No Effect
                #frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # No Effect

                fgmask = BackgroundSubtractor.apply(frame)
                fgmask_blurred = cv2.GaussianBlur(fgmask,(5,5),0)
                fgmask_blurred = cv2.medianBlur(fgmask_blurred,7)

                # Can use Adaptive Thresholding below
                ret,fgmask_thresholded = cv2.threshold(fgmask_blurred,150,255,cv2.THRESH_BINARY)
                kernel = np.ones((3,3), np.uint8)
                # fgmask_thresholded= cv2.erode(fgmask_thresholded, kernel, iterations=2) # Decreases Accuracy
                fgmask_thresholded= cv2.dilate(fgmask_thresholded, kernel, iterations=2) # Best Setting
                
                cv2.imshow('original',frame)                # For Visualisation
                cv2.imshow('BackgroundSubtractor', fgmask_thresholded)
                final_img = cv2.resize(fgmask_thresholded,(320,240))
                if  file_number >= eval_start and file_number<= eval_end :
                    cv2.imwrite(str(os.path.join(args.out_path,"gt"+file[:-4][2:]))+".png",final_img)
                #print(file)
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    print("Done")
                    break
                # Closes all the frames
        cv2.destroyAllWindows()
    pass

def jitter_bgs(args):
    with open(args.eval_frames,"r") as eval_frames_file:
        eval_interval = str(eval_frames_file.readline()).split(' ')
        eval_start = int(eval_interval[0])
        eval_end = int(eval_interval[1])
    # Create an OpenCV BackgroundSubtractor object to generate the foreground mask
    # BackgroundSubtractor = cv2.createBackgroundSubtractorMOG2() 
    BackgroundSubtractor = cv2.createBackgroundSubtractorKNN(history = 1000,dist2Threshold = 200.0) # Better than MoG2

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    frames = []
    files_arr = []
    for dirs,subdirs,files in os.walk(args.inp_path):
        for file in (files):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                files_arr.append(str(file))
                frames.append(cv2.imread(os.path.join(args.inp_path,file)))
    previous_frame = frames[0]
    previous_frame_grayscale = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY) # Set Reference for checking Transformation
    # Get width and height of video stream
    w,h = previous_frame_grayscale.shape
    # Pre-define the transformation-store array
    transforms = np.zeros((len(frames)-1, 3), np.float32)

    for i in tqdm(range(1,len(frames)-1)):
        frame = frames[i]
        # Detect the good feature points in previous_frame
        prev_features = cv2.goodFeaturesToTrack(previous_frame_grayscale,maxCorners=100,qualityLevel=0.1,minDistance=50,blockSize=11)
        new_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate optical flow (i.e. track feature points)
        new_features, status, err = cv2.calcOpticalFlowPyrLK(previous_frame_grayscale, new_frame_grayscale, prev_features, None)
        # Filter only valid points
        valid_idx = np.where(status==1)[0]
        prev_features = prev_features[valid_idx]
        new_features = new_features[valid_idx]
        # Transformation matrix
        M = cv2.estimateAffinePartial2D(prev_features, new_features)
        # valid_idx = np.where(M[1]==1)[0]
        # prev_features = prev_features[valid_idx]
        # new_features = new_features[valid_idx]
        M = M[0]
        dx = M[0,2]
        dy = M[1,2]
        da = np.arctan2(M[1,0], M[0,0])
        # Reconstruct transformation matrix accordingly to new values
        M = np.zeros((2,3), np.float32)
        M[0,0] = np.cos(da)
        M[0,1] = -np.sin(da)
        M[1,0] = np.sin(da)
        M[1,1] = np.cos(da)
        M[0,2] = dx
        M[1,2] = dy
        # Invert Transformation to get untransformed frame
        M_inv = cv2.invertAffineTransform(M)
        # Undo the transformation & Stabilize the frame
        new_frame_grayscale_stabilised = cv2.warpAffine(frame, M_inv, (h,w))

        fgmask = BackgroundSubtractor.apply(new_frame_grayscale_stabilised)
        #fgmask = cv2.warpAffine(fgmask, M, (h,w))
        fgmask_blurred = cv2.GaussianBlur(fgmask,(15,15),0)
        fgmask_blurred = cv2.medianBlur(fgmask_blurred,11)
        # Can use Adaptive Thresholding below
        ret,fgmask_thresholded = cv2.threshold(fgmask_blurred,200,255,cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        fgmask_thresholded= cv2.erode(fgmask_thresholded, kernel, iterations=1) 
        fgmask_thresholded= cv2.dilate(fgmask_thresholded, kernel, iterations=2)
        contours, hierarchy = cv2.findContours(fgmask_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
              
        cv2.imshow('Original',frame)                                           # For Visualisation
        cv2.imshow('Stabilized',new_frame_grayscale_stabilised)                # For Visualisation
        cv2.imshow('BackgroundSubtractor', fgmask_thresholded)
        file = files_arr[i]
        file_number = int(str(file)[2:][:-4])
        if  file_number >= eval_start and file_number<= eval_end:
            cv2.imwrite(str(os.path.join(args.out_path,"gt"+file[:-4][2:]))+".png",fgmask_thresholded)
        #print(file)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF==ord('q'):
            print("Done")
            break

    # Closes all the frames
    cv2.destroyAllWindows()
    pass

def dynamic_bgs(args):
    with open(args.eval_frames,"r") as eval_frames_file:
        eval_interval = str(eval_frames_file.readline()).split(' ')
        eval_start = int(eval_interval[0])
        eval_end = int(eval_interval[1])
    # Create an OpenCV BackgroundSubtractor object to generate the foreground mask
    #BackgroundSubtractor = cv2.createBackgroundSubtractorMOG2()
    BackgroundSubtractor = cv2.createBackgroundSubtractorKNN(history = 500,dist2Threshold = 400.0) # Better than MoG2

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    
    for dirs,subdirs,files in os.walk(args.inp_path):
        for file in tqdm(files):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                file_number = int(str(file)[2:][:-4])
                input_image = os.path.join(args.inp_path,file)
                frame = cv2.imread(input_image)
                
                frame = cv2.GaussianBlur(frame,(3,3),0)    # No Effect
                frame = cv2.medianBlur(frame,3)              # No Effect

                cv2.imshow('original',frame)                # For Visualisation
                fgmask = BackgroundSubtractor.apply(frame)
                fgmask_blurred = cv2.GaussianBlur(fgmask,(11,11),0)
                fgmask_blurred = cv2.medianBlur(fgmask_blurred,3)
                # Can use Adaptive Thresholding below
                ret,fgmask_thresholded = cv2.threshold(fgmask_blurred,150,255,cv2.THRESH_BINARY)
                        
                kernel = np.ones((3,3), np.uint8)
                fgmask_thresholded= cv2.erode(fgmask_thresholded, kernel, iterations=1) # 
                #fgmask_thresholded= cv2.dilate(fgmask_thresholded, kernel, iterations=2) # decreases accuracy

                contours, hierarchy = cv2.findContours(fgmask_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in contours:
                    if cv2.contourArea(cnt)>=100:
                        cv2.fillPoly(fgmask_thresholded, pts =[cnt], color=(255,255,255))
                        #cv2.fillPoly(frame, pts =[cnt], color=(0,255,0))
                        
                    if cv2.contourArea(cnt)<100:
                        cv2.fillPoly(fgmask_thresholded, pts =[cnt], color=(0,0,0))
                        #cv2.fillPoly(frame, pts =[cnt], color=(0,0,255))
                
                cv2.imshow('original',frame)                # For Visualisation
                cv2.imshow('BackgroundSubtractor', fgmask_thresholded)
                
                #print(file)
                # Press Q on keyboard to  exit
                if cv2.waitKey(1) & 0xFF==ord('q'):
                    print("Done")
                    break
                if  file_number >= eval_start and file_number<= eval_end:
                    cv2.imwrite(str(os.path.join(args.out_path,"gt"+file[:-4][2:]))+".png",fgmask_thresholded)

        # Closes all the frames
        cv2.destroyAllWindows()
    pass


def main(args):
    if args.category not in "bijd":
        raise ValueError("category should be one of b/i/j/m - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "d": dynamic_bgs,
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
