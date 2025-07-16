#%%
from ultralytics import YOLO
import torch
import ultralytics
import cv2
import urllib
import numpy as np
import time
from datetime import datetime
import sys
import os


print('ultralytics version : ',ultralytics.__version__)
print('torch version : ',torch.__version__)

#%%
print ('Current Working directory :',os.getcwd())  # Prints the current working directory

#os.chdir(r'C:\apps\Schaeffler\AI-aionthefly-environment\yolo')  # Provide the new path here

#print ('New Current Working directory :',os.getcwd())  # Prints the current working directory


#%%
def GetModel():
    return ('yolov8m.pt')

def TrainModel(classifications,epochs,imgsize, weights=None, projects =r'C:\apps\Schaeffler\AI-aionthefly-environment\runs\detect'):

    print('YOLO version : ',ultralytics.__version__)
    print('parms',classifications,epochs,imgsize, weights, projects)

    if (weights == None):
        model = YOLO('yolov8s.pt')
        print('yolov8s.pt')
    else:
        model = YOLO(weights)
        print(weights)
    
    tStart = time.time()
    # Train the model on the COCO8 example dataset for 100 epochs
    #results = model.train(data='cat8.yaml', epochs=10, imgsz=640)
    #results = model.train(data='demomix.yaml', epochs=10, imgsz=640)
    results = model.train(data=classifications, epochs=epochs, imgsz=imgsize, name='train', project=projects, exist_ok =True, batch = 16, freeze = None)
    
    #results = model.train(data='demomix.yaml', epochs=200, imgsz=640)

    tEnd = time.time()
    format_float = '{:.3f}'.format(tEnd-tStart)
    print('Train time : ', format_float)
    return format_float, results


#%%

run = True

if __name__ == "__main__":
    print(GetModel())
    print ('Params=', sys.argv, len(sys.argv))
    
    if run:
        if (len(sys.argv) > 0):
            param_1= sys.argv[0] 
            print ('Params=', param_1)
        if (len(sys.argv) > 1):
            classfile= sys.argv[1] 
            print ('classfile=', classfile)
        if (len(sys.argv) > 2):
            epoch= sys.argv[2]  
            print ('epoch=', epoch)

        if (len(sys.argv) > 3):
            imgsize= sys.argv[3]  
            print ('imgsize=', imgsize)
            if (len(sys.argv) < 4):
                format_float, results = TrainModel(classfile, int(epoch), int(imgsize))

        if (len(sys.argv) > 4):
            weightfile= sys.argv[4]  
            print ('Weightfile=', weightfile)
            if (len(sys.argv) < 6):
                format_float, results = TrainModel(classfile, int(epoch), int(imgsize),weightfile)
            else:
                project= sys.argv[5]  
                print ('project=', project)
                format_float, results = TrainModel(classfile, int(epoch), int(imgsize),weightfile, project)    
        else:
        #TrainModel('demomix.yaml', 10, 640)
            #format_float, results = TrainModel(r'D:\AIData\catlytic\carscombo\carscombo.yaml', 10, 640)
            print('hardparams')
            format_float, results = TrainModel(r'N:\AIData\AIonthefly_Pictures\AIonthefly_Pictures\AIonthefly_Pictures.yaml', 11, 640)

    else:

       format_float, results = TrainModel(r'D:\AIData\marketplace\bearings\bearings.yaml', 100, 640,r'C:\apps\Schaeffler\AIOTFInterface\AI-aionthefly-environment\yolo\yolo11m.pt')
       # format_float, results = TrainModel(r'N:\AIData\SLA\SLA2Trainset-5-12-2025\top\top.yaml', 10, 640,r'C:\apps\Schaeffler\AIOTFInterface\AI-aionthefly-environment\yolo\yolov8m.pt')
       #format_float, results = TrainModel(r'N:\AIData\AIonthefly_Pictures\AIonthefly_Pictures\AIonthefly_Pictures.yaml', 2, 640,r'C:\apps\Schaeffler\AIOTFInterface\AI-aionthefly-environment\yolo\yolov8m.pt')

    print('Complete')


    #input("Press enter to exit;")
    #while 1:




# %%
