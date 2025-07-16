# -*- coding: utf-8 -*-
"""
Date:           Version:        Dev:            Comment:
11/13/2024      1.0.0.2         TJG             Fix bug when reloading a model that was already added
05-22-2024      1.0.0.1         TJG             Add auto class detection counter and remove hardcode
10-23-2023      1.0.0.0         TJG             Created orginal project


"""



#%%
from ultralytics import YOLO
import ultralytics
import cv2
import urllib
import numpy as np
import json
import time
import os
from time import sleep
from collections import Counter
from pathlib import Path
from threading import Thread
from time import sleep

#%%
print('ultralytics version : ',ultralytics.__version__)
    

#%%
Version = '1.0.0.2'

class yolodetector:

    

    def __init__(self, model, verbose = False):

        # Create storage list for models
        self.learner_list= []
        self.currmodel = ''
        if model is None:
            print("None model")
        else:
            path = Path(model)
            self.modelname = path.stem
            self.loadmodel( model)
            print('init model',model)

        if (verbose):
            print('ultralytics version : ',ultralytics.__version__)
        '''
        path = Path(model)
        print(path.stem)
        
        #print(self.model)
        
        self.model = YOLO(model)
        self.this_model = {"ID": path.stem , "ModelName":path.stem ,"Model" : self.model};
        self.learner_list.append(self.this_model)
       # print(self.this_model)
        self.this_model = next(filter(lambda x: x['ID'] == path.stem, self.learner_list),"NotFound")
        print(self.this_model)
        # self.classes = ['GoodCap','BadCap','GoodInductor', 'BadInductor']
       '''
    def GetVersion(self):

        return Version

    def loadmodel(self, model):
        path = Path(model)
        print('loadmodel :', path.stem)
      #  print('loadmodel :', model)
        
        self.currmodel = path.stem
        if (len(self.learner_list) < 1):
            self.modelname = path.stem
            self.model = YOLO(model, verbose=False)
            self.this_model = {"ID": path.stem , "ModelName":path.stem ,"Model" : self.model};
            self.learner_list.append(self.this_model)
            print ('Inital Model created : ', path.stem)
            #print('self',type(self.this_model))
        else:
            self.this_model = next(filter(lambda x: x['ID'] == path.stem, self.learner_list),"NotFound")
            #print(self.this_model)
            if self.this_model=="NotFound":
                self.modelname = path.stem
                self.model = YOLO(model, verbose=False)
                self.this_model = {"ID": path.stem , "ModelName":path.stem ,"Model" : self.model};
                self.learner_list.append(self.this_model)
                print('Model Changed and Added to list :',path.stem)
               #print('self',self.this_model)
            elif (self.modelname != self.this_model ["ModelName"]):
                #self.model = self.this_model.Model
                #print('self',self.this_model)
                self.modelname = path.stem
                self.model = self.this_model["Model"]
                #print(self.this_model ["ModelName"], self.this_model ["Model"])
                print('Model ', self.this_model ["ModelName"],' loaded from memory')
            else:
                print('Model ', self.this_model ["ModelName"],' Aready Current')

    def GetCurrentModel(self):


        #print(self.learner_list)
        print(self.currmodel)

        return (self.currmodel)

    def predict(self, img, conf, viewimage = False, verbose = False):
        
        rtnresults = ""
       # goodcaps = 0
       # badcaps = 0
       ## goodinductors = 0
       # badinductors = 0
        
        locboxes = []
        # Predict with the model
        results = self.model(source = img,conf=conf, iou=0.4)  # predict on an image
        
        #results2 = self.model.predict(source = img,conf=conf)
        #print('result2: ',results2)
        
        # Extract the results
      #  results = self.model.track(source = img,conf=0.8, persist=True, classes=[0,1])
        #AIResult data
        list = []
        classlist = []
        AI_data = {"Result": "","AIResults": [list]}
        #print(results)
        #print(type(AI_data))
        class_titles = ["classname","classes", "confidence", "boxes"]
    
        for result in results:   
            

           # if (verbose):
            #    print(result)
            boxes = result.boxes.cpu().numpy()                         # get boxes on cpu in numpy

            for box in boxes:                                          # iterate boxes
               # if (verbose):
                #    print('box :',box)
                r = box.xyxy[0].astype(int)      
               # print (r)                      # get corner points as int
                if (viewimage):
                    cv2.rectangle(img, r[:2], r[2:], (0, 255, 0), 2)   # draw boxes on 

                conf = "{:.3f}".format(box.conf[0])
                value = result.names[int(box.cls[0])] +' (' + conf+ ')' 
                name = result.names[int(box.cls[0])] 
               
                locboxes = r[0].tolist(), r[1].tolist(), r[2].tolist(), r[3].tolist()
                #print(type(r[0].tolist()))
            #    locboxes = r[0].tolist(),2,3,4
                #label = str(self.classes[self.class_ids[i]])
                spec_details = [ name, value, conf, locboxes]
           
                ai_data = dict(zip(class_titles, spec_details))
                list.append(ai_data)

                #Get the name of the classes found
                classlist.append(name)
                if (viewimage):
                    cv2.putText(img,value,(r[2]+12, r[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[0, 255, 0],thickness=2)  
                
               
        if (viewimage):
            cv2.imshow('YOLO V8 Detection', img)
            cv2.waitKey()
    
        #Get the count of classes found
        d = {x:classlist.count(x) for x in classlist}
        
       # rtnresults = 'GoodCaps :'+str(goodcaps)+',Badcaps :'+str(badcaps)+', GoodInductors :'+str(goodinductors)+',BadInductor :'+str(badinductors)
        #convert to JSON
        rtnresults =str(d)
        AI_data["Result"] = rtnresults
        AI_data['model'] = self.currmodel
       # print('Data',AI_data)
        return rtnresults, AI_data 
    
 
#%%   


if __name__ == "__main__":


    model_path = r"models"
    model_extension = r".pt"
    image_extension = r".jpg"

    model_id = 'cars_weights-5-5-2024'
    modelPath = os.path.join(model_path,model_id+ model_extension)
    print(modelPath)
    #yo = yolodetector(r'models/cars_weights-5-1-2024.pt')   
 #   yo = yolodetector(r'models/cars_weights-5-1-2024.pt')  
 #   yo.loadmodel(modelPath)
    #yo.GetCurrentModel()
    
  #  yo = yolodetector(r'models/SLASolder-10-23-2024v11.pt')  
  #  modelPath = os.path.join(model_path,'SLASolder-10-23-2024v11'+ model_extension)
  #  print(modelPath)
  #  yo.loadmodel(modelPath)
 #   yo = yolodetector(r'models/SLALidDetect-11-13-2024.pt')  
  #  modelPath = os.path.join(model_path,'SLALidDetect-11-13-2024'+ model_extension)
  #  print(modelPath)
  #  yo.loadmodel(modelPath)
    #file = r'C:\AIData\catlytic\cars2\captureimage_2024055103606.jpg'
   # modelPath = os.path.join(model_path,'SLASolder-10-23-2024v11'+ model_extension)
   # print(modelPath)
   # yo.loadmodel(modelPath)
    #yo.GetCurrentModel()
    yol = []
    yol.append(yolodetector(None))


   # modelPath = os.path.join(model_path,'SLALidDetect-11-13-2024'+ model_extension)
   # print(modelPath)
   # yo.loadmodel(modelPath)
  #  yo.GetCurrentModel()


 #   yo = yolodetector(r'models/S
    #img = cv2.imread(file)

    #print(type(img))
    #start = time.time()
    #results, AI_data = yo.predict(img, 0.3, False, True)

    #print (results,AI_data)
    #json.dumps(AI_data)
    #end = time.time()
    #print(" Time: {:.2f} ".format((end - start)), " seconds")
 
    
# %%
#start = time.time()
#yo.loadmodel(r'models/cars_weights-5-5-2024.pt')

#results, AI_data = yo.predict(img, 0.3, False, True)
#end = time.time()
#print(" Time: {:.2f} ".format((end - start)), " seconds")
 
# %%
