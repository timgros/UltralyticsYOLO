
"""
Date:           Version:        Dev:            Comment:
11/17/2024      1.0.0.2         TJG             Fix non thread safe operation
                                                Add classifyYOLOv8_1,2,3,4 to allow parallel REST operation.
                                                Will change later with a better solution, but short term this should work.
05-22-2024      1.0.0.1         TJG             Remove hardcoded weights and load them when called from interface
                                                Add auto load model if not loaded.
10-23-2023      1.0.0.0         TJG             Created orginal project


"""


#from flask import Flask, request, jsonify
#%%
#import logging
#logging.basicConfig(filename="backend.log", level=logging.DEBUG, format="%(asctime)s:%(levelname)s:%(message)s")


from PIL import Image
import base64
import flask
from flask import Flask, redirect, url_for, render_template,jsonify, make_response, request                                      

import cv2
import io
import os
import time
import json
#import yolodetector
import yolodetector as YO
import numpy as np
import glob
from threading import Thread
from time import sleep
import threading

print('Loaded')
print('Flask : ',flask.__version__)

MAXMODELS = 4
#%%
cwd = os.getcwd()
print (cwd)

#%%
print ('Current Working directory :',os.getcwd())  # Prints the current working directory

os.chdir(r'C:\apps\Schaeffler\AI-aionthefly-environment\flaskserver\flaskserverY8')  # Provide the new path here

print ('New Current Working directory :',os.getcwd())  # Prints the current working directory



#%%

version = '1.0.0.2'
model_path = r"models"
model_extension = r".pt"
image_extension = r".jpg"

lock = threading.Lock()

#yo = yolodetector.YOLO8deploy(r'C:\git\yolodetector/best.pt')
#yo = YO.yolodetector(r'models/cars_weights-5-1-2024.pt')



yo = YO.yolodetector(r'models/demomix-6-13-2024.pt')   
yol = []
for x in range(MAXMODELS):
    yol.append(YO.yolodetector(None))

print(len(yol))
app = Flask(__name__)

def _read_data_to_bytes(data):
    img_binary = io.BytesIO(data)
    return img_binary

def allowed_content_type(content_type):
    allowed_types = ['application/octet-stream']
    return content_type in allowed_types

def make_tree(path):
    tree = dict(name=path, children=[])
    try: lst = os.listdir(path)
    except OSError:
        pass #ignore errors
    else:
        for name in lst:
            fn = os.path.join(path, name)
            if os.path.isdir(fn):
                tree['children'].append(make_tree(fn))
            else:
                tree['children'].append(dict(name=fn))
    return tree

@app.route('/')
def index():
    return '''Server Works!<hr>
<form action="/processing" method="POST" enctype="multipart/form-data">
<input type="file" name="image">
<button>OK</button>
</form>    
'''
@app.route('/ping', methods=['GET','POST'])
def ping():
    status = 200 #if learner else 404
    print('pong')
    return flask.Response(response='pong', status=status, mimetype='application/json')


@app.route('/version', methods=['GET','POST'])
def version():
    status = 200 #if learner else 404
    print(version)
    return flask.Response(response='1.0.0.1', status=status, mimetype='application/json')

   # return "ping severside"
@app.route('/processing', methods=['POST'])
def process():
    status = 200
    file = request.files['image']
  #  print('Rx: ',file)
    
    img = Image.open(file.stream)
    
    data = file.stream.read()
    #data = base64.encodebytes(data)
    data = base64.b64encode(data).decode()   

    return jsonify({
                'msg': 'success', 
                'size': [img.width, img.height], 
                'format': img.format,
                'img': data
           })
    
@app.route('/proimage', methods=['POST'])
def proimage():
    file = request.files['image']
    
    img = Image.open(file.stream)
    print(type(img))
    img = img.convert('L')   # ie. convert to grayscale

    #data = file.stream.read()
    #data = base64.b64encode(data).decode()
    
    buffer = io.BytesIO()
    img.save(buffer, 'png')
    buffer.seek(0)
    
    data = buffer.read()
    data = base64.b64encode(data).decode()

    return f'<img src="data:image/png;base64,{data}">'

@app.route('/models', methods=['GET'])
def getmodels():
    status = 200 
    #print(glob.glob(model_path+'/*.pt'))
    #return flask.Response(response='1.0.0.1', status=status, mimetype='application/json')
    return flask.Response(response=json.dumps(glob.glob(model_path+'/*.pt')), status=status, mimetype='application/json')

@app.route('/classifyYOLOv8/<model_id>', methods=['POST'])
def classify_YOLOv8_image(model_id):
    
    start = time.time()
    print('Requested model ', model_id)
    if not flask.request.data:
        return flask.Response(response='No Data available', status=400, mimetype='text/plain')
    if not allowed_content_type(flask.request.content_type):
        return flask.Response(response='The inference only accepts binary data via application/octet-stream',
                              status=415, mimetype='text/plain')
    
    lock.acquire()

    img_binary = _read_data_to_bytes(flask.request.data)
    
    
    img_binary.seek(0)
    
  
    img = Image.open(img_binary).convert("RGB") #<class 'PIL.Image.Image'>

 #   print ('Type:', type(img))
    modelPath = os.path.join(model_path,model_id+ model_extension)
    print(modelPath)
    yo.loadmodel(modelPath)
    prediction = 'NOK results'
    prediction, aidata =yo.predict(img, 0.3, False, True)
 #   print(type(aidata))
    end = time.time()
    print(" Time: {:.2f} ".format((end - start)), " seconds")
    lock.release()
 #   JsonLogger('InferenceApi').event(
 #       'InferenceSuccessful',
  #      {'model': this_model['ModelName'], 'prediction': prediction}
  #  )
    return flask.Response(response=json.dumps(aidata), status=200, mimetype='application/json')


@app.route('/classifyYOLOv8_1/<model_id>', methods=['POST'])
def classify_YOLOv8_1_image(model_id):
    
    start = time.time()
    print('Requested model ', model_id)
    if not flask.request.data:
        return flask.Response(response='No Data available', status=400, mimetype='text/plain')
    if not allowed_content_type(flask.request.content_type):
        return flask.Response(response='The inference only accepts binary data via application/octet-stream',
                              status=415, mimetype='text/plain')
    
  
    img_binary = _read_data_to_bytes(flask.request.data)
    
    
    img_binary.seek(0)
    
  
    img = Image.open(img_binary).convert("RGB") #<class 'PIL.Image.Image'>

 #   print ('Type:', type(img))
    modelPath = os.path.join(model_path,model_id+ model_extension)
    print(modelPath)
    yol[0].loadmodel(modelPath)
   # yo.loadmodel(modelPath)
    prediction = 'NOK results'
    #prediction, aidata =yo.predict(img, 0.3, False, True)
    prediction, aidata = yol[0].predict(img, 0.4, False, True)
    
 #   print(type(aidata))
    end = time.time()
    print(" Time: {:.2f} ".format((end - start)), " seconds")

    return flask.Response(response=json.dumps(aidata), status=200, mimetype='application/json')

@app.route('/classifyYOLOv8_2/<model_id>', methods=['POST'])
def classify_YOLOv8_2_image(model_id):
    
    start = time.time()
    print('Requested model ', model_id)
    if not flask.request.data:
        return flask.Response(response='No Data available', status=400, mimetype='text/plain')
    if not allowed_content_type(flask.request.content_type):
        return flask.Response(response='The inference only accepts binary data via application/octet-stream',
                              status=415, mimetype='text/plain')
    
    img_binary = _read_data_to_bytes(flask.request.data)
    img_binary.seek(0)
    img = Image.open(img_binary).convert("RGB") #<class 'PIL.Image.Image'>
    modelPath = os.path.join(model_path,model_id+ model_extension)
    print(modelPath)
    yol[1].loadmodel(modelPath)
    prediction = 'NOK results'
    prediction, aidata = yol[1].predict(img, 0.4, False, True)
    
    end = time.time()
    print(" Time: {:.2f} ".format((end - start)), " seconds")

    return flask.Response(response=json.dumps(aidata), status=200, mimetype='application/json')

@app.route('/classifyYOLOv8_3/<model_id>', methods=['POST'])
def classify_YOLOv8_3_image(model_id):
    
    start = time.time()
    print('Requested model ', model_id)
    if not flask.request.data:
        return flask.Response(response='No Data available', status=400, mimetype='text/plain')
    if not allowed_content_type(flask.request.content_type):
        return flask.Response(response='The inference only accepts binary data via application/octet-stream',
                              status=415, mimetype='text/plain')
 
    img_binary = _read_data_to_bytes(flask.request.data)
    
    img_binary.seek(0)
  
    img = Image.open(img_binary).convert("RGB") #<class 'PIL.Image.Image'>

    modelPath = os.path.join(model_path,model_id+ model_extension)
    print(modelPath)
    yol[2].loadmodel(modelPath)
    prediction = 'NOK results'
    prediction, aidata = yol[2].predict(img, 0.3, False, True)
    
    end = time.time()
    print(" Time: {:.2f} ".format((end - start)), " seconds")
    return flask.Response(response=json.dumps(aidata), status=200, mimetype='application/json')

@app.route('/classifyYOLOv8_4/<model_id>', methods=['POST'])
def classify_YOLOv8_4_image(model_id):
    
    start = time.time()
    print('Requested model ', model_id)
    if not flask.request.data:
        return flask.Response(response='No Data available', status=400, mimetype='text/plain')
    if not allowed_content_type(flask.request.content_type):
        return flask.Response(response='The inference only accepts binary data via application/octet-stream',
                              status=415, mimetype='text/plain')
 
    img_binary = _read_data_to_bytes(flask.request.data)
    
    img_binary.seek(0)
  
    img = Image.open(img_binary).convert("RGB") #<class 'PIL.Image.Image'>

    modelPath = os.path.join(model_path,model_id+ model_extension)
    print(modelPath)
    yol[3].loadmodel(modelPath)
    prediction = 'NOK results'
    prediction, aidata = yol[3].predict(img, 0.3, False, True)
    
    end = time.time()
    print(" Time: {:.2f} ".format((end - start)), " seconds")
 
    return flask.Response(response=json.dumps(aidata), status=200, mimetype='application/json')


#%%
if __name__ == '__main__':
    print('starting server on 0.0.0.0')
    app.run(host='0.0.0.0', debug=False, port=5000, threaded=True)


# %%
