import numpy as np
from flask import Flask, request, jsonify, render_template
import h5py
import pickle
import cv2
import pandas as pd
import tensorflow
from tensorflow.keras.models import Sequential
import keras
import zipfile
import os
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation, Dense, Reshape, BatchNormalization, Input, Conv2D

import h5py

from tensorflow.keras.models import Model

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer

from numpy import argmax

app = Flask(__name__)
json_file = open('model.json', 'r')
model = json_file.read()
model1 = model_from_json(model)

@app.route('/')
def home():
    
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    files_zip_path = "Images2.zip"


# In[9]:


    file_zip_path = "Images2"
    archive = zipfile.ZipFile(files_zip_path, 'r')
    archive.extractall()
    
    class IdentityMetadata():
        def __init__(self, base, name, file):
        # print(base, name, file)
        # dataset base directory
            self.base = base
        # identity name
            self.name = name
        # image file name
            self.file = file

        def __repr__(self):
            return self.image_path()

        def image_path(self):
            return os.path.join(self.base, self.name, self.file) 
    
    def load_metadata(path):
        metadata = []
        for i in os.listdir(path):
            for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
                ext = os.path.splitext(f)[1]
                if ext == '.jpg' or ext == '.jpeg':
                    metadata.append(IdentityMetadata(path, i, f))
        return np.array(metadata)

# metadata = load_metadata('images')
    metadata = load_metadata('Images2')


# In[12]:


#metadata[0]


# In[13]:



    def load_image(path):
        img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them

        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(im_rgb)
    #plt.show()

        return img[...,::-1]

    x = metadata.size   # I have used first 500 images as loading of around 10000 images keeps crashing my system.

    #img_path3 = []
    #img = [0] * (x)
    #embeddings = [0] * (x)

    #for i in range(x):
  
   #   img_path = metadata[i].image_path()
  #    img[i] = load_image(img_path)
     # img_path3.append(img_path)
    
# Normalising pixel values from [0-255] to [0-1]: scale RGB values to interval [0,1]
 #     img[i] = (img[i] / 255.).astype(np.float32)

#      img[i] = cv2.resize(img[i], dsize = (224,224))
#print(img.shape)

# Obtain embedding vector for an image
# Get the embedding vector for the above image using vgg_face_descriptor model and print the shape 

    #  embeddings[i] = vgg_face_descriptor.predict(np.expand_dims(img[i], axis=0))[0]
      #print(embedding_vector.shape)
    
    data = metadata[0:x]
    
    train_idx = np.arange(x) % 5 != 0
    test_idx = np.arange(x) % 5 == 0
    
    X_train2 = np.array(img)[train_idx]
    X_test2= np.array(img)[test_idx]

    target = np.array([m.name for m in data])

    y_train2 = target[train_idx]
    y_test2 = target[test_idx]


# In[29]:


    X_train2.shape


# In[30]:


    nRows,nCols,nDims = X_train2.shape[1:]
    train_data = X_train2.reshape(X_train2.shape[0], nRows, nCols, nDims)
    test_data = X_test2.reshape(X_test2.shape[0], nRows, nCols, nDims)
    input_shape = (nRows, nCols, nDims)

    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    

# In[31]:


    #train_data.shape, test_data.shape


# In[32]:





# In[33]:


    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(y_train2)
    test_labels = encoder.fit_transform(y_test2)


# In[34]:




    norm_img = np.zeros((224,224))
    train_scaled = cv2.normalize(train_data,  norm_img, 0, 255, cv2.NORM_MINMAX)
    test_scaled = cv2.normalize(test_data, norm_img, 0, 255, cv2.NORM_MINMAX)
#scaler = Normalizer()
#train_scaled = scaler.fit_transform(train_data)
#test_scaled = scaler.fit_transform(test_data)


# In[35]:


    classes = np.unique(train_labels)
    nClasses = len(classes)



# In[36]:


#train_data /= 255
#test_data /= 255

    train_labels_one_hot = to_categorical(train_labels)
    test_labels_one_hot = to_categorical(test_labels)  
    
    
    
    file_path1 = request.form.get('fileupload1')
    image_new1 = cv2.imread('Images_all/' + file_path1,1)
    
    #data = request.get_json
    #data = np.array(data)
    #print(image_new1)
    
    #file_path2 = request.form.get('fileupload2')
    #image_new2 = cv2.imread(file_path2)
    
    norm_img = np.zeros((224,224))
    img_scaled = cv2.normalize(image_new1,  norm_img, 0, 255, cv2.NORM_MINMAX)
        
    img = cv2.resize(img_scaled, dsize = (224,224))
    
    arr4d1= np.expand_dims(img, 0) 
    #arr4d2= np.expand_dims(image_new2, 0) 
    test_img = np.vstack((arr4d1, arr4d1))
    
    
    y_pred = model1.predict(test_img)
    
    #encoder = LabelEncoder()
    
    def prediction(ind):
        predict1 = ind.round()
        predict1_st = np.argmax(predict1, axis =1)
        example_identity1 = encoder.inverse_transform(predict1_st)
    
        return example_identity1

    output = prediction(y_pred)
    
    print("The person is identified as: ", output[0])
    #print("The person is identified as: ", test_img.shape)
    
    return render_template('index.html', prediction_text='The person is identified as: {}'.format(output[0]))

#@app.route('/predict_api',methods=['POST'])
#def predict_api():
 #   '''
  #  For direct API calls trought request
   # '''
    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

 #   output = prediction[0]
#    return jsonify(output)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000)
    app.run(host = '0.0.0.0', port = port, threaded=True, debug=True)