import base64, os
from django.shortcuts import render
from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser 
from django.http import FileResponse
from rest_framework import status
from base64 import b64encode
from PIL import Image
import numpy as np
import cv2 as cv
import subprocess

from rest_framework.response import Response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import matplotlib.pyplot as plt
 
from .models import Tutorial
from .serializers import TutorialSerializer
from rest_framework.decorators import api_view

# from IPython.display import HTML, Image
# import wget
import shutil
from shutil import copyfile


@api_view(['GET', 'POST'])
def imageMerge(request):
    
    print(request.method)
    if request.method == 'POST':
        value = request.FILES['imagebg']
        value1 = request.FILES['imagefg']
        print(request.POST)
        print("Image",value)
        print("Image1",value1)

        imagebg = Image.open(value).convert('RGBA')
        image = Image.open(value1).convert('RGBA')
        print(imagebg.size)
        print(image.size)
        
        new_im= Image.new('RGBA', imagebg.size, (0, 0, 0, 0))
        new_im.paste(imagebg, (0,0))
        new_im.paste(image, (0,0), mask=image)

        name = 'newIm.png'
        new_im.save(name, format="png")

        response = FileResponse(open(name, 'rb'))
        print('New Image',new_im)
        
        return response


@api_view(['POST'])
def imagePartRemove(request):
    if request.method == 'POST':
        print(request.method)
        imageVal = request.FILES['image']
        maskVal = request.FILES['mask']

        imagePIL = Image.open(imageVal).convert('RGB')
        newMaskPIL = Image.open(maskVal).convert('RGB')

        image = np.array(imagePIL)
        newMask = np.array(newMaskPIL)

        image =image[:, :, ::-1].copy()
        newMask =newMask[:, :, ::-1].copy()

        # image = cv.imread(imageVal)
        # cv.imshow("Person", image)
        # cv.waitKey(0)

        # newMask = cv.imread(maskVal)
        mask = cv.cvtColor(newMask, cv.COLOR_BGR2GRAY)

        # cv.imshow('Mask', mask)
        # cv.waitKey(0)


        masked = cv.bitwise_and(image, image, mask=mask)
        # cv.imshow("Mask Applied to Image", masked)
        # cv.waitKey(0)

        name = 'masked.png'
        cv.imwrite(name, masked)

        response = FileResponse(open(name, 'rb'))
        # print('New Image',masked)
        
        return response


@api_view(['POST' , 'GET'])
def imageDetectObject(request):
    if request.method == 'POST':
        print(request.method)
        imageVal = request.FILES['image']
        
        imagePIL = Image.open(imageVal).convert('RGB')
        img = imagePIL.resize((224, 224))
        
        model=load_model('model_inception.h5')
        
        x=image.img_to_array(img)
        x=x/255
        

        x=np.expand_dims(x,axis=0)
        img_data=preprocess_input(x)
        print(img_data.shape)
        
        a=np.argmax(model.predict(img_data), axis=1)

        if a[0]==0:
            value = 'ANT'
            print("ANT")
        elif a[0]==1:
            value = 'BEE'
            print("BEE")
        elif a[0]==2:
            value = 'BUG'
            print("BUG")
        elif a[0]==3:
            value = 'BUTTERFLY'
            print("BUTTERFLY")

        response = { 'category': value }
        return Response(response)

    elif request.method == 'GET':
        response = {'value': 'GET API'}


        # name = 'masked.png'
        # cv.imwrite(name, masked)

        # response = FileResponse(open(name, 'rb'))
        
        return Response(response)



@api_view(['POST' , 'GET'])
def removeObject(request):
    if request.method == 'POST':
        print(request.method)
        imageVal = request.FILES['image']
        maskVal = request.FILES['mask']


        imagePIL = Image.open(imageVal).convert('RGB')
        newMaskPIL = Image.open(maskVal).convert('RGB')

        image = np.array(imagePIL)
        newMask = np.array(newMaskPIL)

        image =image[:, :, ::-1].copy()
        newMask =newMask[:, :, ::-1].copy()

        name = 'image.png'
        cv.imwrite(name, image)

        mask = cv.cvtColor(newMask, cv.COLOR_BGR2GRAY)

        # Setting path for storing our test image   
        shutil.rmtree('./data_for_prediction', ignore_errors=True)
        os.mkdir('./data_for_prediction')

        # copying file to our test folder and removing it from our current OS folder
        copyfile(name, f'./data_for_prediction/{name}')
        os.remove(name)
        name = f'./data_for_prediction/{name}'
        
        image64 = base64.b64encode(open(name, 'rb').read())
        image64 = image64.decode('utf-8')

        print(f'Will use {name} for inpainting')
        img = np.array(plt.imread(f'{name}')[:,:,:3])

        img = np.array((1-mask.reshape(mask.shape[0], mask.shape[1], -1))*plt.imread(name)[:,:,:3])
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        cv.imshow('mix', img)
        cv.waitKey(0)


        subprocess.call([r'C:\Users\muhammadannasasif\Desktop\Python Files\restfulApi\pythonRun.bat'])
        # print('Run inpainting')
        # if '.jpeg' in fname:
        #     print()
        #     !PYTHONPATH=. TORCH_HOME=$(pwd) python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=/content/output dataset.img_suffix=.jpeg > /dev/null
        # elif '.jpg' in fname:
        #     print('-------------2----------------')
        #     !python3 ./lama/bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=/content/output dataset.img_suffix=.jpg > /dev/null
        #     print('-------------2----------------')
        # elif '.png' in fname:
        #     print('-------------3----------------')
        #     !PYTHONPATH=. TORCH_HOME=$(pwd) python3 bin/predict.py model.path=$(pwd)/big-lama indir=$(pwd)/data_for_prediction outdir=/content/output dataset.img_suffix=.png > /dev/null
        # else:
        #   print(f'Error: unknown suffix .{fname.split(".")[-1]} use [.png, .jpeg, .jpg]')
        
        
        # response = FileResponse(open(name, 'rb'))
        # # print('New Image',masked)
        
        # return response
        cv.destroyAllWindows()
        response = {'value': 'GET API'}
        
        return Response(response)


