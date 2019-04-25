# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback
import json

import flask
import uuid

import base64
from PIL import Image
import io
from flask import jsonify
import cv2
import numpy as np
import imutils

prefix = '/opt/ml/model/'

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

if not os.path.exists("/tmp/jobs_folder"):
    os.mkdir("/tmp/jobs_folder")

job_folder = "/tmp/jobs_folder"


class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    
    @classmethod
    def get_model(cls):
        return "Nothing to check"

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
                
        imgBarcodeDict={}
        # load the image and convert it to grayscale
        image = cv2.imread(input)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # compute the Scharr gradient magnitude representation of the images
        # in both the x and y direction using OpenCV 2.4
        ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
        gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)

        # subtract the y-gradient from the x-gradient
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # blur and threshold the image
        blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

        # construct a closing kernel and apply it to the thresholded image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations = 4)
        closed = cv2.dilate(closed, None, iterations = 4)

        # find the contours in the thresholded image, then sort the contours
        # by their area, keeping only the largest one
        cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.int32(box)

        # draw a bounding box arounded the detected barcode and display the
        # image
        imgBarcodeDict['box']=box.tolist()
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        output_image = os.path.join(prefix,"output_barcode_boxes.jpeg")
        cv2.imwrite(output_image,image)
        with open(output_image) as f:
            out_image = base64.b64encode(f.read())

        imgBarcodeDict["output"] = out_image
        
        return imgBarcodeDict
       
       

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    data = flask.request.data
    filename = os.path.join(prefix,"imagToSave.png")
    image = Image.open(io.BytesIO(data))
    image.save(filename)
    result = ScoringService.predict(filename)

    return jsonify(result)
