from rocketml.io import VideoSet,Video,MongoWriter
from rocketml.feature_extraction import ObjectDetection
from rocketml import Pipeline
import subprocess
import cv2
import sys

filename = sys.argv[1]
hname = subprocess.check_output(["hostname"])
mongo_uri = "mongodb://"+hname.strip()+":27017"

vd = VideoSet(filelist=[filename])
od = ObjectDetection()
mw = MongoWriter(uri=mongo_uri,db="rml_video_bbox",collection="test1")

#pp = Pipeline([od,mw])
#pp.fit(vd)


vv = Video(filename)
xll,yll,xur,yur = vv.get_activity_window(mw)
print("%d,%d,%d,%d"%(xll,yll,xur,yur))

cap = cv2.VideoCapture(filename)
cap.set(cv2.CAP_PROP_POS_FRAMES,100)
ret,frame = cap.read()
cv2.rectangle(frame,(xll,yll),(xur,yur),(125, 255, 51), thickness=2)
cap.release()
cv2.imwrite("out_bbox.png",frame)
