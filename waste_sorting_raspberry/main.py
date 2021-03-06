from platform import python_branch
import sys

sys.path.append('/usr/local/lib/python3.9/site-packages')

# GitHub: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
import cv2

#cv2.dnn.writeTextGraph('wasteSorting.pb', 'wasteSortingGraph.pbtxt')

thres = 0.45 # Threshold to detect object
 
cap = cv2.VideoCapture(1)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
 
classNames= []
classFile = 'waste.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('n').split('n')

''' Code de base
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
'''
weightsPath = 'wasteSorting_pb/wasteSorting.pb'
configPath = 'wasteSorting_pb/wasteSortingGraph.pbtxt'
cv2.dnn.writeTextGraph(weightsPath, configPath)

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
 
while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)
 
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
 
    cv2.imshow("Output",img)
    cv2.waitKey(1)
