import sys
import os,glob
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

image_folder = '/home/haofucv-2/AIRBUS/EAST/Radome Labels'

#path = os.path(image_folder, '*g')
files = glob.glob(image_folder)
os.chdir(image_folder)
# Read image from folder
for f in glob.glob("*.JPG"):
	im = cv2.imread(image_folder+"/"+f, cv2.IMREAD_COLOR)
	image_resize = cv2.resize(im,(1280,720))
	cv2.imwrite(f,image_resize)

