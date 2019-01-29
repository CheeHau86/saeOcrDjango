import sys
import os,glob
import cv2
import numpy as np
import xmltodict

xml_folder = './training xml files/'

files = glob.glob(xml_folder)
os.chdir(xml_folder)
# Read image from folder
for f in glob.glob("*.xml"):
	with open(xml_folder + f) as fd:
		doc = xmltodict.parse(fd.read())
		filename = f[:-3]
		file = open(filename + 'txt', 'w')
		for i in range(0, len(doc['annotation']['object'])):
			#print(doc['annotation']['object'][i]['polygon']['pt'][0])
			
			if(int(doc['annotation']['object'][i]['deleted']) == 0):
			
				to_write_string = ('%s,%s,%s,%s,%s,%s,%s,%s,%s' % (doc['annotation']['object'][i]['polygon']['pt'][0]['x'],
																  doc['annotation']['object'][i]['polygon']['pt'][0]['y'],
																  doc['annotation']['object'][i]['polygon']['pt'][1]['x'],
																  doc['annotation']['object'][i]['polygon']['pt'][1]['y'],
																  doc['annotation']['object'][i]['polygon']['pt'][2]['x'],
																  doc['annotation']['object'][i]['polygon']['pt'][2]['y'],
																  doc['annotation']['object'][i]['polygon']['pt'][3]['x'],
																  doc['annotation']['object'][i]['polygon']['pt'][3]['y'],
																  doc['annotation']['object'][i]['name'].upper()))
				file.write(to_write_string+'\n') 
			
	file.close() 
	