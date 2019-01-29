from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import pytesseract
import imutils
from math import atan2,degrees
import locality_aware_nms as nms_locality
import lanms
from PIL import Image
import json
import base64

#tf.app.flags.DEFINE_string('test_data_path', '/home/haofucv-2/AIRBUS/sae_ocr_east/test_images/', '')
#tf.app.flags.DEFINE_string('checkpoint_path', '/home/haofucv-2/AIRBUS/sae_ocr_east/models/model_150ksteps/', '')
#tf.app.flags.DEFINE_string('output_dir', '/home/haofucv-2/AIRBUS/sae_ocr_east/test_results/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')


import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

#
# @brief function to get all images inside a folder directory
# @param 
# @return 
def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


#
# @brief function to resize image  to ensure that the image is in a size multiple of 32 (requirement based on the network architecture)
# @param input image, max size of the image
# @return the resized image, ratio of height being resized and ratio of width being resized
def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

#
# @brief function perform thresholding and non-maximum suppresion based on the score map and geometry map of the bounding box produced by the neural network
# @param score map of the bounding box, geometry map of the bounding box, timer, score map threshold, bounding box threshold and non-maximum supression
# @return the filtered boxes and time taken
def detect(score_map, geo_map, timer, score_map_thresh=0.01, box_thresh=0.01, nms_thres=0.3):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    #print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

#
# @brief function to sort the 4 points of the bounding box in order
# @param 4 points of the bounding box in order
# @return sorted points of the bounding box
def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

#
# @brief function to rotate the image based on it's contours
# @param image
# @return the rotated image
def rotate_image(roi_image):
	gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	edged = cv2.Canny(gray, 20, 100)
	im2,contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if contours != None:
		contours = np.vstack(contours).squeeze()
		
		angle = cv2.minAreaRect(contours)[-1]
		#print(angle)
		if angle < -45:
			angle = -(90 + angle)
		else:
			angle = -angle
	
	if(abs(angle) >= 1): #if the angle is larger than 1. rotate the image
		rotated = imutils.rotate_bound(roi_image, angle)
		# Resize the image back to it's original size
		rotated = cv2.resize(rotated, (roi_image.shape[1], roi_image.shape[0]),0,0, cv2.INTER_LINEAR)
	else:
		rotated = roi_image
	
	return rotated
#
# @brief function to write the decoded information into a json file
# @param output filename, decoded information
# @return 	
def write_to_json_file(output_filename,detected_information):
	# Write to Json file
	detection_flag = False
	if(any("PNR" in info for info in detected_information) and any("SER" in info for info in detected_information)):
		detection_flag = True

	PNR =()
	SER =()
	PNR = [info for info in detected_information if "PNR" in info]	
	SER = [info for info in detected_information if "SER" in info]	
	
	#convert list to string for json response
	global SERN, PNRN
	SERN = str(SER)
	PNRN = str(PNR)

	data = {
		"PNR and SER Detection":str(detection_flag),
		"PNR":str(PNR)[2:-2],
		"SER":str(SER)[2:-2]
	}

	with open(output_filename, 'w') as outfile:
		json.dump(data, outfile)

#
# @brief function to read config file (JSON)
# @param config file's directory
# @return the input image's directory, model checkpoint, output annonated image path and output file containg information of detection
def read_config(config_file):
	with open(config_file) as json_file:  
		data = json.load(json_file)
		input_image = data["Input File"]
		checkpoint_path = data["Checkpoint Path"]
		output_image_path = data["Output Image Path"]
		output_filename = data["Output File"]
		
	return input_image, checkpoint_path, output_image_path, output_filename

# Create your views here.
@api_view(["POST"])

#def main(argv=None):
def checkOCR(OCRdata):

	image_base64=json.loads(OCRdata.body)
	
	imgdata = base64.b64decode(image_base64)
	filename = '../saeOcrDjango/test_images/converted_JSON.jpg'  
	#filename = '../saeOcrDjango/test_images/converted_JSON.jpg'
	with open(filename, 'wb') as f:
		f.write(imgdata)
	
	import os
	os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
	#tesseract_config = ('-l eng --oem 1 --psm 6')
	tesseract_config = ('--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata" -l eng -oem 1 -psm 6')
	
	# Read Config from JSON File
	test_image_directory, checkpoint_path, output_image_path, output_filename = read_config("config.txt")
	tf.app.flags.DEFINE_string('test_data_path', test_image_directory, '')
	tf.app.flags.DEFINE_string('checkpoint_path', checkpoint_path, '')
	tf.app.flags.DEFINE_string('output_dir', output_image_path, '')
	
	try:
		os.makedirs(FLAGS.output_dir)
	except OSError as e:
		if e.errno != 17:
			raise

	with tf.get_default_graph().as_default():
		input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

		f_score, f_geometry = model.model(input_images, is_training=False)

		variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
		saver = tf.train.Saver(variable_averages.variables_to_restore())

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			# Load the checkpoint
			ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
			model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
			print('Restore from {}'.format(model_path))
			saver.restore(sess, model_path)
			
			detection_list = ["SER", "PNR"]
			im_fn_list = get_images()
			for im_fn in im_fn_list:
				im = cv2.imread(im_fn)[:, :, ::-1]
				image_bgr = cv2.imread(im_fn)
				blur = cv2.GaussianBlur(image_bgr,(7,7),0)
												
				im_write = cv2.imread(im_fn)
				im_clone = im_write.copy()
				start_time = time.time()
				im_resized, (ratio_h, ratio_w) = resize_image(blur[:, :, ::-1])

				timer = {'net': 0, 'restore': 0, 'nms': 0}
				start = time.time()
				# Predict the score and geometry map based on the test mage
				score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
				timer['net'] = time.time() - start
				# Threshold and apply non-maximum supression
				boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
				#print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

				if boxes is not None:
					boxes = boxes[:, :8].reshape((-1, 4, 2))
					boxes[:, :, 0] /= ratio_w
					boxes[:, :, 1] /= ratio_h

				duration = time.time() - start_time
				print('----- [scene text detection timing] {:.4f} seconds -----'.format(duration))

				# save to file
				if boxes is not None:
					res_file = os.path.join(
						FLAGS.output_dir,
						'{}.txt'.format(
							os.path.basename(im_fn).split('.')[0]))

					with open(res_file, 'w') as f:
						detected_information = []
						start_time_ocr = time.time()
						for box in boxes:
							# To avoid submitting errors, sort the 4 corners of the box
							box = sort_poly(box.astype(np.int32))
							if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
								continue
							#f.write('{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
							cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
							cv2.polylines(im_write, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
							roi_image = im_clone[box[1, 1]:box[3, 1], box[0, 0]:box[1, 0]]
							#cv2.imwrite('roi_image.jpg',roi_image)
							# Run tesseract OCR on the roi image
							if(roi_image.size!=0):
								#roi_image = rotate_image(roi_image)
								text = pytesseract.image_to_string(roi_image, lang='eng', config=tesseract_config)
								cv2.putText(im_write, text,(box[0,0],box[0,1]+10), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)
								# Print recognized text
								if(text[:3] in detection_list):
									detected_information.append(text)
								#cannot write, temp ignore write to file
								f.write('{},{},{},{},{},{},{},{},{}\r\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1], text))
						ocr_duration = time.time() - start_time_ocr
						print('----- [optical character recognition timing] {:.4f} seconds -----'.format(ocr_duration))
					print('----- [total timing] {:.4f} seconds -----'.format(duration+ocr_duration))
				
				if not FLAGS.no_write_images:
					# Write to Image
					img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
					#cv2.imwrite(img_path, im[:, :, ::-1])
					cv2.imwrite(img_path, im_write)
					# Write to JSON File
					output_file = os.path.join(
						FLAGS.output_dir,
						'{}_data.txt'.format(
							os.path.basename(im_fn).split('.')[0]))
					write_to_json_file(output_file,detected_information)
					return JsonResponse("SER: "+SERN+" ; PNR: "+PNRN,safe=False)
					
if __name__ == '__main__':
    tf.app.run()
