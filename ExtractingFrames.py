# Importing all necessary libraries 
import cv2 
import os 
import time

# Read the video from specified path 
cam = cv2.VideoCapture('./data/video/test2.mp4')

try: 
	
	# creating a folder named data 
	if not os.path.exists('Frames'): 
		os.makedirs('Frames') 

# if not created then raise error 
except OSError: 
	print ('Error: Creating directory of Frames') 

# frame 
currentframe = 0
t1 = time.time()

while(True): 
	
	# reading from frame 
	ret,frame = cam.read() 

	if ret: 
		# if video is still left continue creating images 
		name = './Frames/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name) 

		# writing the extracted images 
		cv2.imwrite(name, frame) 

		# increasing counter so that it will 
		# show how many frames are created 
		currentframe += 1
	else: 
		break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 
# print(time.time())
print((float(time.time()) - float(t1)))