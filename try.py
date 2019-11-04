import cv2
import numpy as np

cap=cv2.VideoCapture(0)
min_match=10
detect=cv2.xfeatures2d.SIFT_create()
detector=cv2.AKAZE_create()

flann_index_kdittre=0
flann_param=dict(algorithm=flann_index_kdittre, tree=5)
search_param=dict(checks=50)
flann=cv2.FlannBasedMatcher(flann_param,search_param)

train_img=cv2.imread("/home/vanessa/Documents/VISION/PRACTICA31/tajin.jpeg",0)
(trainKP,trainDesc)=detector.detectAndCompute(train_img,None)



while True:
	_ret , QueryImgBGR=cap.read()

	QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
	(queryKP,queryDesc)=detector.detectAndCompute(QueryImg,None)
	matches=flann.knnMatch(np.asarray(queryDesc,np.float32),np.asarray(trainDesc,np.float32),2)

	goodMatch=[]
	for m,n in matches:
		if(m.distance < 0.7*n.distance):
			goodMatch.append(m)
	print(len(goodMatch))
	if (len(goodMatch)>min_match):
		tp=[]
		qp=[]
		for m in goodMatch:
			tp.append(trainKP[m.trainIdx].pt)
			qp.append(queryKP[m.queryIdx].pt)
		tp,qp=np.float32((tp,qp))
		H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
		h,w=train_img.shape
		trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
		queryBorder=cv2.perspectiveTransform(trainBorder,H)
		cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
	else:
		print("no matches")

	cv2.imshow('result',QueryImgBGR)
	if cv2.waitKey(1)==27:
		break

