import cv2
import numpy as np
import dlib

def createBox(img,points,scale=5,masked=False,cropped=True):
    if masked:
        mask = np.zeros_like(img)
        mask = cv2.fillPoly(mask,[points],(255,255,255))
        
        img = cv2.bitwise_and(img,mask)
        #cv2.imshow("Mask",img)
    if cropped:
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        return imgCrop
    else:
        return mask
def empty(a):
    pass
cv2.namedWindow('BGR')
cv2.resizeWindow('BGR',440,240)
cv2.createTrackbar("Blue","BGR",0,255,empty)
cv2.createTrackbar("Green","BGR",0,255,empty)
cv2.createTrackbar("Red","BGR",0,255,empty)
while True:
    img = cv2.imread(r"C:/Users/Aqib HOME/Pictures/1.jpg")
    
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    imgOriginal = img.copy()
    
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(imgGray)
    
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        #imgOriginal = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        landmarks = predictor(imgGray,face)
        myPoints = []
        for n in range(68):
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            myPoints.append([x,y])
            #cv2.circle(imgOriginal,(x,y),5,(50,50,255),cv2.FILLED)
            #cv2.putText(imgOriginal,str(n),(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(0,255,0),1)
        
        myPoints = np.array(myPoints)
        #imgLeftEye = createBox(img,myPoints[36:42])
        imgLips = createBox(img,myPoints[48:61],3,masked=True,cropped=False)
        
        imgColorLips = np.zeros_like(imgLips)
        b = cv2.getTrackbarPos('Blue','BGR')
        g = cv2.getTrackbarPos('Green','BGR')
        r = cv2.getTrackbarPos('Red','BGR')
        imgColorLips[:] = b,g,r
        imgColorLips = cv2.bitwise_and(imgLips,imgColorLips)
        imgColorLips = cv2.GaussianBlur(imgColorLips,(7,7),10)
        imgOriginalGray = cv2.cvtColor(imgOriginal,cv2.COLOR_BGR2GRAY)
        imgOriginalGray = cv2.cvtColor(imgOriginalGray,cv2.COLOR_GRAY2BGR)
        imgColorLips = cv2.addWeighted(imgOriginalGray,1,imgColorLips,0.4,0)
        
        cv2.imshow("BGR",imgColorLips)
    #cv2.imshow("Orginial",imgOriginal)
    cv2.waitKey(1)
