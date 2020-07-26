import cv2,time


face=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile=cv2.CascadeClassifier("smile.xml")

video=cv2.VideoCapture(0)

while True:
    check,frame=video.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    f=face.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in f:
        img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        s=smile.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=20)
        for x, y, w, h in f:
            img=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)


    cv2.imshow('gotcha',frame)
    key=cv2.waitKey(1)

    if key==ord('q'):  #order
        break

video.release()
cv2.destroyAllWindows



