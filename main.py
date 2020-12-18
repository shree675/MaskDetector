import cv2

vid = cv2.VideoCapture(0)

ret = vid.set(3, 1280)
ret = vid.set(4, 720)

face_classifier = cv2.CascadeClassifier('haarcascade_frontface_default.xml')
mouth_classifier = cv2.CascadeClassifier('haarcascade_mouth.xml')

while True:

    ret, frame = vid.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.25, 4)
    mouths = mouth_classifier.detectMultiScale(gray, 1.6, 20)

    if faces is ():
        cv2.putText(frame, 'NO FACES DETECTED', (65,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,250), 2)

    for (x,y,w,h) in faces:
        a=0
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 250, 0), 2)          
        for (x1,y1,w1,h1) in mouths:
            if x1>=x and y1>=y and x1<=(x+w) and y1<=(y+h):
                cv2.putText(frame, 'NOT WEARING MASK', (x+w//2,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,250), 2)
                a+=1
        if a==0:
            cv2.putText(frame, 'WEARING MASK', (x+w//2,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,250,0), 2)

    cv2.imshow('Classified', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):               // press q to exit
        break

cv2.destroyAllWindows()
