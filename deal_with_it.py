import cv2


eyes    = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
face    = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
image   = cv2.imread("glasses.png", -1) 
capture = cv2.VideoCapture(0)

def deal_with_it(eyes_cascade, face_cascade, glasses_img):
    '''live meme generator'''

    #mask
    glasses_mask     = glasses_img[:, :, 3]
    glasses_mask_inv = cv2.bitwise_not(glasses_mask)

    glasses_img = glasses_img[:, :, 0:3]
    orig_glasses_height, orig_glasses_width = glasses_img.shape[:2]

    while (True):
        # Capture video feed
        unused, img = capture.read()
        frame       = cv2.flip(img, 1)
    
        # grayscale
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    
        for (x, y, w, h) in faces:

            roi_gray  = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eyes_cascade.detectMultiScale(roi_gray)
    
            for (nx,ny,nw,nh) in eyes: 
                
                glasses_width  =  nw * 4
                glasses_height = orig_glasses_height * 2
                area = glasses_width * glasses_height
    
                # possition
                x1 = nx - glasses_width * 2
                x2 = nx + nw + glasses_width * 2
                y1 = ny + nh - glasses_height
                y2 = ny + nh + glasses_height
    
                # black magic
                if x1 < 0:
                    x1 = 0
                if x2 > w:
                    x2 = w
                if y1 < 0:
                    y1 = 0                
                if y2 > h:
                    y2 = h
    
                # adjust bassed on black magic
                glasses_height = y2 - y1
                glasses_width  = x2 - x1
                    
                # rescale
                glasses  = cv2.resize(glasses_img, (glasses_width,glasses_height), interpolation = cv2.INTER_AREA)
                mask     = cv2.resize(glasses_mask, (glasses_width,glasses_height), interpolation = cv2.INTER_AREA)
                mask_inv = cv2.resize(glasses_mask_inv, (glasses_width,glasses_height), interpolation = cv2.INTER_AREA)
    
                # some more black magic
                roi = roi_color[y1:y2, x1:x2]
    
                # difference between regions of interest
                roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)    
                roi_fg = cv2.bitwise_and(glasses,glasses,mask = mask)
    
                # combine regions
                dst = cv2.add(roi_bg,roi_fg)               
                roi_color[y1:y2, x1:x2] = dst

                break
    
        # video feed
        cv2.imshow("Deal With It @ SalemStateMemes.edu", frame)
    
        #exit
        if cv2.waitKey(1) == 27: #esc key
            break  
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
	deal_with_it(eyes, face, image)