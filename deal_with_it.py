import cv2 # import OpenCV

# import classifiers and image of sunglasses to be used
eye     = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
face    = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
image   = cv2.imread("glasses.png", cv2.IMREAD_UNCHANGED)  # second parameter includes the alpha channel
capture = cv2.VideoCapture(0) # selects video source

def adjust_glasses(h, eyes_w, eyes_h):
    ''' helper function that computes values to scale/translate glasses '''

    # derive initial glasses possition/scale from eyes
    v1 = eyes_h * (-1)
    v2 = eyes_h * 5
    u1 = eyes_w * (-1)
    u2 = eyes_w * 8

    # normalize values
    u1 = max(0, u1) 
    v1 = max(0, v1)
    u2 = min(h, u2)    
    v2 = min(h, v2)

    return v1, v2, u1, u2

def deal_with_it(eyes_cascade, face_cascade, glasses_img):
    ''' function detects faces and places the classic meme sunglasses on the detected face(s)! '''

    # creates mask
    glasses_mask     = cv2.split(glasses_img)[3]     # extracts alpha channel to be used as mask
    glasses_mask_inv = cv2.bitwise_not(glasses_mask) # inverts original mask

    # Strips alpha channel from glasses
    glasses_img = cv2.cvtColor(glasses_img, cv2.COLOR_RGBA2RGB)   

    while (True):
        
        img   = capture.read()[1] # Capture webcam's video feed
        video = cv2.flip(img, 1)  # Flips image horizontally to compensate for webcam's mirrored image

        # detectes faces within given frame and saves them to a list
        faces = face_cascade.detectMultiScale(video)
        
        for (x, y, w, h) in faces:

            # extract window/kernel of face region
            face_region = video[y:y+h, x:x+w]

            # detectes eyes within each face and saves them to a list
            eyes = eyes_cascade.detectMultiScale(face_region)
    
            for (_, _, eyes_w, eyes_h) in eyes:

                v1, v2, u1, u2 = adjust_glasses(h, eyes_w, eyes_h)

                # resize glasses and masks
                glasses  = cv2.resize(glasses_img,      (u2 - u1, v2 - v1))
                mask     = cv2.resize(glasses_mask,     (u2 - u1, v2 - v1))
                mask_inv = cv2.resize(glasses_mask_inv, (u2 - u1, v2 - v1))
    
                # selects mask's area from within the face region
                roi = face_region[v1:v2, u1:u2]
    
                # applies masks
                foreround  = cv2.bitwise_and(glasses, glasses, mask = mask)
                background = cv2.bitwise_and(roi, roi, mask = mask_inv)
    
                # combines fg and bg of the eyes/glasses region by adding them              
                face_region[v1:v2, u1:u2] = cv2.add(foreround, background) 

                break
    
        # displays video in a new window
        cv2.imshow("Sunglass-o-matic", video)
    
        # The Escape key has to be pressed to exit the window
        if cv2.waitKey(1) == 27:
            break  
    
    capture.release() # releases the video source device
    cv2.destroyAllWindows() 

if __name__ == '__main__':
    deal_with_it(eye, face, image)