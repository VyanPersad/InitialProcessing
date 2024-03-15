#import the necessary packages
#import imutils
import numpy as np
import argparse
import cv2

def skinTone(pic_path):

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help="path to the image file", default=pic_path)
                    
    args = vars(ap.parse_args())

    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([3, 15, 10], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")

    #load image
    image = cv2.imread(args["image"])
    # resize the image to a reasonable width for display (optional)
    image = cv2.resize(image, (300, 300))

    # convert the image to the HSV color space
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # determine the HSV pixel intensities that fall into
    # the specified upper and lower boundaries
    skinMask = cv2.inRange(converted, lower, upper)
    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    # blur the mask to help remove noise
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # apply the mask to the original image
    skin = cv2.bitwise_and(image, image, mask=skinMask)

    #This was just to test
    # display the original image and the skin-detected image side by side
    #cv2.imshow("Original Image", image)
    #cv2.imshow("Skin Detection", np.hstack([image, skin]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Find contours in the skin mask
    contours, _ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialize a list to store bounding boxes of non-black/skin regions
    non_black_boxes = []

    # Loop over the contours
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out small bounding boxes (adjust the threshold as needed)
        # The number is essentially in pixels
        # 100 is an industry standard in image processing
        if cv2.contourArea(contour) > 100:
            non_black_boxes.append((x, y, w, h))
        
    # Create a copy of the original skin image
    skin_cropped = skin.copy()

    # Draw bounding boxes on the copy
    # This only draws the reactangle bounding boxes
    for box in non_black_boxes:
        x, y, w, h = box
        #img output,upper left, lower right, BGR Color, thickness
        cv2.rectangle(skin_cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Determine Largest Contour
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_mask = np.zeros_like(skinMask)
    cv2.drawContours(largest_contour_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)

    # Apply the mask to the original image
    largest_contour_image = cv2.bitwise_and(image, image, mask=largest_contour_mask)
    
    #This specifically writes the image to a file called skin1.png
    cv2.imwrite('skin1.png',largest_contour_image)
    
    #This was just to test.
    #cv2.imwrite('skin2.png',skin_cropped)
    # Display the result
    #cv2.imshow('Non-Black Regions', skin_cropped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



#path_to_pic = 'C:\\Users\\Vyan Persad\\Downloads\\Pictures\\100.jpg'
#skinTone(path_to_pic)

