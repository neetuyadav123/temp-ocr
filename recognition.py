
from preprocess import preprocess, detect_text, localize
from predictor import prediction
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def recognition(gray_image, show ="show"):
    segments, template, th_img, text_color = preprocess(gray_image)
    labels = []
    accuracy = []
    show_img = gray_image[:]
    print(len(segments))
    
    for segment in segments: 
        plt.imshow(segment)
        plt.show()
        recimg, bimg = detect_text(show_img, th_img, segment, text_color)
        label, sure = prediction(bimg)
        if(sure > 80):
            print(segment)
            labels.append(str(label))
            accuracy.append(sure)
            show_img = localize(show_img, th_img, segment, text_color, show)
        char = labels
    accuracy = np.average(accuracy)
    char = ''.join(char)
    if accuracy < 80:
        recimg, bimg = detect_text(show_img, th_img, template, text_color)
        show_img = localize(show_img, th_img, template, text_color, show)
        char, accuracy = prediction(bimg)
        
    if (show == 'show'):
        plt.imshow(show_img)
        plt.title('Detecting')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        cv2.imshow('Detecting..', cv2.cvtColor(show_img, cv2.COLOR_GRAY2BGR))
    
    print('The prediction accuracy for ', char,' is ',"%.2f" % round(accuracy,2), '%')
    
    #plt.imshow(cv2.cvtColor(show_img, cv2.COLOR_GRAY2RGB))
    #plt.show()

if __name__ == "__main__":
    test = "tes1.png"
    test = cv2.imread(test, 0)
    plt.imshow(cv2.cvtColor(test, cv2.COLOR_GRAY2RGB))
    plt.xticks([])
    plt.yticks([])
    time1 = time.time()
    in_img = recognition(test, 'show')
    print("In %f" %(time.time()-time1), 'sec')