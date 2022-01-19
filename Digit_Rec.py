###################################################################################
# This is a Python code for Handwritten Multi Digit Recognition With Machine
# Learning created by Soha Boroojerdi in Utah Valley University.
# For more info please see: 
# https://github.com/SohaBoroojerdi/Handwritten-Multi-Digit-Recognition/tree/master
###################################################################################

import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
from PIL import ImageEnhance
from numpy import asarray
import cv2, time
import timeit

np.set_printoptions(linewidth=np.inf)

filename = "train.csv"
count = 0
## To see the data set
data = pd.read_csv(filename).to_numpy()

## Spliting data for testing and training
clf_DT = DecisionTreeClassifier()
clf_RF = RandomForestClassifier(n_estimators=150)
clf_MLP = MLPClassifier(solver='sgd', hidden_layer_sizes=(150, ), random_state=1, activation='relu')

data_part = 20000

## Training dataset
xtrain = data[0:data_part, 1:]
train_label = data[0:data_part, 0]

print("Fitting DT ...")
#start = timeit.timeit()
clf_DT = clf_DT.fit(xtrain,train_label)
#print("\tElapsed time :", timeit.timeit() - start)
print("Fitting RF ...")
#start = timeit.timeit()
clf_RF = clf_RF.fit(xtrain,train_label)
#print("\tElapsed time :", timeit.timeit() - start)
print("Fitting MLP ...")
#start = timeit.timeit()
clf_MLP = clf_MLP.fit(xtrain,train_label)  
#print("\tElapsed time :", timeit.timeit() - start)
print("")
## Testing dataset
xtest = data[data_part:, 1:]
actual_label = data[data_part:,0]

## Calculating accuracy
predictData_DT = clf_DT.predict(xtest)
predictData_RF = clf_RF.predict(xtest)
predictData_MLP = clf_MLP.predict(xtest)

DT_acc = accuracy_score(actual_label,predictData_DT)*100
RF_acc = accuracy_score(actual_label,predictData_RF)*100
MLP_acc = accuracy_score(actual_label,predictData_MLP)*100

print("***********************************************************************************")
print("\t\t\tDT Algorithm Accuracy: %", DT_acc)
print("\t\t\tRF Algorithm Accuracy: %", RF_acc)
print("\t\t\tMLP Algorithm Accuracy: %", MLP_acc)
print("***********************************************************************************")


## Taking sample from the dataset
Smpl_from = int(input("Enter your choice number ...\n\t0 : Exit \n\t1 : Test using MNIST dataset\n\t2 : Load from an input JPEG file\n\t3 : Take an input picture from webcam\n>>> "))

while Smpl_from != 0:
    ## Taking sample from MNIST
    if (Smpl_from == 1):
        while True:
            num = int(input("Enter your choice for testing MNIST dataset ...\n\t-1 : Quit testing MNIST\n>> "))
            if(num == -1):
                #Smpl_from = int(input("Enter \n\t-1 to exit \n\t 1 to get a sample from jpg file \n\t 2 to get a sample from MNIST \n\t 3 to get a sample from webcam: "))
                break
            d = xtest[num]
            d.shape = (28,28)
            ## 255-d makes the background color white
            pt.imshow(255-d,cmap = "gray")
            print(d)
            print("DT predicted number is: ", clf_DT.predict([xtest[num]]),"\n")
            print("RF predicted number is: ", clf_RF.predict([xtest[num]]),"\n")
            print("MLP predicted number is: ", clf_MLP.predict([xtest[num]]),"\n")
            pt.show()

    ## Taking sample from a jpg file
    elif (Smpl_from == 2):
        while True:
            testfile = input("Enter your test file name ...\n>> ")
            #num_digits = 0 #int(input("Enter the number of digits : "))
            #if (testfile == "-1"):
            #    Smpl_from = int(input("Enter \n\t-1 to exit \n\t 1 to get a sample from jpg file \n\t 2 to get a sample from MNIST \n\t 3 to get a sample from webcam: "))
            #    break
            #else:
            try:
                img = Image.open(testfile).convert('L')
                pix = img.load()
                max = 0
                min = 255
                # bipolar adjustment of pixel values
                for i in range(img.size[0]):
                    for j in range(img.size[1]):
                        if max < pix[i,j]:
                            max = pix[i,j]
                        if min > pix[i,j]:
                            min = pix[i,j]
                avg = (max - min) / 2
                for i in range(img.size[0]):
                    for j in range(img.size[1]):
                        if avg < pix[i,j]:
                            pix[i,j] = 255
                        else:
                            pix[i,j] = 0
                # Removing whitespaces
                xmin = ymin = 0
                xmax = img.size[0]-1
                ymax = img.size[1]-1
                state = 0
                for j in range(img.size[1]):
                    blank = True
                    for i in range(img.size[0]):
                        if pix[i,j] == 0:
                            blank = False
                            break
                    if state == 0:
                        if blank == False:
                            ymin = j
                            state = 1
                    elif state == 1:
                        if blank == True:
                            ymax = j
                            break
                state = 0
                for i in range(img.size[0]):
                    blank = True
                    for j in range(img.size[1]):
                        if pix[i,j] == 0:
                            blank = False
                            break
                    if state == 0:
                        if blank == False:
                            xmin = i
                            break
                state = 0
                for i in reversed(range(img.size[0])):
                    blank = True
                    for j in range(img.size[1]):
                        if pix[i,j] == 0:
                            blank = False
                            break
                    if state == 0:
                        if blank == False:
                            xmax = i
                            break
                # Margin calculation
                marg = (ymax-ymin)/10
                ymin -= marg
                ymax += marg
                xmin -= marg
                xmax += marg
                if ymin < 0:
                    ymin = 0
                if xmin < 0:
                    xmin = 0
                if ymax >= img.size[1]:
                    ymax = img.size[1]-1
                if xmax >= img.size[0]:
                    xmax = img.size[0]-1
                img = img.crop((xmin, ymin, xmax, ymax))
                # Detect individual digits
                pix = img.load()
                state = 0
                num_digits = 0
                dig_border = 0
                frame_left = 0
                for i in range(img.size[0]):
                    blank = True
                    for j in range(img.size[1]):
                        if pix[i,j] == 0:
                            blank = False
                            break
                    if state == 0:
                        if num_digits != 0:
                            if (i == (img.size[0]-1)) or (blank == False):
                                dig = img.crop((frame_left, 0, int((i + dig_border)/2), img.size[1]-1))
                                dig = dig.resize((28, 28))
                                #con = ImageEnhance.Sharpness(dig)
                                #con.enhance(0.9)
                                num = asarray(dig)
                                num = 255-num
                                print(num)
                                num = np.reshape(num, 28*28)
                                print("DT Predicted number is: ", clf_DT.predict([num]),"\n")
                                print("RF predicted number is: ", clf_RF.predict([num]),"\n")
                                print("MLP predicted number is: ", clf_MLP.predict([num]),"\n")
                                #num.shape = (28,28)
                                #pt.imshow(255-num,cmap = "gray")
                                #pt.show()

                        if blank == False:
                            frame_left = int((i + dig_border)/2)
                            state = 1
                    elif state == 1:
                        if blank == True:
                            dig_border = i
                            num_digits += 1
                            state = 0
            except IOError:
                pass

            if(input("Do you want to test another file? [Y]\n>> ") != "Y"):
                break

    ## Taking sample from webcam
    elif (Smpl_from == 3):
        #Create an object 
        video = cv2.VideoCapture(0)
        count = 0
        while True:
            #Create a frame
            check, frame = video.read()
            # show the frame
            cv2.imshow("Capturing", frame)
            if not check:
                break
            # Press key to out
            k =cv2.waitKey(1)
            if (k % 256 ==27):
                print("Webcame is closed")
                break
            elif (k % 256 == 32):
                print("Image " + str(count) +" saved")
                imgfile = '/Users/soha/Desktop/Term_Project/digit_recognition_code/img'+str(count)+'.jpg'
                cv2.imwrite(imgfile,frame)
                count +=1
                try:
                    img = Image.open(imgfile).convert('L')
                    pix = img.load()
                    max = 0
                    min = 255
                    for i in range(img.size[0]):
                        for j in range(img.size[1]):
                            if max < pix[i,j]:
                                max = pix[i,j]
                            if min > pix[i,j]:
                                min = pix[i,j]
                    avg = (max - min) / 2
                    for i in range(img.size[0]):
                        for j in range(img.size[1]):
                            if avg < pix[i,j]:
                                pix[i,j] = 255
                            else:
                                pix[i,j] = 0
                    xmin = ymin = 0
                    xmax = img.size[0]-1
                    ymax = img.size[1]-1
                    state = 0
                    for j in range(img.size[1]):
                        blank = True
                        for i in range(img.size[0]):
                            if pix[i,j] == 0:
                                blank = False
                                break
                        if state == 0:
                            if blank == False:
                                ymin = j
                                state = 1
                        elif state == 1:
                            if blank == True:
                                ymax = j
                                break
                    state = 0
                    for i in range(img.size[0]):
                        blank = True
                        for j in range(img.size[1]):
                            if pix[i,j] == 0:
                                blank = False
                                break
                        if state == 0:
                            if blank == False:
                                xmin = i
                                break
                    state = 0
                    for i in reversed(range(img.size[0])):
                        blank = True
                        for j in range(img.size[1]):
                            if pix[i,j] == 0:
                                blank = False
                                break
                        if state == 0:
                            if blank == False:
                                xmax = i
                                break
                    marg = (ymax-ymin)/10
                    ymin -= marg
                    ymax += marg
                    xmin -= marg
                    xmax += marg
                    if ymin < 0:
                        ymin = 0
                    if xmin < 0:
                        xmin = 0
                    if ymax >= img.size[1]:
                        ymax = img.size[1]-1
                    if xmax >= img.size[0]:
                        xmax = img.size[0]-1
                    img = img.crop((xmin, ymin, xmax, ymax))
                    # count num digits
                    pix = img.load()
                    state = 0
                    num_digits = 0
                    dig_border = 0
                    frame_left = 0
                    for i in range(img.size[0]):
                        blank = True
                        for j in range(img.size[1]):
                            if pix[i,j] == 0:
                                blank = False
                                break
                        if state == 0:
                            if num_digits != 0:
                                if (i == (img.size[0]-1)) or (blank == False):
                                    dig = img.crop((frame_left, 0, int((i + dig_border)/2), img.size[1]-1))
                                    dig = dig.resize((28, 28))
                                    #con = ImageEnhance.Sharpness(dig)
                                    #con.enhance(0.9)
                                    num = asarray(dig)
                                    num = 255-num
                                    print(num)
                                    num = np.reshape(num, 28*28)
                                    print("DT Predicted number is: ", clf_DT.predict([num]),"\n")
                                    print("RF predicted number is: ", clf_RF.predict([num]),"\n")
                                    print("MLP predicted number is: ", clf_MLP.predict([num]),"\n")
                                    #num.shape = (28,28)
                                    #pt.imshow(255-num,cmap = "gray")
                                    #pt.show()

                            if blank == False:
                                frame_left = int((i + dig_border)/2)
                                state = 1
                        elif state == 1:
                            if blank == True:
                                dig_border = i
                                num_digits += 1
                                state = 0
                except IOError:
                    pass
                #shutdown the camera
                video.release()
                cv2.destroyAllWindows
                break
    Smpl_from = int(input("Enter your choice number ...\n\t0 : Exit \n\t1 : Test using MNIST dataset\n\t2 : Load from an input JPEG file\n\t3 : Take an input picture from webcam\n>>> "))
