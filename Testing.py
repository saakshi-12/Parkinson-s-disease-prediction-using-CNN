import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk



window = tk.Tk()

window.title("Dr.Diagnosis ")

##window.geometry("500x510")
##window.configure(background ="lavender")
window.geometry('1550x850')

img=Image.open("demo.jpg")
img=img.resize((1550,850))
bg=ImageTk.PhotoImage(img)
a=tk.Label(window,image=bg)
a.place(x=0,y=0)

title = tk.Label(text="Click below to choose picture for testing disease....", background = "black", fg="white", font=("", 30))
title.grid()


def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'Parkinson-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data


    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    #tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))

        status='Not Defied'
        if np.argmax(model_out) == 0:
            str_label = 'No Parkinson'
        elif np.argmax(model_out) == 1:
            str_label = 'Parkinson'
        message.config(text='Status: '+str_label)
        message.place(x=600,y=400)

        
def analysiss(fi):
    t="o"
    if fi=="y":
        t="Parkinson....."
    elif fi=="n":
        t="No parkinson"
    #message.config(text='Status: '+str_label)
    #message.place(x=600,y=400)
    b = tk.Button(text=t, background = "black", fg="white", font=("times", 18))
    b.place(x=700,y=550)


def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    
    fileName = askopenfilename(initialdir='test/', title='Select image for analysis ',filetypes=[('image files', '.jpg')])
    dst = "testpicture"
    print(fileName)
    img_name=os.path.split(fileName)[-1]
    print (os.path.split(fileName)[-1])
    print("")
    fi=img_name[0]
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    load=load.resize((500,500))
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="500", width="500")
    img.image = render
    img.place(x=30, y=30)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    #button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysiss(fi), background = "black", fg="white", font=("times", 18))
    button2.place(x=400,y=550)
button1 = tk.Button(text="Get Photo", command = openphoto, background = "black", fg="white", font=("times", 18))
button1.place(x=200,y=550)

message = tk.Label(text="", background="lightgreen",
                           fg="Brown", font=("times", 25))

window.mainloop()



