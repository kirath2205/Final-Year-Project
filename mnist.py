import tensorflow as tf
from tensorflow import keras
import numpy as np

def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images=np.expand_dims(train_images,axis=3)
    test_images=np.expand_dims(test_images,axis=3)
    if(training):
        return (train_images,train_labels)
    if(not training):
        return (test_images,test_labels)

def build_model():
    model=tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(64,kernel_size=3,activation="relu",input_shape=(28,28,1)))
    model.add(keras.layers.Conv2D(32,kernel_size=3,activation="relu"))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10,activation='softmax'))
    model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model
    
def train_model(model,train_img,train_lab,test_img,test_lab,T):
    train_lab = keras.utils.to_categorical(train_lab)
    test_lab=keras.utils.to_categorical(test_lab)
    model.fit(train_img,train_lab,epochs=T,validation_data=(test_img,test_lab))

def predict_label(model, images, index):
    k=model.predict(images)
    prediction=(k[index]).copy()
    labels=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    for i in range(len(labels)):
        for j in range(0,len(labels)-1):
            if(prediction[j]<prediction[j+1]):
                t=prediction[j]
                prediction[j]=prediction[j+1]
                prediction[j+1]=t
                t2=labels[j]
                labels[j]=labels[j+1]
                labels[j+1]=t2
    print(labels[0],":",round((prediction[0]*100),2),"%")
    print(labels[1],":",round((prediction[1]*100),2),"%")
    print(labels[2],":",round((prediction[2]*100),2),"%")


train_images,train_labels=get_dataset(True)
test_images,test_labels=get_dataset(False)
model=build_model()
train_model(model,train_images,train_labels,test_images,test_labels,2)






