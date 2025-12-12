import os

import keras
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


nombre_de_classe=7
img_rows,img_columns=48,48
batch_size=32

data_entrainement="C:/Users/eyaka/OneDrive/Desktop/expressions/train"
data_test="C:/Users/eyaka/OneDrive/Desktop/expressions/test"

generer_nv_images = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

validation_img = ImageDataGenerator(rescale=1./255)



entrainement=generer_nv_images.flow_from_directory(data_entrainement,color_mode='grayscale',target_size=(img_rows,img_columns),batch_size=batch_size,class_mode='categorical',shuffle=True)


validation=validation_img.flow_from_directory(data_test,color_mode='grayscale',target_size=(img_rows,img_columns),batch_size=batch_size,class_mode='categorical',shuffle=True)


model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_columns,1)))
mod0el.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))



model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))



model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))



model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))



model.add(Dense(nombre_de_classe,kernel_initializer='he_normal'))
model.add(Activation('softmax'))
print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
checkpoint=ModelCheckpoint('Emotion_little_vgg.h5',
                           monitor='val_loss',
                           mode='min',
                           save_best_only=True,
                           verbose=1)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=1e-6),
    ModelCheckpoint('best_emotion_model.h5', save_best_only=True)
]


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])

element_entrainement=24172
elem_validation=3006
epochs=60

entrainement=model.fit(
                      entrainement,
                      steps_per_epoch=element_entrainement//batch_size,
                      epochs=epochs,
                      callbacks=callbacks,
                      validation_data=validation,
                      validation_steps=elem_validation//batch_size
                      )
