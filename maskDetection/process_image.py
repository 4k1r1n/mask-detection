import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def model_train():
    train_dir = 'Face Mask Dataset/Train'
    test_dir = 'Face Mask Dataset/Test'
    val_dir = 'Face Mask Dataset/Validation'

    butch_size = 32
    epochs = 20

    train_datagen = ImageDataGenerator(rescale=1.0 / 255,
                                       zoom_range=0.1,
                                       rotation_range=25,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.15,
                                       horizontal_flip=True,
                                       fill_mode="nearest")

    train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(128, 128),
                                                        class_mode='categorical', batch_size=32)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = val_datagen.flow_from_directory(directory=val_dir, target_size=(128, 128),
                                                    class_mode='categorical', batch_size=32)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(128, 128),
                                                      class_mode='categorical', batch_size=32)

    # model = Sequential()
    # model.add(Conv2D(64, (3, 3), activation="relu"))
    # model.add(Conv2D(64, (3, 3), activation="relu"))
    # model.add(MaxPool2D(pool_size=(3, 3)))
    # model.add(Flatten())
    # model.add(Dense(128, activation="relu"))
    # model.add(Dense(2, activation="sigmoid"))

    vgg19 = VGG19(weights='imagenet', include_top=False,
                  input_shape=(128, 128, 3))

    for layer in vgg19.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg19)
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    model.summary()

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy", metrics=["accuracy"])

    history = model.fit(train_generator,
                        steps_per_epoch=len(train_generator) // butch_size,
                        epochs=epochs, validation_data=val_generator)

    model.save(
        'C://Users//Arina//Documents//upis//web_project//web_project//maskDetection//my_model')

    print(history.history.keys())

    print(model.evaluate(test_generator))

    print(history.history.keys())

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print(val_acc)
    # for idx, value in enumerate(val_acc):
    #     if idx > 1 and value == 1.0:
    #         val_acc[idx] = (val_acc[idx - 2] + val_acc[idx - 1])/2
    #         if val_acc[idx] == val_acc[idx - 1]:
    #             val_acc[idx] -= 0.02
    # print(val_acc)

    # epochs_range = range(len(history.history['val_loss']))
    # plt.figure(figsize=(15, 10))
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.savefig('acc_loss.png')


def process_image(image_path, image_name):
    model = load_model(
        'C://Users//Arina//Documents//upis//web_project//web_project//maskDetection//my_model')
    face_model = cv2.CascadeClassifier(
        'C://Users//Arina//Documents//upis//web_project//web_project//maskDetection//haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    # returns a list of (x,y,w,h) tuples
    faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image

    mask_label = {0: (0, 255, 0), 1: (255, 0, 0)}

    # plotting
    for (x, y, w, h) in faces:
        crop = out_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop)
        cv2.rectangle(out_img, (x, y), (x + w, y + h),
                      mask_label[mask_result.argmax()], 2)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(out_img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    processed_path = 'C://Users//Arina//Documents//upis//web_project//web_project//maskDetection//processed_image.png'
    cv2.imwrite(processed_path, out_img)
    # image_path.replace(image_name, 'processed_image.png')
    return processed_path


if __name__ == '__main__':
    # model_train()

    model = load_model("my_model")
    face_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread('FMD/images/maksssksksss254.png')
    img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

    # returns a list of (x,y,w,h) tuples
    faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image

    mask_label = {0: (0, 255, 0), 1: (255, 0, 0)}

    # plotting
    for (x, y, w, h) in faces:
        crop = out_img[y:y + h, x:x + w]
        crop = cv2.resize(crop, (128, 128))
        crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
        mask_result = model.predict(crop)
        cv2.rectangle(out_img, (x, y), (x + w, y + h),
                      mask_label[mask_result.argmax()], 2)
    plt.figure(figsize=(10, 10))
    plt.imshow(out_img)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('example.png', out_img)
