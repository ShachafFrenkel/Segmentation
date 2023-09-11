# import all required packages:

import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import random
from scipy import ndimage
import tifffile
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
import tensorflow.keras as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator


print('-'*100)
# function for data importing from dataset
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str,
                        dest='data_path',
                        default='data',
                        help='data folder mounting point')

    return parser.parse_args()


if __name__ == '__main__':

    # parse the parameters passed to the this script
    args = parse_args()
    # all the images for training and the their segmentation
    image_1 = os.path.join(args.data_path, 'position3-frame_41.tif')
    image_2 = os.path.join(args.data_path, 'position3-frame_13.tif')
    image_3 = os.path.join(args.data_path, 'position3-frame_29.tif')
    image_4 = os.path.join(args.data_path, 'position3-frame_111.tif')
    image_5 = os.path.join(args.data_path, 'position3-frame_162.tif')
    image_6 = os.path.join(args.data_path, 'position3-frame_80.tif')
    image_7 = os.path.join(args.data_path, 'position3-frame_140.tif')
    image_8 = os.path.join(args.data_path, 'position3-frame_189.tif')
    image_9 = os.path.join(args.data_path, 'position3-frame_1.tif')
    image_10 = os.path.join(args.data_path, 'position3-frame_55.tif')
    image_11 = os.path.join(args.data_path, 'position3-frame_85.tif')
    image_labels_1 = os.path.join(args.data_path, 'position3_seg_rotate-frame_41.tif')
    image_labels_2 = os.path.join(args.data_path, 'position3_seg-frame_13_rotate.tif')
    image_labels_3 = os.path.join(args.data_path, 'position3_seg-frame_29_rotate.tif')
    image_labels_4 = os.path.join(args.data_path, 'position3_seg-frame_111.tif')
    image_labels_5 = os.path.join(args.data_path, 'position3_seg-frame_162.tif')
    image_labels_6 = os.path.join(args.data_path, 'position3_seg-frame_80.tif')
    image_labels_7 = os.path.join(args.data_path, 'position3_seg-frame_140.tif')
    image_labels_8 = os.path.join(args.data_path, 'position3_seg-frame_189.tif')
    image_labels_9 = os.path.join(args.data_path, 'position3_seg-frame_1.tif')
    image_labels_10 = os.path.join(args.data_path, 'position3_seg-frame_55.tif')
    image_labels_11 = os.path.join(args.data_path, 'position3_seg-frame_85.tif')

    image_data = [image_1,image_2,image_3,image_4,image_5,image_6,image_7,image_8,image_9,image_10]
    label_data = [image_labels_1,image_labels_2,image_labels_3,image_labels_4,image_labels_5,image_labels_6,image_labels_7,image_labels_8,image_labels_9,image_labels_10]


    # Create ImageGenerators
    def image_generator_train( bs, mode="train", aug=None):
        while True:
            n=0
            images_label = []
            images = []
            i=0
            while len(images_label) < bs:
                if mode == "eval":
                    break
                # pick a random integer for selecting a random image in the dataset:
                rand_image = random.randint(0,9)
                # pick a random slice of the image at a random angle:
                rand_int_1 = random.randint(1,1600)
                rand_int_2 = random.randint(1,1600)
                rand_angle = random.choice([0,90,180,270])
                # normalization and slicing, slicing both image and segmentation the same:
                image = tifffile.imread(image_data[rand_image])
                image_l = tifffile.imread(label_data[rand_image])
                m = np.max(image[:,rand_int_1:rand_int_1+256,rand_int_2:rand_int_2+256])
                single_image = np.divide(image[:,rand_int_1:rand_int_1+256,rand_int_2:rand_int_2+256],m)
                single_image_label = image_l[1,rand_int_1:rand_int_1 + 256, rand_int_2:rand_int_2 + 256]
                single_image_rotate = ndimage.rotate(single_image,rand_angle,axes=(2,1),reshape=False)
                single_image_label_rotate = ndimage.rotate(single_image_label, rand_angle,axes=(1,0),reshape=False)
                single_image_rotate = np.transpose(single_image_rotate).reshape((256, 256,2))
                single_image_label_rotate = np.transpose(single_image_label_rotate.reshape((256,256)))
                # creating "one-hot" labeling instead of 0,1,2 labels in the original segmentation:
                single_image_label_2channels = np.zeros((2, 256, 256))
                single_image_label_2channels[0][single_image_label_rotate == 1] = 1
                single_image_label_2channels[1][single_image_label_rotate == 2] = 1
                single_image_label_2channels = single_image_label_2channels.transpose((1,2,0))
                # add for training only images with enough non-zero pixels:
                count_of_boundaries = np.count_nonzero(single_image_label_rotate == 0)
                norm_count = count_of_boundaries/(256*256)
                if norm_count<0.3:
                    images.append(single_image_rotate)
                    images_label.append(single_image_label_2channels)
                print("train", i)
                i+=1
                print(norm_count)
            n += 1
            yield tf.convert_to_tensor(np.array(images), dtype=None, dtype_hint=None, name=None),tf.convert_to_tensor(np.array(images_label), dtype=None, dtype_hint=None, name=None)
    def image_generator_test( bs, mode="train", aug=None):
        while True:
            n=0
            images_label = []
            images = []
            i=0
            while len(images_label) < bs:
                if mode == "eval":
                    break
                rand_int_1 = random.randint(1,1600)
                rand_int_2 = random.randint(1,1600)
                rand_angle = random.choice([0,90,180,270])
                # normalization and slicing randomly:
                image = tifffile.imread(image_11)
                image_l = tifffile.imread(image_labels_11)
                m = np.max(image[:,rand_int_1:rand_int_1+256,rand_int_2:rand_int_2+256])
                single_image = np.divide(image[:,rand_int_1:rand_int_1+256,rand_int_2:rand_int_2+256],m)
                single_image_label = image_l[1,rand_int_1:rand_int_1 + 256, rand_int_2:rand_int_2 + 256]
                single_image_rotate = ndimage.rotate(single_image,rand_angle,axes=(2,1),reshape=False)
                single_image_label_rotate = ndimage.rotate(single_image_label, rand_angle,axes=(1,0),reshape=False)
                # was first 1
                single_image_rotate = np.transpose(single_image_rotate).reshape((256, 256,2))
                # single_image = single_image.reshape((128, 128,2))
                single_image_label_rotate = np.transpose(single_image_label_rotate.reshape((256,256)))
                single_image_label_2channels = np.zeros((2, 256, 256))
                single_image_label_2channels[0][single_image_label_rotate == 1] = 1
                single_image_label_2channels[1][single_image_label_rotate == 2] = 1
                single_image_label_2channels = single_image_label_2channels.transpose((1,2,0))
                count_of_boundaries = np.count_nonzero(single_image_label_rotate == 0)
                norm_count = count_of_boundaries/(256*256)
                if norm_count<0.3:
                    images.append(single_image_rotate)
                    images_label.append(single_image_label_2channels)
                print("train", i)
                i+=1
            n += 1
            yield tf.convert_to_tensor(np.array(images), dtype=None, dtype_hint=None, name=None),tf.convert_to_tensor(np.array(images_label), dtype=None, dtype_hint=None, name=None)
    print('Creating train ImageDataGenerator')
    # rotation - bigger
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	    horizontal_flip=True, fill_mode="nearest")
# initialize both the training and testing image generators
    # BS = Batch size, initial training was on 32, but 64 is better.
    BS = 64
    trainGen = image_generator_train( BS,
	    mode="train", aug=aug)
    testGen = image_generator_test( BS,
	    mode="train", aug=None)

    # Build the model
    def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
        x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        x= BatchNormalization(axis=-1)(x)
   # Conv2D then ReLU activation
        x = Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
        x= BatchNormalization(axis=-1)(x)
        return x

    def downsample_block(x, n_filters):
        f = double_conv_block(x, n_filters)
        p = MaxPool2D(2)(f)
        p = Dropout(0.3)(p)
        return f, p

    def upsample_block(x, conv_features, n_filters):
   # upsample
        x = Conv2DTranspose(n_filters, 3, 2, padding="same")(x)

   # concatenate
        x = concatenate([x, conv_features])
   # dropout
        x = Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
        x = double_conv_block(x, n_filters)
        return x


    def build_unet_model(InputShape):
        inputs = Input(shape=InputShape)
        f1, p1 = downsample_block(inputs, 128)
        f2, p2 = downsample_block(p1, 256)
        f3, p3 = downsample_block(p2,512)
        bottleneck = double_conv_block(p3, 1024)
        u1 = upsample_block(bottleneck,f3,512)
        u2 = upsample_block(u1,f2,256)
        u3 = upsample_block(u2, f1, 128)
        outputs = Conv2D(2,1,padding="same",activation="softmax")(u3)
        unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
        return unet_model
    # input size must be specified here-
    unet_model = build_unet_model((256,256,2))

    # choice of optimizer! I tried both.

    #unet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    #              loss="categorical_crossentropy",
    #              metrics="accuracy")

    unet_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4), loss="mse",
                                metrics=tf.keras.metrics.MeanIoU(num_classes=2))

    # fit model and store history
    NUM_EPOCHS = 50
    NUM_TRAIN_IMAGES = 1024
    NUM_TEST_IMAGES = 1024
# STEPS_PER_EPOCH = TRAIN_LENGTH // BS
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = BS // VAL_SUBSPLITS
    model_history = unet_model.fit(trainGen,
                              epochs=NUM_EPOCHS,
                              steps_per_epoch=NUM_TRAIN_IMAGES // BS,
                              validation_steps=NUM_TEST_IMAGES // BS,
                              validation_data=testGen)
    unet_model.save(f'outputs/model_image_segmentation_GPU_run')
    print('Saving model history...')
    with open(f'outputs/model.history', 'wb') as f:
        pickle.dump(model_history.history, f)
    print('Saving model history...')
    unet_model.save(f'outputs/model.model')
    N = NUM_EPOCHS
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), model_history.history["loss"], label="train_loss")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    # plt.ylim(0,min(H.history["loss"])*10)
    plt.legend(loc="lower left")
    plt.savefig(f'outputs/plot_train_loss.png')
    plt.figure()
    plt.plot(np.arange(0, N), model_history.history["val_loss"], label="val_loss")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.ylim(0, min(model_history.history["loss"]) * 10)
    plt.legend(loc="lower left")
    plt.savefig(f'outputs/plot_val_loss.png')
    print('Done!')
    print('-'*100)