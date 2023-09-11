
import argparse
import tifffile
import os
import numpy as np
from keras.models import load_model
from scipy import ndimage
import tensorflow as tf
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
import os
import skimage as skl

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

    image_path = os.path.join(args.data_path, 'position3-frame_88.tif')
    image = tifffile.imread(image_path)
    image_label_path = os.path.join(args.data_path, 'position3_seg-frame_88.tif')
    image_label = tifffile.imread(image_label_path)
    print(image_label.shape)
    image_label = np.transpose(image_label)
    m = np.max(image)
    single_image = np.divide(image, m)
    print(single_image.shape)
    single_image_reshaped = np.transpose(single_image).reshape((1, image.shape[1], image.shape[2], 2))
    shape1 = single_image_reshaped.shape[1]
    shape2 = single_image_reshaped.shape[2]
    for i in range(shape1):
        if 2 ** i >= shape1:
            first_axis_pixels = 2 ** i
            break
    for j in range(shape2):
        if 2 ** j >= shape2:
            second_axis_pixels = 2 ** j
            break

    new_shape = [first_axis_pixels, second_axis_pixels]
    npad = ((0, 0), (new_shape[0] - shape1, 0), (new_shape[1] - shape2, 0), (0, 0))
    padded_image = np.pad(single_image_reshaped, npad)
    # plt.imshow(padded_image[0,:,:,0])
    # plt.show()
    # Build the model
    def double_conv_block(x, n_filters):
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization(axis=-1)(x)
        # Conv2D then ReLU activation
        x = Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
        x = BatchNormalization(axis=-1)(x)
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
        # inputs
        inputs = Input(shape=InputShape)
        f1, p1 = downsample_block(inputs, 128)
        f2, p2 = downsample_block(p1, 256)
        f3, p3 = downsample_block(p2, 512)
        bottleneck = double_conv_block(p3, 1024)
        u1 = upsample_block(bottleneck, f3, 512)
        u2 = upsample_block(u1, f2, 256)
        u3 = upsample_block(u2, f1, 128)
        outputs = Conv2D(2, 1, padding="same", activation="softmax")(u3)
        unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
        return unet_model


    new_model = build_unet_model((padded_image.shape[1], padded_image.shape[2], 2))
    weight_path = os.path.join(args.data_path, 'model_image_segmentation_run_after_load_weights_previous_retrained_7_BS128.h5')
    new_model.load_weights(weight_path)
    print(new_model.summary())
    prediction_labels = new_model.predict(padded_image)
    new_model.save(f'outputs/model_image_segmentation_GPU_run')
    print('Saving model history...')
    new_model.save(f'outputs/model.model')
    tifffile.imsave(f'outputs/predicted_labels_0.tif',prediction_labels[:,:,:,0])
    tifffile.imsave(f'outputs/predicted_labels_1.tif',prediction_labels[:,:,:,1])
    unpadded_predictions = prediction_labels[:, npad[1][0]:, npad[2][0]:, :]
    tifffile.imsave(
        f'outputs/predicted_labels_0_unpadded.tif',
        unpadded_predictions[:, :, :, 0]*255)
    tifffile.imsave(
        f'outputs/predicted_labels_1_unpadded.tif',
        unpadded_predictions[:, :, :, 1]*255)
    segmentation = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    SC = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    SC[unpadded_predictions[0, :, :, 1] > 0.8] = 2
    SC[unpadded_predictions[0, :, :, 0] < 0.1] = 2
    tifffile.imsave(f'outputs/segmentation_prediction_seg_SC.tif', SC)
    HC = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    HC[unpadded_predictions[0, :, :, 0] > 0.8] = 1
    HC[unpadded_predictions[0, :, :, 1] < 0.8] = 1
    tifffile.imsave(f'outputs/segmentation_prediction_seg_HC.tif', HC)
    Boundaries = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    Boundaries[unpadded_predictions[0, :, :, 0] > 0.2] = 0
    Boundaries[unpadded_predictions[0, :, :, 1] < 0.8] = 0
    tifffile.imsave(f'outputs/segmentation_prediction_seg_BOUNDARIES.tif', Boundaries)
    seg = HC+SC+Boundaries
    tifffile.imsave(f'outputs/segmentation_prediction_seg_new.tif', seg)
    #segmentation[unpadded_predictions[0, :, :, 0] <= 0.2] = 0
    #segmentation[unpadded_predictions[0, :, :, 0] <= 0.2] = 0
    #segmentation[unpadded_predictions[0, :, :, 0] > 0.8] = 1
    #segmentation[unpadded_predictions[0, :, :, 1] > 0.7] = 2

    # segmentation[unpadded_predictions[0, :, :, 0] > 0.4] = 0
    # segmentation[unpadded_predictions[0, :, :, 1] < 0.8] = 0
    # segmentation[unpadded_predictions[0, :, :, 0] > 0.8] = 1
    # segmentation[unpadded_predictions[0, :, :, 1] < 0.8] = 1
    # segmentation[unpadded_predictions[0, :, :, 1] > 0.8] = 2
    # segmentation[unpadded_predictions[0, :, :, 0] < 0.1] = 2

    segmentation[(unpadded_predictions[0, :, :, 0] > 0.4) & (unpadded_predictions[0, :, :, 1] < 0.8)] = 0
    segmentation[(unpadded_predictions[0, :, :, 0] > 0.8) & (unpadded_predictions[0, :, :, 1] < 0.8)] = 1
    segmentation[(unpadded_predictions[0, :, :, 1] > 0.8) & (unpadded_predictions[0, :, :, 0] < 0.1)] = 2
    tifffile.imsave(f'outputs/segmentation_prediction_segmentation_new.tif', segmentation)
    print(segmentation.shape)
    print(image_label.shape)
    HC_B = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    HC_B[(unpadded_predictions[0, :, :, 0])>0.1]=255
    SC = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    SC[(unpadded_predictions[0,:,:,1])>0.1]=255
    kernel_ero = np.ones((5, 5), np.uint8)
    kernel_dil = np.ones((5, 5), np.uint8)
    img_erosion = skl.morphology.erosion(segmentation, kernel_ero)
    img_area_close = skl.morphology.area_closing(segmentation)
    img_close = skl.morphology.closing(segmentation)
    img_dilation = skl.morphology.dilation(HC_B, kernel_dil)
    seg_del_eros = skl.morphology.erosion(img_dilation, kernel_dil)
    for i in range(0,100):
        img_dilation = skl.morphology.dilation(seg_del_eros, kernel_dil)
        seg_del_eros = skl.morphology.erosion(img_dilation, kernel_dil)


    sobel_img = skl.filters.sobel(img_erosion)
    sobel_img= (np.ones(sobel_img.shape)+np.ones(sobel_img.shape))-sobel_img
    # sobel_img = np.invert(sobel_img)
    water_shed_img = skl.segmentation.watershed(segmentation)
    segmentation = segmentation.astype(int)
    print("max sobel",np.max(sobel_img))
    print("min sobel",np.min(sobel_img))
    print("max close",np.max(img_area_close))
    print("min close",np.min(img_area_close))
    remove_hole_img = skl.morphology.remove_small_holes(segmentation,area_threshold=3)
    sobel_close_img = sobel_img+img_area_close


    tifffile.imsave(f'outputs/segmentation_prediction_holes.tif', remove_hole_img)
    tifffile.imsave(f'outputs/segmentation_prediction_sobel.tif', sobel_img)
    tifffile.imsave(f'outputs/segmentation_prediction_watershed.tif', water_shed_img)
    tifffile.imsave(f'outputs/segmentation_prediction_close.tif', img_close)
    tifffile.imsave(f'outputs/segmentation_prediction_erode.tif', img_erosion)
    tifffile.imsave(f'outputs/segmentation_prediction_area_close.tif', img_area_close)
    tifffile.imsave(f'outputs/segmentation_prediction_erode_dilate.tif', seg_del_eros)
    tifffile.imsave(f'outputs/segmentation_prediction.tif', segmentation)
    tifffile.imsave(f'outputs/segmentation_true.tif', image_label[:,:,1])
    tifffile.imsave(f'outputs/segmentation_prediction_sobel_close.tif', sobel_close_img)
    seg_new = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    seg_new[(seg_del_eros > 0.4) & (unpadded_predictions[0, :, :, 1] < 0.9)] = 0
    seg_new[(seg_del_eros > 0.8) & (unpadded_predictions[0, :, :, 1] < 0.8)] = 1
    seg_new[(unpadded_predictions[0, :, :, 1] > 0.8) & (seg_del_eros < 0.1)] = 2
    tifffile.imsave(f'outputs/segmentation_prediction_segmentation__dil_ero.tif', seg_new)
    tifffile.imsave(f'outputs/segmentation_prediction_segmentation_SC.tif', SC)

    seg_new_NEW = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    # mark boundaries:
    seg_new_NEW[(seg_del_eros > 0.4) & (SC < 0.9)] = 0
    # mark HC:
    seg_new_NEW[(seg_del_eros > 0.8) & (SC < 0.8)] = 1
    # mark SC:
    seg_new_NEW[(SC > 0.8) & (seg_del_eros < 0.1)] = 2
    tifffile.imsave(f'outputs/segmentation_prediction_segmentation_SEG_NEW.tif', seg_new_NEW)

    # watershed try:
    # watershed_on_predictions_1 = skl.segmentation.watershed(SC, watershed_line = True)
    # tifffile.imsave(f'outputs/watershed_1.tif', watershed_on_predictions_1)
    # watershed_on_predictions_0 = skl.segmentation.watershed(seg_del_eros, watershed_line=True)
    # tifffile.imsave(f'outputs/watershed_0.tif', watershed_on_predictions_0)
    # tifffile.imsave(f'outputs/seg_del_eros.tif', seg_del_eros)
    # tifffile.imsave(f'outputs/SC.tif', SC)
    # boundary = np.zeros((unpadded_predictions.shape[1], unpadded_predictions.shape[2]))
    # boundary[watershed_on_predictions_0 == 0] = 1
    # boundary[watershed_on_predictions_1 == 0] = 1
    # tifffile.imsave(f'outputs/boundaries_watershed.tif',boundary)

    pre_watershed =