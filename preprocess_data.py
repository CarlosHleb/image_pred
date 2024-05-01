
import tensorflow as tf
import os
import numpy as np

def pre_process():
    # Load and order images in array
    images_path = os.environ["WORKING_DIR"] +"train/images/"
    images = os.listdir(images_path)
    ordered = [int(f[:-4]) for f in images]
    ordered.sort()
    ordered = [str(f)+".png" for f in ordered]

    # Make features and labels
    def img_preprocess(path):
        imd = tf.image.decode_png(tf.io.read_file(path), channels=1)
        #imd = tf.image.resize(imd, [360, 640])
        imd = tf.cast(imd, tf.float32)
        imd = imd / 255
        return imd

    frames_per_batch = int(os.environ["FRAMES_PER_BATCH"])
    data_split = int(os.environ["DATA_SPLIT"])
    train_data = [[]]
    train_labels = [[]]
    test_data = [[]]
    test_labels = [[]]
    train_batch_i = 0
    test_batch_i = 0

    for idx, im in enumerate(ordered):
        imdx = int(im[:-4])
        imdx_hundreds = imdx
        if imdx > 1000:
            imdx_hundreds = int(im[1:-4])
            
        if imdx in [300, 1300, 2300, 3300, 4300]:
            continue

        if (idx != 0 and idx % frames_per_batch == 0) or imdx in [1001, 2001, 3001, 4001]:
            if imdx_hundreds >= data_split:
                test_batch_i += 1
                test_data.append([])
                test_labels.append([])
            else:
                train_batch_i += 1
                train_data.append([])
                train_labels.append([])

        if imdx_hundreds >= data_split:
            test_data[test_batch_i].append(img_preprocess(images_path + im))
            test_labels[test_batch_i].append(img_preprocess(images_path + str(imdx_hundreds+1) + im[-4:]))
        else:
            train_data[train_batch_i].append(img_preprocess(images_path + im))
            train_labels[train_batch_i].append(img_preprocess(images_path + str(imdx_hundreds+1) + im[-4:]))

    # Remove batches with wrong size
    def remove_wrong_length_batches(l):
        idx_to_remove = []
        for idx, f in enumerate(l):
            if len(f) != frames_per_batch:
                idx_to_remove.append(idx)
        for i, idx in enumerate(idx_to_remove):
            l.pop(idx-i)
        return l
    
    test_data = remove_wrong_length_batches(test_data)
    test_labels = remove_wrong_length_batches(test_labels)

    print("frames per batch: ", len(train_data[0]))
    print("Shape: ", train_data[0][0].shape)
    print("Train feature length: ", len(train_data))
    print("Train label length: ", len(train_labels))
    print("Test Feature length: ", len(test_data))
    print("Test Labels length: ", len(test_labels))   

    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    return dataset, train_data, train_labels, test_data, test_labels
