from keras.applications.vgg16 import VGG16
import matplotlib.image as mpimg
from keras import backend as K
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2
from keras.models import load_model
import numpy as np
import os
import re
import sys
from get_dataset_n import load_csv_dataset
import argparse
import tensorflow as tf 
import time
from cam_fire import create_csv


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def performance_calculate(gt_name, heatmap):
    name = "/home/bamaral/thesis/AnnotDatasets/Smoke_dataset/gt/" + str(gt_name[0:4]) + "_gt_smk.png"
    gt_img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
    # gt_img = cv2.resize(gt_img,(500,500))
    # gt_img[np.where(gt_img < 0.5)] = 0
    # gt_img[np.where(gt_img > 0.5)] = 1

    # heatmap = cv2.resize(heatmap,(500,500))
    # heatmap[np.where(heatmap < 0.5)] = 0
    # heatmap[np.where(heatmap > 0.5)] = 1

    miou = tf.keras.metrics.MeanIoU(num_classes=2)
    miou.reset_states()
    miou.update_state(gt_img/255,heatmap)
    
    # jacc = jaccard_score(gt_img/255,heatmap, average= "micro") #, average="micro")
    jacc =1
    return miou.result().numpy(), jacc

def cam(folder_path, model_name, path_test, alpha, method, perc):
    
    images_np = sorted_alphanumeric(os.listdir(path_test))
    K.clear_session()
    model = load_model(folder_path+model_name)
    model.summary()
    out_class = 1

    perf_csv = [["image", "miou","jacc"]]
    miou_list = []
    jacc_list = []
    time_list = []

    for img in images_np:
        print(img)
        img_path=path_test+img
        # img_name = img[0:4]
        img_name = img[:-4]
        original_img = cv2.imread(img_path)

        # output_orig = folder_path+"cam/"+str(j)+'_in.png'
        # cv2.imwrite(output_orig, original_img)

        width, height, _ = original_img.shape
        imag = np.array(np.float32(original_img[:,:,:]))
        img= np.array(cv2.resize(imag,(256,256)))

        img2 = image.load_img(img_path, target_size=(256, 256))

        x = image.img_to_array(img2)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)
        # preds = predict(model,x)
        preds = model.predict(x)
        start_time = time.time()

        # Get the 512 input weights
        class_weights = model.layers[-1].get_weights()[0]

        final_conv_layer = get_output_layer(model, "block5_conv3")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([x])
        # print("PRED",predictions)
        conv_outputs = conv_outputs[0, :, :, :]

        #Create the class activation map
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        for p, w in enumerate(class_weights[:, out_class]):
            # if w > 0:
            cam += w * conv_outputs[:, :, p]

        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))

        if method == "max":
            tresh = alpha *np.max(cam)
        if method == "perc":
            tresh = alpha * np.percentile(cam,perc)
        time_list.append(time.time() - start_time)
        # print("--- %s seconds ---" % (time.time() - start_time))

        # Create cam heatmap and add it to original image
        heatmap_cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap_cam[np.where(cam < 0.1)] = 0
        # output_cam = folder_path+"compare_smoke/cam_max_02/img/"+img_name+'_out_2.png'
        output_cam = "/home/bamaral/thesis/gestosa_all/PIC_0381_CAM/out/"+ img_name + "_out_2.png"

        img = heatmap_cam*0.8 + original_img
        cv2.putText(img,"Smoke: "+str(predictions[0][0]),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2)
        # cv2.imwrite(output_cam, img)
        # cv2.imwrite(folder_path+"compare_smoke/cam_max_02/img/"+img_name+'_mask.png',heatmap_cam)

        
        #CAM segmented 
        heatmap_seg = np.uint8(cam)
        heatmap_seg[np.where(cam < tresh)] = 0
        heatmap_seg[np.where(cam >= tresh)] = 1
        # output_seg = folder_path+"compare_smoke/cam_max_02/mask/"+img_name+'_cam_mask.png'
        output_seg = "/home/bamaral/thesis/gestosa_all/PIC_0381_CAM/cam/"+img_name+'_cam_mask.png'
        cv2.imwrite(output_seg, 255*heatmap_seg)

        # CAM segmented on image
        heatmap_seg_img = cv2.cvtColor(heatmap_seg ,cv2.COLOR_GRAY2RGB)
        img_mask = original_img + heatmap_seg_img*0.3*255
        # output_seg_img = folder_path+"compare_smoke/cam_max_02/img/"+img_name+'_seg_img.png'
        output_seg_img = "/home/bamaral/thesis/gestosa/PIC_0381_CAM/"+img_name+'_seg_img.png'
        # cv2.imwrite(output_seg_img, img_mask)
        # print("--- %s seconds ---" % (time.time() - start_time))

    #     miou, jacc = performance_calculate(img_name, heatmap_seg)

    #     miou_list.append(miou)
    #     jacc_list.append(jacc)
    # print("mIoU: ", np.mean(miou_list))
    # print("Jacc: ", np.mean(jacc_list))
    # print("Average Time: ",np.mean(time_list))
    # final_csv = [alpha, method, perc, np.mean(miou_list),np.mean(jacc_list),np.std(miou_list),np.std(jacc_list)]    # create_csv(folder_path,"miou_jacc",final_csv)
    
    final_csv = []

    return final_csv

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def cam_main(folder_path,model_name,alpha, method, perc): 
    # try:
    #     os.makedirs(folder_path+'compare_smoke/cam_max_02')
    # except OSError:
    #     print("folder")

    # try:
    #     os.mkdir(folder_path+'compare_smoke/cam_max_02/img')        
    # except OSError:
    #     print("Folder")

    # try:
    #     os.mkdir(folder_path+'compare_smoke/cam_max_02/mask')        
    # except OSError:
    #     print("folder")

    # path_test= "/home/bamaral/thesis/AnnotDatasets/Smoke_dataset/img/"
    path_test= "/home/bamaral/thesis/gestosa_all/PIC_0381/"

    final_csv = cam(folder_path, model_name, path_test, alpha, method, perc)

    return final_csv

if __name__ == '__main__':
    folder_path = "/home/bamaral/thesis/smoke_model/"
    model_name = "smoke_adam_5.h5"

    alpha_list = [0.2]#0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    method_list = ["max"]#,"perc"]#["max"]
    perc_list = [80,85,90,95]
    
    final_csv = [["alpha","method","percentile","miou","jacc","std_miou","std_jacc"]]

    for alpha in alpha_list:
        for method in method_list:
            if method == "perc":
                for perc in perc_list:
                    print("alpha: ",alpha)
                    print("perc: ",perc)
                    csv_line = cam_main(folder_path,model_name,alpha, method, perc)
                    final_csv.append(csv_line)
            else:
                print("alpha ", alpha)
                perc = 0
                csv_line = cam_main(folder_path,model_name,alpha, method, perc)
                final_csv.append(csv_line)
    print(final_csv)
    # create_csv(folder_path, "compare_smoke/cam_max_02",final_csv)