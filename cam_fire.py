import time
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
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
from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score, roc_curve, auc
import csv


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

def create_csv(path,name,row_list):
    with open(path+name+".csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)
    return None

def performance_calculate(gt_name, heatmap):
    name = "/home/bamaral/thesis/AnnotDatasets/Fire_dataset/gt/" + str(gt_name[0:4]) + "_gt_fire.png"
    gt_img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)

    # gt_img = cv2.resize(gt_img,(500,500))
    # gt_img[np.where(gt_img < 0.5)] = 0
    # gt_img[np.where(gt_img > 0.5)] = 1

    # heatmap = cv2.resize(heatmap,(500,500))
    # heatmap[np.where(heatmap < 0.5)] = 0
    # heatmap[np.where(heatmap > 0.5)] = 1

    # print(np.shape(gt_img))
    # print(np.shape(heatmap))
    miou = tf.keras.metrics.MeanIoU(num_classes=2)
    miou.reset_states()
    miou.update_state(gt_img/255,heatmap)
    
    jacc = jaccard_score(gt_img/255,heatmap, average= "micro") #, average="micro")
    # jacc = 1
    return miou.result().numpy(), jacc


def classif_metrics(true_list,pred_list,pred_list_float,folder_path):
    acc = accuracy_score(true_list,pred_list)
    prec = precision_score(true_list,pred_list)
    rec = recall_score(true_list,pred_list)
    
    print("Acc: ", acc)
    print("Recall: ", prec)
    print("Precision: ", rec)

    return acc, prec, rec


def cam(folder_path, model_name, path_test, alpha, method, perc):

    images_np = sorted_alphanumeric(os.listdir(path_test))[600:]
    # images_np_2 = [601,602,619,626,672,710,714,716,717,718,765]
    # images_np = np.array(images_np)

    # print(images_np[images_np_2])
    # images_np = images_np[images_np_2]
    K.clear_session()
    model = load_model(folder_path+model_name)
    # model.summary()
    out_class = 0
    perf_csv = [["image", "miou","jacc"]]
    miou_list = []
    jacc_list = []
    true_list = []
    pred_list = []
    pred_list_float = []
    for img in images_np:
        img_path=path_test+img
        img_name = img[0:4]
        # print("Image: ", img)
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

        #Get the 512 input weights to the sigmoid.
        class_weights = model.layers[-1].get_weights()[0]

        final_conv_layer = get_output_layer(model, "block5_conv4")
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([x])
        # print("PRED",predictions)

        conv_outputs = conv_outputs[0, :, :, :]

        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        weigts_csv = [["feature_map", "weight"]]

        for p, w in enumerate(class_weights[:, out_class]):
            if w > 0:
                cam += w * conv_outputs[:, :, p]

            # weigts_csv.append([p,w])
            # feature_map = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
            # feature_map += w * conv_outputs[:, :, p]
            # feature_map /= np.max(feature_map)
            # feature_map = cv2.resize(feature_map, (height, width))
            # feature_map_color = cv2.applyColorMap(np.uint8(255*feature_map), cv2.COLORMAP_JET)
            # # feature_map_color[np.where(feature_map_color < 0.15)] = 0
            # output_fm = folder_path+"compare/feature_maps/"+str(img_name)+"_fm_"+str(p)+".png"
            # print(output_fm)
            # cv2.imwrite(output_fm, feature_map_color)
            # create_csv(folder_path+"experiment_B/feature_maps/",str(img_name)+"_fm",weigts_csv)

        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        print("time= ", time.time()-start_time )
        if method == "max":
            tresh2 = alpha *np.max(cam)
        if method == "perc":
            tresh2 = alpha * np.percentile(cam,perc)


        heatmap1_1 = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap1_1[np.where(cam < 0.01)] = 0
        output = folder_path+"experiment_B/wo_wo/cam_0.5/"+img_name+'_out.png'
    
        # tresholded
        heatmap2_seg = np.uint8(cam)
        heatmap2_seg[np.where(cam < tresh2)] = 0
        heatmap2_seg[np.where(cam >= tresh2)] = 1
        output_seg2 = folder_path+"experiment_B/wo_wo/mask_0.5_2/"+img_name+'_cam_mask.png'

        img = heatmap1_1*0.8 + original_img
        # cv2.putText(img,"Fire: "+str(predictions[0][0]),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 0, 0),2)

        heatmap2_seg_img = cv2.cvtColor(heatmap2_seg ,cv2.COLOR_GRAY2RGB)

        img_mask = original_img + heatmap2_seg_img*0.3*255
        output_segimg2 = folder_path+"experiment_B/wo_wo/maskOnImage/"+img_name+'_seg_img.png'

        # cv2.imwrite(output, img)
        # cv2.imwrite(folder_path+"compare/cam_w0/img/"+img_name+'_mask.png',heatmap1_1)
        # cv2.imwrite(output_segimg2, img_mask)
        cv2.imwrite(output_seg2, 255*heatmap2_seg)

        miou, jacc = performance_calculate(img_name, heatmap2_seg)
        # print("miou: ", miou)

        miou_list.append(miou)
        jacc_list.append(jacc)
            
        perf_csv.append([img_name,miou,jacc])

    # print("mIoU: ", np.mean(miou_list))
    # print("Jacc: ", np.mean(jacc_list))

    # acc, prec, rec = classif_metrics(true_list,pred_list,pred_list_float,folder_path)
    final_csv = [alpha, method, perc, np.mean(miou_list),np.mean(jacc_list),np.std(miou_list),np.std(jacc_list)]
    # create_csv(folder_path,"miou_jacc",final_csv)
    # create_csv(folder_path,"experiment_B/w_wo/cam_"+str(alpha),perf_csv)

    return final_csv

def get_output_layer(model, layer_name):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer

def cam_main(folder_path,model_name,alpha, method, perc):
    
    # try:
    #     os.makedirs(folder_path+'compare/cam_w0')
    # except OSError:
    #     print("folder")
    
    # try:
    #     os.makedirs(folder_path+'compare/cam_w0/img')
    # except OSError:
    #  print("folder")

    # try:
    #     os.makedirs(folder_path+'compare/cam_w0/mask')
    # except OSError:
    #     print("folder")

    path_test= "/home/bamaral/thesis/AnnotDatasets/Fire_dataset/img/"

    final_csv = cam(folder_path, model_name, path_test, alpha, method, perc)

    return final_csv

if __name__ == '__main__':

    folder_path = "/home/bamaral/thesis/other_models/adam_1e-05/"
    model_name = "adam_1e-05.h5"
    alpha_list = [0.5]
    method_list = ["max"]#["max"]
    perc_list = [80]#,90,95]
    
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
    # create_csv(folder_path, "experiment_B/cam_perc_wo_w0",final_csv)



# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--folder_path', default='', help='Folder for the to get the model and save the metrics')
# parser.add_argument('--model_name', default='', help='Model name')
# args = parser.parse_args()
# print("\n\n\n\n",len(args.model_name))
# print("\n\n\n\n",args.model_name)

# cam_main(args.folder_path, args.model_name)
