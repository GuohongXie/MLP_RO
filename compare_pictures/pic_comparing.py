## import necessary packages
import os
from statistics import mean
import sys
import time
from pictures_compare_functions import *


def get_median(data, round_num):
    data.sort()
    half = len(data) // 2
    return round((data[half] + data[~half]) / 2, round_num)


## set working directory 
os.chdir(sys.path[0])
print(os.getcwd())

## set data path
predict_data_folder_name = "inside_plus_outside_threeNet_mlp_4_layer_struct_0x4"
result_csv_file_list = [7, 55, 169, 170, 268, 289, 723]
similarity_algo = ["ssim_grey", "ssim_multi_channel", "psnr", "mse_grey"]

## data path
raw_pic_folder = r"../pictures/U_P_C_raw/raw_data/"
predict_pic_folder = r"../pictures/U_P_C_predict/" + predict_data_folder_name + "/"
print_log_path = r"./results/" + predict_data_folder_name + ".csv"


####### record the start time ###########
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("start time is %s" %start_time)

U_similarity = {}
P_XY_similarity = {}
C_similarity = {}
C_XY_similarity = {}
for i in range(len(similarity_algo)):
    U_similarity[similarity_algo[i]] = []
    P_XY_similarity[similarity_algo[i]] = []
    C_similarity[similarity_algo[i]] = []
    C_XY_similarity[similarity_algo[i]] = []


for i  in range(len(result_csv_file_list)):
    predict_U_pic_folder_i = predict_pic_folder + "U/" + str(result_csv_file_list[i]) + "/"
    predict_C_XY_pic_folder_i = predict_pic_folder + "C_XY/" + str(result_csv_file_list[i]) + "/"
    predict_P_XY_pic_folder_i = predict_pic_folder + "P_XY/" + str(result_csv_file_list[i]) + "/"
    predict_C_pic_folder_i = predict_pic_folder + "C/" + str(result_csv_file_list[i]) + "/"
    

    raw_U_pic_folder_i = raw_pic_folder + "U/" + str(result_csv_file_list[i]) + "/"
    raw_C_XY_pic_folder_i = raw_pic_folder + "C_XY/" + str(result_csv_file_list[i]) + "/"
    raw_P_XY_pic_folder_i = raw_pic_folder + "P_XY/" + str(result_csv_file_list[i]) + "/"
    raw_C_pic_folder_i = raw_pic_folder + "C/" + str(result_csv_file_list[i]) + "/"
    

    for j in range(5):
        predict_U_pic_j = predict_U_pic_folder_i + "U_" + str(j+1) + ".png"
        predict_C_XY_pic_j = predict_C_XY_pic_folder_i + "C_XY_" + str(j+1) + ".png"
        predict_P_XY_pic_j = predict_P_XY_pic_folder_i + "P_XY_" + str(j+1) + ".png"
        predict_C_pic_j = predict_C_pic_folder_i + "C_" + str(j+1) + ".png"
        
        
        
        raw_U_pic_j = raw_U_pic_folder_i + "U_" + str(j+1) + ".png"
        raw_C_XY_pic_j = raw_C_XY_pic_folder_i + "C_XY_" + str(j+1) + ".png"
        raw_P_XY_pic_j = raw_P_XY_pic_folder_i + "P_XY_" + str(j+1) + ".png"
        raw_C_pic_j = raw_C_pic_folder_i + "C_" + str(j+1) + ".png"
        
       
        
        for k in range(len(similarity_algo)):
            try:
                U_similarity[similarity_algo[k]].append(eval(similarity_algo[k])(raw_U_pic_j, predict_U_pic_j))
                C_similarity[similarity_algo[k]].append(eval(similarity_algo[k])(raw_C_pic_j, predict_C_pic_j))
                P_XY_similarity[similarity_algo[k]].append(eval(similarity_algo[k])(raw_P_XY_pic_j, predict_P_XY_pic_j))
                C_XY_similarity[similarity_algo[k]].append(eval(similarity_algo[k])(raw_C_XY_pic_j, predict_C_XY_pic_j))
            except:
                pass


with open(print_log_path, "a") as log_file:
    for i in range(len(similarity_algo)):
        print("U_%s"%similarity_algo[i], *U_similarity[similarity_algo[i]], sep=",", file=log_file)
    for i in range(len(similarity_algo)):
        print("C_XY_%s"%similarity_algo[i], *C_XY_similarity[similarity_algo[i]], sep=",", file=log_file) 
    for i in range(len(similarity_algo)):
        print("P_XY_%s"%similarity_algo[i], *P_XY_similarity[similarity_algo[i]], sep=",", file=log_file)
    for i in range(len(similarity_algo)):
        print("C_%s"%similarity_algo[i], *C_similarity[similarity_algo[i]], sep=",", file=log_file)
    print("\n", file=log_file)
    for i in range(len(similarity_algo)):
        print("U_%s_statistic"%similarity_algo[i], max(U_similarity[similarity_algo[i]]), min(U_similarity[similarity_algo[i]]), round(mean(U_similarity[similarity_algo[i]]), 6), get_median(U_similarity[similarity_algo[i]], 6),  sep=",", file=log_file)
    for i in range(len(similarity_algo)):
        print("C_XY_%s_statistic"%similarity_algo[i], max(C_XY_similarity[similarity_algo[i]]), min(C_XY_similarity[similarity_algo[i]]), round(mean(C_XY_similarity[similarity_algo[i]]), 6), get_median(C_XY_similarity[similarity_algo[i]], 6),  sep=",", file=log_file)
    for i in range(len(similarity_algo)):
        print("P_XY_%s_statistic"%similarity_algo[i], max(P_XY_similarity[similarity_algo[i]]), min(P_XY_similarity[similarity_algo[i]]), round(mean(P_XY_similarity[similarity_algo[i]]), 6), get_median(P_XY_similarity[similarity_algo[i]], 6),  sep=",", file=log_file)
    for i in range(len(similarity_algo)):
        print("C_%s_statistic"%similarity_algo[i], max(C_similarity[similarity_algo[i]]), min(C_similarity[similarity_algo[i]]), round(mean(C_similarity[similarity_algo[i]]), 6), get_median(C_similarity[similarity_algo[i]], 6),  sep=",", file=log_file)
    log_file.close()





############# record end time ################
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print("end time is %s" %end_time)






