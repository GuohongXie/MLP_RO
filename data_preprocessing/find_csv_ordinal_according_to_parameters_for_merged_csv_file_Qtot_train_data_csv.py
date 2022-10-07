import os
import sys
import time 
import ntpath
import random
import numpy as np
import pandas as pd  
from sklearn import preprocessing


if __name__ == '__main__':
    # set working directory
    os.chdir(sys.path[0])
    print(os.getcwd())

    ################ data_path according to Operating System type   ###############

    target_dict = {}
    source_Qtot_csv_path = r"../../data/train_and_valid_merged_csv_Qtot/train_data_csv/train_csv.csv"
    all_raw_csv_info_path = r'../../data/all_raw_734_info_D1A_D2A_D1B_D2B_a1_u_Dtot_Qtot.csv' 
    print_log_folder_path = r'../results/data_processing_log/'

    file_name = ntpath.basename(os.path.abspath(__file__))
    file_name = file_name[:-3]
    print_log_file_path = print_log_folder_path + file_name + ".txt"

    # record time
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start time is:',start_time)
    with open(print_log_file_path, "a") as log_file:
        print('start time is:',start_time, file=log_file)
        #print("the lenth of train_dataset is %d"%len(train_dl.dataset), file=log_file)
        #print("the lenth of valid_dataset is %d"%len(valid_dl.dataset), file=log_file)
        log_file.close()

    df_raw_info = pd.read_csv(all_raw_csv_info_path,\
                         header = None,\
                         names=['D1A','D2A','D1B','D2B','angle','u','Dtot','Qtot'],\
                         encoding="utf8")

    #print(df_raw_info['Qtot'][0])
    

    df_source_Qtot_csv = pd.read_csv(source_Qtot_csv_path, \
                                     #header = None,\
                                     #names=['x','y','z','D1A','D2A','D1B','D2B','angle','Qtot','Dtot','U','P','C'],\
                                     encoding="utf8")
    print("the lenth of df_raw_info is %d"%len(df_raw_info))
    print("the lenth of df_source_Qtot_csv is %d"%len(df_source_Qtot_csv))
    with open(print_log_file_path, "a") as log_file:
        print("the lenth of df_raw_info is %d"%len(df_raw_info), file=log_file)
        print("the lenth of df_source_Qtot_csv is %d"%len(df_source_Qtot_csv), file=log_file)
        log_file.close()

    for j in range(734):
        if(abs((df_raw_info['D1A'][j] - df_source_Qtot_csv['D1A'][0])) < 1e-8):
            if(abs((df_raw_info['D2A'][j] - df_source_Qtot_csv['D2A'][0])) < 1e-8):
                if(abs((df_raw_info['D1B'][j] - df_source_Qtot_csv['D1B'][0])) < 1e-8):
                    if(abs((df_raw_info['D2B'][j] - df_source_Qtot_csv['D2B'][0])) < 1e-8):
                        if(abs((df_raw_info['Dtot'][j] - df_source_Qtot_csv['Dtot'][0])) < 1e-8):
                            if(abs((df_raw_info['angle'][j] - df_source_Qtot_csv['angle'][0])) < 1e-2):
                                if(abs((df_raw_info['Qtot'][j] - df_source_Qtot_csv['Qtot'][0])) < 1e-1):
                                    target_dict['1'] = j+1
                                    break

    cnt = 1
    for i in range(1, len(df_source_Qtot_csv)):
        if(abs((df_source_Qtot_csv['Qtot'][i] - df_source_Qtot_csv['Qtot'][i-1])) > 1e-4):
            cnt += 1
            print("cnt now is %d"%cnt)
            with open(print_log_file_path, "a") as log_file:
                print("cnt now is %d"%cnt, file=log_file)
                log_file.close()
        else:
            continue
                                    
        for j in range(734):
            if(abs((df_raw_info['D1A'][j] - df_source_Qtot_csv['D1A'][i])) < 1e-8):
                if(abs((df_raw_info['D2A'][j] - df_source_Qtot_csv['D2A'][i])) < 1e-8):
                    if(abs((df_raw_info['D1B'][j] - df_source_Qtot_csv['D1B'][i])) < 1e-8):
                        if(abs((df_raw_info['D2B'][j] - df_source_Qtot_csv['D2B'][i])) < 1e-8):
                            if(abs((df_raw_info['Dtot'][j] - df_source_Qtot_csv['Dtot'][i])) < 1e-8):
                                if(abs((df_raw_info['angle'][j] - df_source_Qtot_csv['angle'][i])) < 1e-2):
                                    if(abs((df_raw_info['Qtot'][j] - df_source_Qtot_csv['Qtot'][i])) < 1e-1):
                                        target_dict[str(cnt)] = j+1
                                        break
                                        

    
    target_list = []
    for i in range(len(target_dict)):
        target_list.append(target_dict[str(i+1)])
    print(target_list)
    print(len(target_dict))
    print(target_dict)
    with open(print_log_file_path, "a") as log_file:
        print('\n', file=log_file)
        print('the length of the result dict is:%d'%len(target_dict), file=log_file)
        print('the result dict is:', file=log_file)
        print(target_dict, file=log_file)
        print('the length of the result list is:%d'%len(target_list), file=log_file)
        print('the result list is:', file=log_file)
        print(target_list, file=log_file)
        print('\n', file=log_file)
        log_file.close()



    # record time
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('end time is:', end_time)
    print("you are all set")
    with open(print_log_file_path, "a") as log_file:
        print('end time is:',end_time, file=log_file)
        log_file.close()
