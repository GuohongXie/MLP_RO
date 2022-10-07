import os
import time 
import sys
import random
import numpy as np
import pandas as pd  
from sklearn import preprocessing


if __name__ == '__main__':
    # set working directory
    os.chdir(sys.path[0])
    print(os.getcwd())

    ################ data_path according to Operating System type   ###############
    source_Qtot_finding_csv_folder_name  = "model_raw_2021_10_29"

    target_dict = {}


    source_Qtot_finding_csv_folder_path = r"../../data/find_csv_ordinal_according_to_parameters/" + source_Qtot_finding_csv_folder_name +"/"
    all_raw_csv_info_path = r'../../data/all_raw_734_info_D1A_D2A_D1B_D2B_a1_u_Dtot_Qtot.csv' 

    df_raw_info = pd.read_csv(all_raw_csv_info_path,\
                         header = None,\
                         names=['D1A','D2A','D1B','D2B','angle','u','Dtot','Qtot'],\
                         encoding="utf8")

    print(df_raw_info['Qtot'][0])
    source_Qtot_csv_list = os.listdir(source_Qtot_finding_csv_folder_path)
    source_Qtot_csv_list.sort(key= lambda x:int(x[:-4]))
    print(source_Qtot_csv_list)

    for i in range(len(source_Qtot_csv_list)):
        source_Qtot_csv_path = source_Qtot_finding_csv_folder_path + source_Qtot_csv_list[i]
        df_source_Qtot_csv = pd.read_csv(source_Qtot_csv_path, header = None,\
                                         names=['x','y','z','D1A','D2A','D1B','D2B','angle','Qtot','Dtot','U','P','C'],\
                                         encoding="utf8")
        print(len(df_source_Qtot_csv))
        for j in range(734):
            if(abs((df_raw_info['D1A'][j] - df_source_Qtot_csv['D1A'][1])) < 1e-8):
                if(abs((df_raw_info['D2A'][j] - df_source_Qtot_csv['D2A'][1])) < 1e-8):
                    if(abs((df_raw_info['D1B'][j] - df_source_Qtot_csv['D1B'][1])) < 1e-8):
                        if(abs((df_raw_info['D2B'][j] - df_source_Qtot_csv['D2B'][1])) < 1e-8):
                            if(abs((df_raw_info['Dtot'][j] - df_source_Qtot_csv['Dtot'][1])) < 1e-8):
                                if(abs((df_raw_info['angle'][j] - df_source_Qtot_csv['angle'][1])) < 1e-2):
                                    if(abs((df_raw_info['Qtot'][j] - df_source_Qtot_csv['Qtot'][1])) < 1e-1):
                                        target_dict[source_Qtot_csv_list[i]] = j+1
                                        break
                                        

    print(target_dict)


