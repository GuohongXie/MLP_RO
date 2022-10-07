# works only under windows 7/8/xp/10/11 series
# change the work directory name first when in UINX OS 
import platform
import numpy as np
import pandas as pd  
import os
import random
from sklearn import preprocessing

################ data_path according to Operating System type   ###############
operating_system_type = platform.system()
if(operating_system_type =="Windows"):
    Folder_Path= r'.\mat2csv\target_csv'    #要拼接的文件夹及其完整路径，注意不要包含中文
    SaveFile_Path= r'D:\xgh_MLN_projects_2022_04_12_edit\data\merged_csv_test'  #拼接后要保存的文件路径
elif(operating_system_type =="Linux"):
    Folder_Path= r'./mat2csv/target_csv'    # when in UNIX OS
    SaveFile_Path= r'D:/xgh_MLN_projects_2022_04_12_edit/data/merged_csv_test'  # when in UNIX OS
SaveFile_Name='merged_csv_from_1_to_2.csv'    #合并后要保存的文件名
file_list = os.listdir(Folder_Path)



print(file_list)