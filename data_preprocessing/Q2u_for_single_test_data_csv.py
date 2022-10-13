import os
import sys
import time 
import pandas as pd  


################ data_path according to Operating System type   ###############



def replace_Q_with_u(source_folder, target_folder, all_info_path, test_csv_file_list):
    #source_file_list = os.listdir(source_folder)
    #source_file_list.sort(key= lambda x:int(x[:-4]))
    df_all_info = pd.read_csv(all_info_path,\
                              header = None,\
                              names=['D1A','D2A','D1B','D2B','angle','u','Dtot','Qtot'],\
                              encoding="utf8")  
    u_list = []
    for i in range(734):
        u_list.append(df_all_info['u'][i])

    for i in range(len(test_csv_file_list)):
        df_source = pd.read_csv(source_folder + str(test_csv_file_list[i]) + '.csv',\
                             header = None,\
                             names=['x','y','z','D1A','D2A','D1B','D2B','angle','u','Dtot','U','P','C'],\
                             encoding="utf8")  #编码默认UTF-8，若乱码自行更改
        df_source['u'] = u_list[test_csv_file_list[i]-1]
        #print(df_source['u'])
        df_source['x']    = df_source['x'].round(9)
        df_source['y']    = df_source['y'].round(9)
        df_source['z']    = df_source['z'].round(9)
        df_source['u']    = df_source['u'].round(6)
        df_source['Dtot'] = df_source['Dtot'].round(7)
        df_source['U']    = df_source['U'].round(8)
        df_source['P']    = df_source['P'].round(4)
        df_source['C']    = df_source['C'].round(2)
        #将读取的第一个CSV文件写入合并后的文件保存  
        save_path = target_folder + str(test_csv_file_list[i]) + '.csv'
        df_source.to_csv(save_path, header=None, index=False, encoding="utf8")
        




if __name__ == '__main__':
    # set working directory
    os.chdir(sys.path[0])
    print(os.getcwd())



    source_csv_folder = r'../../data/test_data_Qtot/raw_data_2021_10_29/'    
    target_csv_folder = r'../../data/test_data_u/u_raw_data_2021_10_29/'
    all_raw_734_info_csv_path = r'../../data/all_raw_734_info_D1A_D2A_D1B_D2B_a1_u_Dtot_Qtot.csv'
    test_csv_list  = [12, 38, 138, 320, 369, 482, 613, 639, 704, 729]

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start time is:',start_time)

    replace_Q_with_u(source_csv_folder, target_csv_folder, all_raw_734_info_csv_path, test_csv_list)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('end time is:', end_time)

    print("you are all set")
