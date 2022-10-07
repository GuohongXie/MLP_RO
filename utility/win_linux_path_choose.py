import platform

operating_system_type = platform.system()
if(operating_system_type == "Windows"):      # when in Windows OS
    raw_csv_folder = r'.\data\raw_csv_all_734'    #要拼接的文件夹及其完整路径，注意不要包含中文 #拼接后要保存的文件路径
    train_data_path = r'.\data\tarin_data_csv_650\train_data_csv_650.csv'
    valid_data_path = r'.\data\valid_data_csv_74\valid_data_csv_74.csv'
    test_data_path = r'.\data\test_data_csv_10\test_data_csv_10.csv'
elif(operating_system_type =="Linux"):       # when in UNIX OS
    raw_csv_folder = r'./data/raw_csv_all_734'    
    train_data_path = r'./data/tarin_data_csv_650/train_data_csv_650.csv' 
    valid_data_path = r'./data/valid_data_csv_74/valid_data_csv_74.csv'
    test_data_path = r'./data/test_data_csv_10/test_data_csv_10.csv'
