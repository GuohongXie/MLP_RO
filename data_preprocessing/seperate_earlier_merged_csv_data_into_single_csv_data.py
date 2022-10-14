import os
import sys
import time 
import ntpath
import pandas as pd  




def seperate_merged_csv_data_into_single_csv_data(source_merged_csv_path, target_single_csv_folder_path, source_csv_list, print_log_path):
    df_source = pd.read_csv(source_merged_csv_path,\
                            header = None,\
                            names=['x','y','z','D1A','D2A','D1B','D2B','angle','Qtot','Dtot','U','P','C'],\
                            encoding="utf8")  
    cnt = 1  ## begin from 1
    section_begin  = 0
    section_end    = 0

    for i in range(1, len(df_source)):
        if(abs((df_source['Qtot'][i] - df_source['Qtot'][i-1])) > 1e-4):

            print("now seperate the %dth csv file"%cnt)
            with open(print_log_path, "a") as log_file:
                print("now seperate the %dth csv file"%cnt, file=log_file)
                log_file.close()

            # now compute the cnt_th section
            section_end = i
            df_tmp = df_source[section_begin:section_end]
            df_tmp['x']    =df_tmp['x'].round(9)
            df_tmp['y']    =df_tmp['y'].round(9)
            df_tmp['z']    =df_tmp['z'].round(9)
            df_tmp['Qtot'] =df_tmp['Qtot'].round(3)
            df_tmp['Dtot'] =df_tmp['Dtot'].round(7)
            df_tmp['U']    =df_tmp['U'].round(8)
            df_tmp['P']    =df_tmp['P'].round(4)
            df_tmp['C']    =df_tmp['C'].round(2)
            save_path = target_single_csv_folder_path + str(source_csv_list[cnt-1]) + '.csv'
            df_tmp.to_csv(save_path, header=None, index=False, encoding="utf8")

            # change index to next section
            section_begin = i
            cnt += 1 

            # compute the last section
            if(cnt == len(source_csv_list)):
                print("now seperate the %dth csv file"%cnt)
                with open(print_log_path, "a") as log_file:
                    print("now seperate the %dth csv file"%cnt, file=log_file)
                    log_file.close()
                
                df_tmp = df_source[section_begin:]
                df_tmp['x']    =df_tmp['x'].round(9)
                df_tmp['y']    =df_tmp['y'].round(9)
                df_tmp['z']    =df_tmp['z'].round(9)
                df_tmp['Qtot'] =df_tmp['Qtot'].round(3)
                df_tmp['Dtot'] =df_tmp['Dtot'].round(7)
                df_tmp['U']    =df_tmp['U'].round(8)
                df_tmp['P']    =df_tmp['P'].round(4)
                df_tmp['C']    =df_tmp['C'].round(2)
                save_path = target_single_csv_folder_path + str(source_csv_list[cnt-1]) + '.csv'
                df_tmp.to_csv(save_path, header=None, index=False, encoding="utf8")
                break
        else:
            continue
    
   
 
    
   
        




if __name__ == '__main__':

    # set working directory
    os.chdir(sys.path[0])
    print(os.getcwd())
    
    ################ data_path according to Operating System type   ###############

    train_source_merged_csv_path = r'../../data/train_and_valid_merged_csv_all__minmax_xyz_earlier/train_data_csv/train_data_csv_650.csv'   
    valid_source_merged_csv_path = r'../../data/train_and_valid_merged_csv_all__minmax_xyz_earlier/valid_data_csv/valid_data_csv_74.csv'  
    test_source_merged_csv_path  = r'../../data/train_and_valid_merged_csv_all__minmax_xyz_earlier/test_data_csv/test_data_csv_10.csv'   

    train_target_single_csv_folder_path = r'../../data/raw_earlier_csv_all_734/'   
    valid_target_single_csv_folder_path = r'../../data/raw_earlier_csv_all_734/'  
    test_target_single_csv_folder_path = r'../../data/raw_earlier_csv_all_734/'

    file_name = ntpath.basename(os.path.abspath(__file__))
    file_name = file_name[:-3]
    log_path = r'../results/data_processing_log/' + file_name + '.txt'

    train_csv_list = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 14, 16, 17, 19, 20, 21, 23, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 18, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 101, 102, 103, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 127, 128, 129, 130, 132, 133, 134, 135, 136, 139, 140, 141, 142, 144, 145, 146, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 211, 212, 213, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 266, 267, 268, 269, 270, 271, 272, 273, 274, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 290, 291, 293, 294, 295, 296, 297, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 314, 316, 317, 319, 321, 322, 323, 324, 325, 326, 327, 328, 329, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 370, 371, 372, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 388, 390, 391, 392, 395, 396, 363, 398, 399, 400, 401, 402, 403, 405, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 412, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 445, 446, 447, 448, 449, 450, 451, 452, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 378, 467, 468, 469, 470, 471, 472, 474, 475, 476, 477, 478, 479, 480, 481, 484, 485, 486, 487, 488, 489, 490, 487, 492, 493, 494, 495, 497, 498, 500, 501, 502, 503, 504, 505, 507, 509, 510, 511, 512, 513, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 522, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 636, 637, 638, 640, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 654, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 702, 218, 705, 707, 708, 709, 710, 711, 712, 713, 716, 718, 719, 720, 721, 722, 724, 726, 727, 728, 730, 731, 732, 733, 734]
    valid_csv_list = [8, 10, 15, 18, 22, 24, 28, 29, 39, 50, 55, 62, 63, 64, 82, 90, 100, 104, 106, 126, 131, 137, 143, 147, 160, 171, 198, 210, 214, 226, 237, 249, 265, 275, 289, 292, 298, 313, 315, 318, 330, 331, 354, 373, 387, 389, 393, 394, 404, 406, 444, 453, 473, 483, 496, 431, 506, 508, 514, 537, 551, 566, 634, 635, 641, 653, 655, 701, 706, 714, 715, 717, 723, 725]
    test_csv_list  = [12, 38, 138, 320, 369, 482, 613, 639, 704, 729]



    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('start time is:',start_time)
    with open(log_path, "a") as log_file:
        print('start time is:%s'%start_time, file=log_file)
        log_file.close()

    
    with open(log_path, "a") as log_file:
        print('\nnow processing the test_data_csv_10', file=log_file)
        log_file.close()
    seperate_merged_csv_data_into_single_csv_data(test_source_merged_csv_path, test_target_single_csv_folder_path, test_csv_list, log_path)
    
    with open(log_path, "a") as log_file:
        print('\nnow processing the valid_data_csv_74', file=log_file)
        log_file.close()
    seperate_merged_csv_data_into_single_csv_data(valid_source_merged_csv_path, valid_target_single_csv_folder_path, valid_csv_list, log_path)

    with open(log_path, "a") as log_file:
        print('\nnow processing the train_data_csv_650', file=log_file)
        log_file.close()
    seperate_merged_csv_data_into_single_csv_data(train_source_merged_csv_path, train_target_single_csv_folder_path, train_csv_list, log_path)

    


    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print('end time is:', end_time)
    print("you are all set")
    with open(log_path, "a") as log_file:
        print('end time is:%s'%end_time, file=log_file)
        print("you are all set", file=log_file)
        log_file.close()
