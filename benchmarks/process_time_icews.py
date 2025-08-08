import os
import pandas as pd
import traceback




in_path = ""
new_path = ""
time_point_formate = "%Y-%m-%d"

def get_time_formate(begin_date, end_date):
    try:
        st_dft = "1900"
        et_dft = "2024"
        st_month = "01"
        et_month = "12"
        begin_date = begin_date.split("-")
        end_date = end_date.split("-")
        if len(begin_date)<2:
            begin_date.append(st_month)    
        if len(end_date)<2:
            end_date.append(et_month)    
            
        if len(begin_date)<3:
            begin_date.append("01")    
        if len(end_date)<3:
            end_date.append("01")    
            
        if begin_date[0] == '####' or not begin_date[0].isdigit():
            begin_date[0] = st_dft
        if end_date[0] == '####' or not end_date[0].isdigit():
            end_date[0] = et_dft
        
        if begin_date[1] == '##' or not begin_date[1].isdigit():
            begin_date[1] = st_month
        if end_date[1] == '##' or not end_date[1].isdigit():
            end_date[1] = et_month
            
        if begin_date[2] == '##' or not begin_date[2].isdigit():
            begin_date[2] = "01"
        if end_date[2] == '##' or not end_date[2].isdigit():
            end_date[2] = "01"
            
        return "-".join(begin_date),"-".join(end_date)
    except Exception as e:
        traceback.print_exc()
        return "1900-01","2024-12"
        
    
def getEverYear( begin_date, end_date, date_formate):
        date_list = []
        date_list = list(range(int(begin_date), int(end_date)+1))
        return date_list

def get_time_point( h, r, t, st, et, date_formate):
        time_list = getEverYear(st, et, date_formate)
        h_tp, r_tp, t_tp, tp = [], [], [], []
        # assert len(h) == len(r) == len(t)
        # for i in range(len(h)):
        for time in time_list:
            h_tp.append(int(h))
            r_tp.append(int(r))
            t_tp.append(int(t))
            tp.append(int(time))
        return h_tp, r_tp, t_tp, tp				

def date_range(start_date, end_date,date_formate="%Y-%m-%d")->set:
    # 创建日期范围
    # print("start_date",start_date)
    # print("end_date",end_date)
    dates = pd.date_range(start=start_date, end=end_date)
    # 将日期转换为字符串格式
    dates_str = dates.strftime(date_formate)
    return set(dates_str.tolist())

def process_time_intervals_type(tri_file,new_file_path):
    
    global time_point_formate
    
    tp_point_set = set()
    
    # st_dft = "1479"
    # et_dft = "2018"
    
    head_tp = []
    tail_tp = []
    rel_tp = []
    tp_point = []
    
    f = open(tri_file, "r")
    triples_total = (int)(f.readline())
    for index in range(triples_total):
        # 注意需要与数据集文件每列的含义对齐
        # h,t,r, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
        h, r, t, st, et = f.readline().strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
        # st = st.split("-")[0]
        # et = et.split("-")[0]
        
        # if st == '####':
        #     st = st_dft
        # if et == '####':
        #     et = et_dft
            
        # if st.find("#") != -1 or len(st) != 4:
        #      continue
        # if et.find("#") != -1  or len(et) != 4:
        #      continue	
                 
        time_point_set =  date_range(st,et,time_point_formate)
            
        for tp in time_point_set:
            head_tp.append(h)
            rel_tp.append(r)
            tail_tp.append(t)
            tp_point.append(tp)	
        tp_point_set.update(time_point_set)
        
    f.close()
    
   
    # 判断长度是否一致
    assert len(head_tp)==len(rel_tp) and len(head_tp)==len(tail_tp) and len(head_tp)==len(tp_point),"四个数组的长度必须一致"
        # 如果数组长度不一致，下面的将不会运行
    
     # 重新生成一份训练数据
    with open(new_file_path, 'w', encoding='utf-8') as out:
        
        out.write(str(len(head_tp))+ '\n')
         
        for row_data  in zip(head_tp,rel_tp,tail_tp,tp_point):
            line_data = '\t'.join(map(str, row_data))
            out.write(line_data + '\n')
    
    return  tp_point_set


def gen_new_file(tp_dict,file_path,new_file_path):
    """
    将时间点转为id
    """
    
    head_tp = []
    tail_tp = []
    rel_tp = []
    tp_point = []
    
    # 生成新的test.txt、vaild.txt、train.txt文件
    # 读取中间的test.txt、vaild.txt和train.txt文件
    f = open(file_path, "r")
    triples_total = (int)(f.readline())
    
    for index in range(triples_total):
        # 注意需要与数据集文件每列的含义对齐
        # h,t,r, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
        h, r, t, tp= f.readline().strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义

        tp_id = tp_dict[tp]
        
        head_tp.append(h)
        rel_tp.append(r)
        tail_tp.append(t)
        tp_point.append(tp_id)		
        
        
    f.close()
    
    # 判断长度是否一致
    assert len(head_tp)==len(rel_tp) and len(head_tp)==len(tail_tp) and len(head_tp)==len(tp_point),"四个数组的长度必须一致"
    # 如果数组长度不一致，下面的将不会运行
        
     # 重新生成一份训练数据
    with open(new_file_path, 'w', encoding='utf-8') as out:
        
        out.write(str(len(head_tp))+ '\n')
         
        for row_data  in zip(head_tp,rel_tp,tail_tp,tp_point):
            line_data = '\t'.join(map(str, row_data))
            out.write(line_data + '\n')
    
def merge_files_with_line_count(file_paths, output_file):
    combined_lines = []
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()[1:]  # 读取文件并跳过第一行
            combined_lines.extend(lines)
    
    # 获取总行数
    total_lines = len(combined_lines)
    
    # 写入新的文件，第一行是总行数，后面是合并的内容
    with open(output_file, 'w') as out_file:
        out_file.write(f"{total_lines}\n")
        out_file.writelines(combined_lines)    
    

def process():
    train_tri_file = "train2id.txt"
    test_tri_file =  "test.txt"
    vaild_tri_file =  "valid.txt"

    tp_point_set = set()
    
    train_tp_set = process_time_intervals_type(os.path.join(in_path, train_tri_file ), os.path.join(new_path, train_tri_file+"_bk"))
    test_tp_set = process_time_intervals_type(os.path.join(in_path, test_tri_file ), os.path.join(new_path, test_tri_file+"_bk"))
    vaild_tp_set = process_time_intervals_type(os.path.join(in_path, vaild_tri_file ), os.path.join(new_path, vaild_tri_file+"_bk"))
    
    tp_point_set.update(train_tp_set)
    tp_point_set.update(test_tp_set)
    tp_point_set.update(vaild_tp_set)
    tp_point2id = {}

    
    # 生成一份时间映射数据
    sorted_dates = sorted(tp_point_set)
    with open(new_path+"tp2id.txt", 'w', encoding='utf-8') as out:
        
        out.write(str(len(tp_point_set))+ '\n')
         
        for i, date in enumerate(sorted_dates):
            out.write(f"{date}\t{i}\n")
            tp_point2id[date] = i
            
    
  
    
    gen_new_file( tp_point2id ,os.path.join(new_path, train_tri_file+"_bk"),os.path.join(new_path, train_tri_file))
    gen_new_file( tp_point2id,os.path.join(new_path, test_tri_file+"_bk"),os.path.join(new_path, test_tri_file))
    gen_new_file( tp_point2id,os.path.join(new_path, vaild_tri_file+"_bk"),os.path.join(new_path, vaild_tri_file))
    merge_files_with_line_count([os.path.join(new_path,train_tri_file),os.path.join(new_path,test_tri_file),
                                 os.path.join(new_path,vaild_tri_file)],os.path.join(new_path,"triple2id.txt"))
   
    
    
def analyse_file(triple_file,first_time_in_rel,last_time_in_rel,rel_time_dict):
  
   
    with open(triple_file,"r") as file:
        event_num = int(file.readline())
        for i in range(event_num):
           
            h,r,t,st,et = file.readline().strip().split()
           
            # st = st.split("-")[0]
            
            r = int(r)
            # st = int(st)
            
            if r not in first_time_in_rel or  first_time_in_rel[r] > st:
                first_time_in_rel[r] = st       
                    
            if r not in last_time_in_rel or  last_time_in_rel[r] < st:
                last_time_in_rel[r] = st
                
            if r not in rel_time_dict:
                rel_time_dict[r] = set()
            rel_time_dict[r].add(st)     
    return first_time_in_rel,last_time_in_rel,rel_time_dict

def analyse(triple_file):
    
    
   
    first_time_in_rel={}
    last_time_in_rel ={}
    global_rel_timenum_dict = {}
    rel_time_dict = {}
    rel_timenum_dict = {}
    # if optional != "tp_point":
    #     init_global_dict(optional)
    
    first_time_in_rel,last_time_in_rel,rel_time_dict = analyse_file(triple_file+"train2id.txt",first_time_in_rel,last_time_in_rel,rel_time_dict)   
    first_time_in_rel,last_time_in_rel,rel_time_dict = analyse_file(triple_file+"test.txt",first_time_in_rel,last_time_in_rel,rel_time_dict) 
    first_time_in_rel,last_time_in_rel,rel_time_dict = analyse_file(triple_file+"valid.txt",first_time_in_rel,last_time_in_rel,rel_time_dict)        
    
    
    
    # print("不同关系的最早发生时间：",first_time_in_rel)
    # print("不同关系的最晚发生时间：", last_time_in_rel)
    for r in first_time_in_rel:
       
        end_date = last_time_in_rel[r].strip()
        start_date = first_time_in_rel[r].strip()
        dates = pd.date_range(start=start_date, end=end_date)
        global_rel_timenum_dict[r] = len(dates)
    # print("不同关系上的时间跨度：", global_rel_timenum_dict)
   
    rel_timenum_dict = {r: len(lst) for r, lst in rel_time_dict.items()}
    # print("不同关系上实际发生时间个数为：",rel_timenum_dict)
    with open(triple_file+"r_time_analyse.txt","w",encoding="utf-8") as file:
        file.write("r \t first\tlast\tduration\tin_time_num\n")
        for i in range(len(first_time_in_rel)):
            
            file.write(str(i)+"\t"+str(first_time_in_rel[i])+"\t"+str(last_time_in_rel[i])+"\t"+str(global_rel_timenum_dict[i])+"\t"+str(rel_timenum_dict[i])+"\n")
        
       




if __name__ == '__main__':
    # in_path = "/u01/cmp/exp_code/OpenKE/benchmarks/has_time/icews14"
    # new_path = "/u01/cmp/exp_code/OpenKE/benchmarks/has_time/icews14_day_filter/"
    in_path = "/u01/cmp/exp_code/OpenKE/benchmarks/has_time/GDELT/"
    new_path = "/u01/cmp/exp_code/OpenKE/benchmarks/has_time/GDELT_day/"
    # 生成一个时间点2id的文件
    process()
    
    # triple_file = "/u01/cmp/exp_code/OpenKE/benchmarks/has_time/icews14/"
    
    # analyse(triple_file)