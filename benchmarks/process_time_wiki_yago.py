import os
import shutil
import pandas as pd
import traceback

'''
将数据的时间拆为时间点，每年为一个时间点
'''

# wiki
# in_path = "/u01/cmp/exp_code/OpenKE/benchmarks/wiki_data_addTime/"
# new_path = "/u01/cmp/exp_code/OpenKE/benchmarks/data/wiki_data_addTime_year_tp2id_filter_bk/"

# st_dft = "1000"
# et_dft ="2020"
# time_point_formate = "%Y"


# yago
# in_path = "/u01/cmp/exp_code/OpenKE/benchmarks/data/yago/large"
# new_path = "/u01/cmp/exp_code/OpenKE/benchmarks/data/yago_data_addTime_year_tp2id_filter_bk/"

# st_dft = "1010"
# et_dft ="2017"

# 声明一个全局的字典存储 关系 的最早时间和最晚时间，以便在补缺失值时使用
first_time_in_rel = {}
last_time_in_rel = {}
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
    print("start_date",start_date)
    print("end_date",end_date)
    dates = pd.date_range(start=start_date, end=end_date)
    # 将日期转换为字符串格式
    dates_str = dates.strftime(date_formate)
    return set(dates_str.tolist())

def copy_file(new_dir,file_path):
    # 获取文件名
    file_name = os.path.basename(file_path)
    # 将其他文件复制到新文件夹中
    destination_file = os.path.join(new_dir, file_name)
        
    shutil.copy(file_path,destination_file)

def process_time_intervals(tri_file,new_file_path,optional='default',first_st_limit=None,last_et_limit=None):
    
    global time_point_formate
    global st_dft 
    global et_dft
    global first_time_in_rel
    global last_time_in_rel 
    
    tp_point_set = set()
    
    
    
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
        st = st.split("-")[0]
        et = et.split("-")[0]
        if optional == 'tp_point':
            if st == '####':
                st = et
            if et == '####':
                et = st
            if st=='####' or et=='####':
                continue
        else:
            
            if st == '####':
                st = str(first_time_in_rel[int(r)])
            #    st =  st_dft
            if et == '####':
                et = str(last_time_in_rel[int(r)])
                # et = et_dft
            
            
        if st.find("#") != -1:
             continue
        if et.find("#") != -1:
             continue	
        
        st = int(st)
        et = int(et)
        
        if st > et:
            print(h,r,t,st,et)
            continue 
        if st and  first_st_limit and st < int(first_st_limit):
            st = first_st_limit
        if et and last_et_limit and et > int(last_et_limit):
            et = last_et_limit
        
        h_tp, r_tp, t_tp, tp = get_time_point(h, r, t, st, et, time_point_formate)
        head_tp.extend(h_tp)
        rel_tp.extend(r_tp)
        tail_tp.extend(t_tp)
        tp_point.extend(tp)	
        # st, et = get_time_formate(st,et)
        # time_point_set =  date_range(st,et,time_point_formate)
            
        
        tp_point_set.update(tp_point)
        
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

def process_time_intervals_type(tri_file,new_file_path,optional='default',first_st_limit=None,last_et_limit=None):
    
    global time_point_formate
    global st_dft 
    global et_dft 
    global first_time_in_rel
    global last_time_in_rel 
    tp_point_set = set()
    
    # st_dft = "1479"
    # et_dft = "2018"
    
    head_tp = []
    tail_tp = []
    rel_tp = []
    tp_point = []
    
    head = []
    tail = []
    rel = []
    st_list = []
    et_list = []
    
    f = open(tri_file, "r")
    triples_total = (int)(f.readline())
    for index in range(triples_total):
        # 注意需要与数据集文件每列的含义对齐
        # h,t,r, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
        h, r, t, st, et = f.readline().strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
        st = st.split("-")[0]
        et = et.split("-")[0]
        
        if optional == 'tp_point':
            if st == '####':
                st = et
            if et == '####':
                et = st
            if st=='####' or et=='####':
                continue
        else:
            
            if st == '####':
                st = str(first_time_in_rel[int(r)])
            #    st =  st_dft
            if et == '####':
                et = str(last_time_in_rel[int(r)])
                # et = et_dft
            
        if st.find("#") != -1:
             continue
        if et.find("#") != -1:
             continue	
        st = int(st)
        et = int(et)
        
        
        if st and first_st_limit and st < int(first_st_limit):
            st = first_st_limit
        if et and last_et_limit and et > int(last_et_limit):
            et = last_et_limit
        
        if st > et:
            
            print(h,r,t,st,et)
            continue
        
        head.append(h)
        tail.append(t)
        rel.append(r)
        st_list.append(st)
        et_list.append(et)
        
        h_tp, r_tp, t_tp, tp = get_time_point(h, r, t, st, et, time_point_formate)
        head_tp.extend(h_tp)
        rel_tp.extend(r_tp)
        tail_tp.extend(t_tp)
        tp_point.extend(tp)	
        # st, et = get_time_formate(st,et)
        # time_point_set =  date_range(st,et,time_point_formate)
            
        
        tp_point_set.update(tp_point)
        
    f.close()
    
   
    # 判断长度是否一致
    assert len(head)==len(rel) and len(head)==len(tail) and len(head)==len(st_list) and len(head)==len(et_list) ,"四个数组的长度必须一致"
        # 如果数组长度不一致，下面的将不会运行
    # 判断长度是否一致
    assert len(head_tp)==len(rel_tp) and len(head_tp)==len(tail_tp) and len(head_tp)==len(tp_point),"四个数组的长度必须一致"
        # 如果数组长度不一致，下面的将不会运行

     # 重新生成一份训练数据
    with open(new_file_path, 'w', encoding='utf-8') as out:  
        out.write(str(len(head))+ '\n')
            
        for row_data  in zip(head,rel,tail,st_list,et_list):
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
        line = f.readline()
        h, r, t, tp= line.strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义

        tp_id = tp_dict[int(tp)]
        
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
            
def gen_new_file_type(tp_dict,file_path,new_file_path):
    """
    将时间点转为id
    """
    
    head_tp = []
    tail_tp = []
    rel_tp = []
    st_list = []
    et_list =[]
    # 生成新的test.txt、vaild.txt、train.txt文件
    # 读取中间的test.txt、vaild.txt和train.txt文件
    f = open(file_path, "r")
    triples_total = (int)(f.readline())
    
    for index in range(triples_total):
        # 注意需要与数据集文件每列的含义对齐
        h,r,t, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
        

        st_id = tp_dict[int(st)]
        et_id = tp_dict[int(et)]
        head_tp.append(h)
        rel_tp.append(r)
        tail_tp.append(t)
        st_list.append(st_id)
        et_list.append(et_id)		
        
        
    f.close()
    
    # 判断长度是否一致
    assert len(head_tp)==len(rel_tp) and len(head_tp)==len(tail_tp) and len(head_tp)==len(st_list) and len(head_tp)==len(et_list),"四个数组的长度必须一致"
    # 如果数组长度不一致，下面的将不会运行
        
     # 重新生成一份训练数据
    with open(new_file_path, 'w', encoding='utf-8') as out:
        
        out.write(str(len(head_tp))+ '\n')
         
        for row_data  in zip(head_tp,rel_tp,tail_tp,st_list,et_list):
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


    

def process(optional="default",first_st_limit = None,last_et_limit = None):
    """_summary_

    Args:
        optional (str, optional): _description_. Defaults to "default".
        optional:default   统计整个数据集的最早和最晚时间 每个关系的最早时间和最晚时间一样，均是整个数据集的最早和最晚
        optional:rel  统计每个关系的最早时间和最晚时间
        optional:tp_point 如果开始时间和结束时间都缺失，则丢弃；如果只缺失一个，用另一个补全
    """
    if optional != "tp_point":
        init_global_dict(optional,first_st_limit,last_et_limit)
        
    train_tri_file = "train2id.txt"
    test_tri_file =  "test.txt"
    vaild_tri_file =  "valid.txt"

    tp_point_set = set()
    
    # 将训练集、测试集、验证集根据年拆成一个个时间点的事件 
    train_tp_set = process_time_intervals(os.path.join(in_path, train_tri_file ), os.path.join(new_path, train_tri_file+"_bk_tp_point"),optional,first_st_limit,last_et_limit)
    process_time_intervals(os.path.join(in_path, test_tri_file ), os.path.join(new_path, test_tri_file+"_bk_tp_point"),optional,first_st_limit,last_et_limit)
    process_time_intervals(os.path.join(in_path, vaild_tri_file ), os.path.join(new_path, vaild_tri_file+"_bk_tp_point"),optional,first_st_limit,last_et_limit)
    # 将测试集、验证集 仍是原来的时间段，只不过将时间按照规范后的方法表达 ，如"%Y"
    process_time_intervals_type(os.path.join(in_path, train_tri_file ), os.path.join(new_path, train_tri_file+"_bk_tp_duration"),optional,first_st_limit,last_et_limit)
    test_tp_set = process_time_intervals_type(os.path.join(in_path, test_tri_file ), os.path.join(new_path, test_tri_file+"_bk_tp_duration"),optional,first_st_limit,last_et_limit)
    vaild_tp_set = process_time_intervals_type(os.path.join(in_path, vaild_tri_file ), os.path.join(new_path, vaild_tri_file+"_bk_tp_duration"),optional,first_st_limit,last_et_limit)
    
    # 统计所有的时间点，将其映射为tp2id的文件
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
            
    
  
    # 将训练集、测试集、验证集根据年拆成一个个时间点的事件，事件的时间根据id表示
    gen_new_file( tp_point2id ,os.path.join(new_path, train_tri_file+"_bk_tp_point"),os.path.join(new_path, train_tri_file))
    gen_new_file( tp_point2id ,os.path.join(new_path, test_tri_file+"_bk_tp_point"),os.path.join(new_path, test_tri_file+"_bk_tp_point_2id"))
    gen_new_file( tp_point2id ,os.path.join(new_path, vaild_tri_file+"_bk_tp_point"),os.path.join(new_path, vaild_tri_file+"_bk_tp_point_2id"))
    
    # 将测试集、验证集 仍是原来的时间段，只不过将时间按照规范后的方法表达 ，如"%Y"，事件的时间根据id表示
    gen_new_file_type( tp_point2id,os.path.join(new_path, train_tri_file+"_bk_tp_duration"),os.path.join(new_path, train_tri_file+"_bk_tp_duration_2id"))
    gen_new_file_type( tp_point2id,os.path.join(new_path, test_tri_file+"_bk_tp_duration"),os.path.join(new_path, test_tri_file))
    gen_new_file_type( tp_point2id,os.path.join(new_path, vaild_tri_file+"_bk_tp_duration"),os.path.join(new_path, vaild_tri_file))
    
    # 将训练集、测试集、验证集仍是原来的时间段，事件的时间根据id表示，将这个文件合并，表示数据集中所有的事件（事件时间仍是时间段）
    merge_files_with_line_count([os.path.join(new_path,"train2id.txt_bk_tp_duration"),os.path.join(new_path,test_tri_file+"_bk_tp_duration"),
                                 os.path.join(new_path,vaild_tri_file+"_bk_tp_duration")],os.path.join(new_path,"triple2id_duration.txt"))
    
    merge_files_with_line_count([os.path.join(new_path,train_tri_file+"_bk_tp_duration_2id"),os.path.join(new_path,test_tri_file),
                                 os.path.join(new_path,vaild_tri_file)],os.path.join(new_path,"triple2id_duration_2id.txt"))
    
     # 将训练集、测试集、验证集根据年拆成一个个时间点的事件，事件的时间根据id表示，将这个文件合并，表示数据集中所有的事件（事件时间用时间点表示）
    merge_files_with_line_count([os.path.join(new_path,"train2id.txt"),os.path.join(new_path,"test.txt_bk_tp_point_2id"),
                                 os.path.join(new_path,"valid.txt_bk_tp_point_2id")],os.path.join(new_path,"triple2id_point.txt"))
    
    # 删除没有用到的文件
    os.remove(os.path.join(new_path, test_tri_file+"_bk_tp_point"))
    os.remove(os.path.join(new_path, vaild_tri_file+"_bk_tp_point"))
    os.remove(os.path.join(new_path, train_tri_file+"_bk_tp_point"))
    
    os.remove(os.path.join(new_path, train_tri_file+"_bk_tp_duration_2id"))
    
    # os.remove(os.path.join(new_path, test_tri_file+"_bk_tp_point_2id"))
    os.remove(os.path.join(new_path, vaild_tri_file+"_bk_tp_point_2id"))
    
    os.remove(os.path.join(new_path, test_tri_file+"_bk_tp_duration"))
    os.remove(os.path.join(new_path, vaild_tri_file+"_bk_tp_duration"))
    os.remove(os.path.join(new_path, train_tri_file+"_bk_tp_duration"))
    
    # 加载triple2id_duration.txt这个文件，对数据集进行分析，生成约束文件，统计不同类型的关系所对应事件的平均发生事件间隔
    static_r = {}
    with open(os.path.join(new_path,"triple2id_duration.txt"),"r") as file:
        totle = int(file.readline())
        for i in range(totle):
            h,r,t,st,et = file.readline().strip().split()
            if int(r) not in static_r:
                static_r[int(r)] = []
            static_r[int(r)].append(int(et) - int(st) + 1)
    average_values = {k: round(sum(v) / len(v)) for k, v in static_r.items()}
   
    # 保存
    with open(os.path.join(new_path,"r_time_static.txt"),"w",encoding="utf-8") as out:
        out.write(str(len(average_values))+ '\n')
         
        for key, value in average_values.items():
            out.write(f"{key}\t{value}\n")
 
def static_rel_first_and_last_time(first_time_in_rel,last_time_in_rel,file_path):
    #统计每个关系的最早时间和最晚时间
    
    with open(os.path.join(in_path, file_path),"r") as file:
        event_num = int(file.readline())
        for i in range(event_num):
            first_skip_flag,last_skip_flag=1,1
            h,r,t,st,et = file.readline().strip().split()
           
            st = st.split("-")[0]
            et = et.split("-")[0]
            if st == '####':
                st = "0050"
                first_skip_flag = 0
            if et == '####':
                et = "3000"
                last_skip_flag = 0
                
            if st.find("#") != -1 :
                 continue
            if et.find("#") != -1 :
                continue	
             
            
            r = int(r)
            st = int(st)
            et = int(et)
           
            if st > et:
                continue
            
            if first_skip_flag and (r not in first_time_in_rel or  first_time_in_rel[r] > st):
                first_time_in_rel[r] = st
            if last_skip_flag and (r not in first_time_in_rel or  first_time_in_rel[r] > et):
                first_time_in_rel[r] = et
                
            if first_skip_flag and (r not in last_time_in_rel or  last_time_in_rel[r] < st):
                last_time_in_rel[r] = st
            if last_skip_flag and (r not in last_time_in_rel or  last_time_in_rel[r] < et):
                last_time_in_rel[r] = et
      
def static_dataset_first_and_last_time(file_path):
    #统计每个关系的最早时间和最晚时间
    first_time = 5000
    last_time = 0000
    with open(os.path.join(in_path, file_path),"r") as file:
        event_num = int(file.readline())
        for i in range(event_num):
            first_skip_flag,last_skip_flag=1,1
            h,r,t,st,et = file.readline().strip().split()
           
            st = st.split("-")[0]
            
            et = et.split("-")[0]
            if st == '####':
                st = "0000"
                first_skip_flag = 0
            if et == '####':
                et = "5000"
                last_skip_flag = 0
                
            if st.find("#") != -1 :
                 continue
            if et.find("#") != -1 :
                continue	
             
            
            r = int(r)
            st = int(st)
            et = int(et)
           
            if st > et:
                continue
            
            if first_skip_flag :
                first_time = min(first_time,st)
                last_time = max(last_time,st)
            if last_skip_flag :
                first_time = min(first_time,et)
                last_time = max(last_time,et)
                
    return first_time,last_time
                
   

def init_global_dict(optional="default",first_st_limit = None,last_et_limit = None):
    """ 根据数据集统计每个关系的最早时间和最晚时间，分别存储于first_time_in_rel和last_time_in_rel
    optional:default   统计整个数据集的最早和最晚时间 每个关系的最早时间和最晚时间一样，均是整个数据集的最早和最晚
    optional:rel  统计每个关系的最早时间和最晚时间

    Args:
        optional (str, optional): _description_. Defaults to "default".
        first_st_limit (int, optional): _description_. Defaults to "default". 默认需要统计，也可以指定最早的开始时间不能小于该时间
        last_et_limit (int, optional): _description_. Defaults to "default". 默认需要统计，也可以指定最晚的结束时间不能大于该时间
    """
    global first_time_in_rel
    global last_time_in_rel
  
    train_tri_file = "train2id.txt"
    test_tri_file =  "test.txt"
    vaild_tri_file =  "valid.txt"
    rel2id_file = "relation2id.txt"
    with open(os.path.join(in_path, rel2id_file),"r") as file:
        rel_total = int(file.readline())

    if optional == 'default':
        
        # 获取到整个数据集的最早时间和最短时间
        
        first_time_train,last_time_train = static_dataset_first_and_last_time(train_tri_file)
        first_time_test,last_time_test = static_dataset_first_and_last_time(test_tri_file)
        first_time_valid,last_time_valid = static_dataset_first_and_last_time(vaild_tri_file)
        first_last_time = [first_time_train,last_time_train,first_time_test,last_time_test,first_time_valid,last_time_valid]
        first_time_in_dataset = min(first_last_time)
        last_time_in_dataset = max(first_last_time)
        if first_st_limit:
            first_time_in_dataset = max(first_time_in_dataset,first_st_limit)
        if last_et_limit:
            last_time_in_dataset = min(last_time_in_dataset,last_et_limit)
        for i in range(rel_total):
            first_time_in_rel[i] = int(first_time_in_dataset)
            last_time_in_rel[i] = int(last_time_in_dataset)
    else:
        static_rel_first_and_last_time(first_time_in_rel, last_time_in_rel, train_tri_file)
        static_rel_first_and_last_time(first_time_in_rel, last_time_in_rel, test_tri_file)
        static_rel_first_and_last_time(first_time_in_rel, last_time_in_rel, vaild_tri_file)
        # print("first_time_in_relorg :" ,{k: first_time_in_rel[k] for k in list(first_time_in_rel)[:10]})
        # print("last_time_in_rel_org :" ,{k: last_time_in_rel[k] for k in list(last_time_in_rel)[:10]})
        if first_st_limit:
            for key in first_time_in_rel:
                first_time_in_rel[key] = max(first_time_in_rel[key],first_st_limit) 
        if last_et_limit:
            for key in last_time_in_rel:
                last_time_in_rel[key] = min(last_time_in_rel[key],last_et_limit) 
        # print("first_time_in_rel :" ,{k: first_time_in_rel[k] for k in list(first_time_in_rel)[:10]})
        # print("last_time_in_rel :" ,{k: last_time_in_rel[k] for k in list(last_time_in_rel)[:10]})
        if len(first_time_in_rel.keys())!=rel_total or len(last_time_in_rel.keys())!=rel_total:
            for i in range(rel_total):
                if i not in first_time_in_rel:
                    # first_time_in_rel[i] = int(st_dft)
                    print(i)
                if i not in last_time_in_rel:
                    # last_time_in_rel[i] = int(et_dft)
                    print(i)
    print("最早时间：",first_time_in_rel)
    print("最晚时间：",last_time_in_rel)

def analyse(triple_file,optional):
    
    global time_point_formate

    global first_time_in_rel
    global last_time_in_rel 
    global_rel_timenum_dict = {}
    rel_time_dict = {}
    rel_timenum_dict = {}
    # 加载triple2id_duration.txt这个文件，对数据集进行分析，生成约束文件，统计不同类型的关系所对应事件的平均发生事件间隔
    r_duration = {}
    event_total_duration_time = 0
    # if optional != "tp_point":
    #     init_global_dict(optional)
    
    with open(triple_file+"triple2id_duration.txt","r") as file:
        event_num = int(file.readline())
        for i in range(event_num):
            first_skip_flag,last_skip_flag=1,1
            h,r,t,st,et = file.readline().strip().split()
            
            r = int(r)
            st = int(st)
            et = int(et)
            if int(r) not in r_duration:
                r_duration[r] = []
            r_duration[r].append(et-st + 1)
            event_total_duration_time = event_total_duration_time + et-st + 1
            
            if first_skip_flag and (r not in first_time_in_rel or  first_time_in_rel[r] > st):
                first_time_in_rel[r] = st
            if last_skip_flag and (r not in first_time_in_rel or  first_time_in_rel[r] > et):
                first_time_in_rel[r] = et
                
            if first_skip_flag and (r not in last_time_in_rel or  last_time_in_rel[r] < st):
                last_time_in_rel[r] = st
            if last_skip_flag and (r not in last_time_in_rel or  last_time_in_rel[r] < et):
                last_time_in_rel[r] = et
            
            h_tp, r_tp, t_tp, tp = get_time_point(h, r, t, st, et, time_point_formate)
            if int(r) not in rel_time_dict:
                rel_time_dict[int(r)] = set()
            rel_time_dict[int(r)].update([time for time in tp])  
            
    r_mean_duration = {k: round(sum(v) / len(v)) for k, v in r_duration.items()}
    
    
   
    print("缺失值补齐方式：", optional)
    print("不同关系的最早发生时间：",first_time_in_rel)
    print("不同关系的最晚发生时间：", last_time_in_rel)
    for r in first_time_in_rel:
        global_rel_timenum_dict[r] = last_time_in_rel[r]-first_time_in_rel[r]+1
    print("不同关系上的时间跨度：", global_rel_timenum_dict)
    
   
    rel_timenum_dict = {r: len(lst) for r, lst in rel_time_dict.items()}
    
    print("不同关系上时间的跨度为：",rel_timenum_dict)
    # for r, lst in rel_time_dict.items():
    #     print(r,min(lst),max(lst))
        
    with open(triple_file+"r_time_analyse.txt","w",encoding="utf-8") as file:
        file.write("event_mean_duration_time:"+str(round(event_total_duration_time/event_num,3))+"\n")
        file.write("{:<4}\t{:<4}\t{:<4}\t{:<4}\t{:<4}\t{:<4}\n".format("r","first","last","duration","in_time_num","r_mean_duration"))
        for i in range(len(first_time_in_rel)):
            file.write("{:<4}\t{:<4}\t{:<4}\t{:<4}\t{:<4}\t{:<4}\n".format(i,first_time_in_rel[i],last_time_in_rel[i],global_rel_timenum_dict[i],rel_timenum_dict[i],
                        r_mean_duration[i] ))
                     
            # file.write(str(i)+"\t"+str(first_time_in_rel[i])+"\t"+str(last_time_in_rel[i])+"\t"+str(global_rel_timenum_dict[i])+"\t"
            #            +str(rel_timenum_dict[i])+"\t"+str(r_mean_duration[i])+"\n")
        
def gen_test_every_year(file_path,new_file_path):
    """处理test文件，将五元组拆为4元组，临时使用

    Args:
        file_path (_type_): _description_
        new_file_path (_type_): _description_
    """
    
    global time_point_formate
    
    head_tp = []
    tail_tp = []
    rel_tp = []
    tp_point = []
    
    f = open(file_path, "r")
    triples_total = (int)(f.readline())
    for index in range(triples_total):
        # 注意需要与数据集文件每列的含义对齐
        # h,t,r, st, et = f.readline().strip().split() #Openke的提供的默认数据集，每列的含义
        h, r, t, st, et = f.readline().strip().split() #拷贝过来的wiki_data_add_time实验数据，，每列的含义
        
        
        h_tp, r_tp, t_tp, tp = get_time_point(h, r, t, st, et, time_point_formate)
        head_tp.extend(h_tp)
        rel_tp.extend(r_tp)
        tail_tp.extend(t_tp)
        tp_point.extend(tp)	
        # st, et = get_time_formate(st,et)
        # time_point_set =  date_range(st,et,time_point_formate)
        
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
    
    
if __name__ == '__main__':
    # 生成一个时间点2id的文件
    # 初始化global first_time_in_rel 和 global last_time_in_rel
   

    in_path = "/u01/cmp/exp_code/OpenKE/benchmarks/wiki_data_addTime/"
    new_path = "/u01/cmp/exp_code/OpenKE/benchmarks/data/wiki_data_addTime_year_tp2id_filter_bk/"

    # st_dft = "1000"
    # et_dft ="2020"
    time_point_formate = "%Y"


    # yago
    # in_path = "/u01/cmp/exp_code/OpenKE/benchmarks/data/yago/large"
    # new_path = "/u01/cmp/exp_code/OpenKE/benchmarks/data/yago_data_addTime_year_tp2id_filter_bk/"


    process("tp_point")
    
    # gen_test_every_year("/u01/cmp/exp_code/OpenKE/benchmarks/data/yago_data_addTime_year_tp2id_filter2/test.txt",
    #                     "/u01/cmp/exp_code/OpenKE/benchmarks/data/yago_data_addTime_year_tp2id_filter2/test_quadruple.txt")