
import argparse
from utils.file_util import ensure_directory_exists

from trans.config import Trainer_AddTime
from trans.config.Tester_AddTime_tp_point import Tester_AddTime
from trans.module.model.TransH2HyTE import  TransH2HyTE
from trans.module.loss import MarginLoss
from trans.module.strategy import NegativeSampling

from trans.data.PyTorchTrainDataLoader_AddTime_tp2id import PyTorchTrainDataLoader
from trans.data.PyTorchTestDataLoader_AddTime_tp2id_tp_point import PyTorchTestDataLoader
import os 

parser = argparse.ArgumentParser(description='HyTE_time_predict')

parser.add_argument('-data_type', dest= "data_type", default ='yago', choices = ['yago','wiki_data',"icews05-15","icews14"], help ='dataset to choose')
parser.add_argument('-dataset_path',dest = 'dataset_path', default = './benchmarks/data/yago_data_addTime_year_tp2id_filter2/',help = 'dataset_path')
parser.add_argument('-neg_sample', 	 dest="M", 		default = 10,   	type=int, 	help='Batch size')
parser.add_argument('-gpu', 	 dest="gpu", 		default='2',			help='GPU to use')
parser.add_argument('-lr',	 dest="lr", 		default=0.01,  type=float,	help='Learning rate')
parser.add_argument('-dim',  dest="dim", 	default = 128,   	type=int, 	help='dimension')

parser.add_argument('-margin', 	 dest="margin", 	default=4,   	type=float, 	help='margin')
parser.add_argument('-batch', 	 dest="batch_size", 	default= 1000,   	type=int, 	help='Batch size')
parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 1000,   	type=int, 	help='Max epochs')
parser.add_argument('-data_process', 	 dest="data_process", 	default= 'time_predict',choices = ["HyTE","tp_point","tp_point1","tp_point2","filter"],   	help='data process means')
parser.add_argument('-save_steps', 	 dest="save_steps", 	default= 25,   type=int,	help='save_steps')

# parser.add_argument('-checkpoint', 	 dest="checkpoint", 	default= "./checkpoint/current.ckpt",  	help='Max epochs')
# parser.add_argument('-transe_checkpoint', 	 dest="transe_checkpoint", default= "./checkpoint_transe_icews05-15_512/train_transr_r_icews05_15_512_6_0.05.ckpt",	help='checkpoint')
# parser.add_argument('-filter', 	 dest="filter", 	default= 1,   type=int,	help='test measure filter')
args = parser.parse_args()
args.checkpoint = "./checkpoint_"+args.data_process + "/HyTE_time_predict/"+args.data_type+"/"+str(args.dim)+"_"+str(args.margin)+"_"+str(args.lr)+"/train"+".ckpt"
# args.transe_checkpoint = "./checkpoint_"+args.data_process + "/transe_"+args.data_type+"/"+str(args.dim)+".ckpt"
# train_dataset_path = args.dataset_path+"_test_tp_point"

print("param",args,"\n")
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) 

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 检验checkpoint和transe_checkpoint所在的文件夹是否存在，如果不存在则创建
# ensure_directory_exists(args.transe_checkpoint)
ensure_directory_exists(args.checkpoint)

# dataloader for training
train_dataloader = PyTorchTrainDataLoader(
    in_path = args.dataset_path, 
	batch_size = args.batch_size,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	sampling_choice= 1,
	neg_ent = 0,
	neg_rel = 0,
 	neg_time=args.M)

# dataloader for valid
valid_dataloader = PyTorchTrainDataLoader(
    # in_path = str(args.dataset_path), 
    train_file=args.dataset_path+"valid.txt",
    ent_file=args.dataset_path+"entity2id.txt",
    rel_file=args.dataset_path+"relation2id.txt",
    tp_file=args.dataset_path+"tp2id.txt",
	# nbatches = 300,
	batch_size = args.batch_size,
	threads = 8, 
	sampling_mode = "normal", 
 	sampling_choice= 1,
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 0,
	neg_rel = 0,
 	neg_time=args.M)

# dataloader for test for valid 
valid_dataloader_for_test = PyTorchTestDataLoader(

	tri_file=args.dataset_path+"triple2id.txt",
    test_tri_file=args.dataset_path+"valid.txt",
    
	threads = 8, 
	sampling_mode = "normal", 
	predict_task="time",
	bern_flag = 0, 
 
 	# 时间测试时，不需要过滤
	# filter_flag = args.filter, 

	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	tp_tot = train_dataloader.get_tp_tot()
	
 )


hyTE = TransH2HyTE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
 	tp_tot = train_dataloader.get_tp_tot(),
	dim = int(args.dim),
	p_norm = 1, 
	norm_flag = True
 	)

model_r = NegativeSampling(
	model = hyTE,
	loss = MarginLoss(margin = float(args.margin)),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
trainer = Trainer_AddTime(model = model_r, data_loader = train_dataloader, train_times = int(args.max_epochs), alpha = float(args.lr), use_gpu = True
                        #   ,save_steps=25, checkpoint_dir = str(args.checkpoint).split(".ckpt")[0], valid_data_loader_for_loss = valid_dataloader,valid_data_loader_for_test=valid_dataloader_for_test
                          )
train_tester = Tester_AddTime(model = hyTE, data_loader = valid_dataloader_for_test, use_gpu = True,constraint_file = args.dataset_path+"r_time_static.txt")

trainer.run(tester = train_tester,save_steps=args.save_steps, checkpoint_dir = str(args.checkpoint).split(".ckpt")[0],valid_data_loader_for_loss = valid_dataloader)

# trainer.run()
hyTE.save_checkpoint(str(args.checkpoint))
# transr.save_checkpoint('./checkpoint/train_transr_r_wiki_data_addTime_tp2id_year_180_10_0.05.ckpt')


# dataloader for test
test_dataloader = PyTorchTestDataLoader(

	in_path = args.dataset_path,
	threads = 8, 
	sampling_mode = "normal", 
 	predict_task="time",
	
	ent_tot = hyTE.ent_tot,
	rel_tot = hyTE.rel_tot,
 	tp_tot = train_dataloader.get_tp_tot()
 )
print("test……")
# test the model
hyTE.load_checkpoint(str(args.checkpoint))
tester = Tester_AddTime(model = hyTE, data_loader = test_dataloader, use_gpu = True, constraint_file = args.dataset_path+"r_time_static.txt")

tester.run_link_prediction()
