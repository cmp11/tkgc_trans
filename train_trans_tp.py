
import argparse
from utils.file_util import ensure_directory_exists
from trans.module.model import TransE_AddTime
import trans
from trans.config import Trainer_AddTime,Tester_AddTime
from trans.config.Tester_AddTime_tp_point import Tester_AddTime as Tester_AddTime_mean
from trans.module.model.TransR_tp_AddTime import  TransR_tp_AddTime
from trans.module.loss import MarginLoss
from trans.module.strategy import NegativeSampling
from trans.data import TrainDataLoader, TestDataLoader

from trans.data.PyTorchTrainDataLoader_AddTime_tp2id import PyTorchTrainDataLoader
from trans.data.PyTorchTestDataLoader_AddTime_tp2id import PyTorchTestDataLoader
from trans.data.PyTorchTestDataLoader_AddTime_tp2id_tp_point import PyTorchTestDataLoader as PyTorchTestDataLoader_quintuple
import os 

parser = argparse.ArgumentParser(description='tranr_tp')

parser.add_argument('-data_type', dest= "data_type", default ='yago', choices = ['yago','wiki_data',"icews05-15","icews14","mmbk_wikidata"], help ='dataset to choose')
parser.add_argument('-dataset_path',dest = 'dataset_path', default = './benchmarks/data/wiki_data_addTime_year_tp2id_HyTE_process/',help = 'dataset_path')
parser.add_argument('-neg_sample', 	 dest="M", 		default = 10,   	type=int, 	help='Batch size')
parser.add_argument('-gpu', 	 dest="gpu", 		default='1',			help='GPU to use')
parser.add_argument('-lr',	 dest="lr", 		default=0.0001,  type=float,	help='Learning rate')
parser.add_argument('-dim',  dest="dim", 	default = 128,   	type=int, 	help='dimension')

parser.add_argument('-margin', 	 dest="margin", 	default=4,   	type=float, 	help='margin')
parser.add_argument('-batch', 	 dest="batch_size", 	default= 1000,   	type=int, 	help='Batch size')
parser.add_argument('-epoch', 	 dest="max_epochs", 	default= 1000,   	type=int, 	help='Max epochs')
parser.add_argument('-data_process', 	 dest="data_process", 	 choices = ["tp_point","tp_point1","tp_point2","filter"],   	help='data process means')
# parser.add_argument('-checkpoint', 	 dest="checkpoint", 	default= "./checkpoint/current.ckpt",  	help='Max epochs')
# parser.add_argument('-transe_checkpoint', 	 dest="transe_checkpoint", default= "./checkpoint_transe_icews05-15_512/train_transr_r_icews05_15_512_6_0.05.ckpt",	help='checkpoint')
parser.add_argument('-filter', 	 dest="filter", 	default= 1,   type=int,	help='test measure filter')
parser.add_argument('-save_steps', 	 dest="save_steps", 	default= 25,   type=int,	help='save_steps')


args = parser.parse_args()
args.checkpoint = "./checkpoint_"+args.data_process + "/transrt_tp/"+args.data_type+"/"+str(args.dim)+"_"+str(args.margin)+"_"+str(args.lr)+"/train"+".ckpt"
args.transe_checkpoint = "./checkpoint_"+args.data_process + "/transe/"+args.data_type+"/"+str(args.dim)+".ckpt"


print("param",args,"\n")
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu) 

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 检验checkpoint和transe_checkpoint所在的文件夹是否存在，如果不存在则创建
ensure_directory_exists(args.transe_checkpoint)
ensure_directory_exists(args.checkpoint)

# dataloader for training
train_dataloader = PyTorchTrainDataLoader(
    in_path = str(args.dataset_path), 
	batch_size = args.batch_size,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	sampling_choice= 1,
	neg_ent = int(args.M),
	neg_rel = 0)

# dataloader for valid
valid_dataloader = PyTorchTrainDataLoader(
    # in_path = str(args.dataset_path), 
    train_file=str(args.dataset_path)+"valid.txt",
    ent_file=str(args.dataset_path)+"entity2id.txt",
    rel_file=str(args.dataset_path)+"relation2id.txt",
    tp_file=str(args.dataset_path)+"tp2id.txt",
	# nbatches = 300,
	batch_size = args.batch_size,
	threads = 8, 
	sampling_mode = "normal", 
 	sampling_choice= 1,
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = int(args.M),
	neg_rel = 0)

# dataloader for test for valid 
valid_dataloader_for_test = PyTorchTestDataLoader(

	tri_file=str(args.dataset_path)+"triple2id.txt",
    test_tri_file=str(args.dataset_path)+"valid.txt",
    
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = args.filter, 

	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot()
	
 )

# define the model
transe = TransE_AddTime(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	tp_tot = train_dataloader.get_tp_tot(),

	dim = int(args.dim), 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = float(args.margin)),
	batch_size = train_dataloader.get_batch_size())

transr = TransR_tp_AddTime(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
 	tp_tot = train_dataloader.get_tp_tot(),
	
	dim_e = int(args.dim),
	dim_r = int(args.dim),
	dim_tp = int(args.dim),
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = float(args.margin)),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
trainer = Trainer_AddTime(model = model_e, data_loader = train_dataloader, train_times = 500, alpha = 0.1, use_gpu = True)
trainer.run()
# transe.load_checkpoint(str(args.transe_checkpoint))
# transe.load_parameters(str(args.transe_checkpoint))
parameters = transe.get_parameters()
transe.save_checkpoint(str(args.transe_checkpoint))
# transe.save_parameters(str(args.transe_checkpoint))
# print(parameters)
# # train transr
transr.set_parameters(parameters)
trainer = Trainer_AddTime(model = model_r, data_loader = train_dataloader, train_times = int(args.max_epochs), alpha = float(args.lr), use_gpu = True
                        #   ,save_steps=10, checkpoint_dir = str(args.checkpoint).split(".ckpt")[0], valid_data_loader_for_loss = valid_dataloader,valid_data_loader_for_test=valid_dataloader_for_test
                          )
train_tester = Tester_AddTime(model = transr, data_loader = valid_dataloader_for_test, use_gpu = True)

trainer.run(tester = train_tester,save_steps=args.save_steps, checkpoint_dir = str(args.checkpoint).split(".ckpt")[0],valid_data_loader_for_loss = valid_dataloader)

# trainer.run()
transr.save_checkpoint(str(args.checkpoint))
# transr.save_checkpoint('./checkpoint/train_transr_r_wiki_data_addTime_tp2id_year_180_10_0.05.ckpt')


# dataloader for test
test_dataloader = PyTorchTestDataLoader_quintuple(
    # in_path = "./benchmarks/data/wiki_data_addTime_year_tp2id_HyTE_process/", 
    in_path = str(args.dataset_path), 
	 
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = args.filter, 
	# neg_ent = transe.ent_tot*2,
	# neg_rel = 0,
	ent_tot = transr.ent_tot,
	rel_tot = transr.rel_tot
 )
print("test……")
# test the model
transr.load_checkpoint(str(args.checkpoint))
tester = Tester_AddTime_mean(model = transr, data_loader = test_dataloader, use_gpu = True)

tester.run_link_prediction()

