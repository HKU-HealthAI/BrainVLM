"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
# from imblearn.over_sampling import SMOTE

import datetime
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import webdataset as wds
from minigpt4.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import is_url
from minigpt4.datasets.data_utils import concat_datasets, reorg_datasets_by_split, ChainDataset
from minigpt4.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import subprocess
import os
import json
import math
import pynvml
from collections import Counter
import ast
import random
import time



def create_dir(log_save_path, Date, Version):
    folder_path = f'{log_save_path}/{Date}_{Version}'
    if not os.path.exists(folder_path):  # 检查文件夹是否存在
        os.makedirs(folder_path)  # 不存在则创建
    return folder_path

def detect_GPUs():
    '''
    本方法用于检测可用GPU，并返回可用的GPU list 和数量值
    输出示例：
    available_num 8
    available_indexs ['0', '1', '2', '3', '4', '5', '6', '7'] 
    '''
    pynvml.nvmlInit()  # 初始化NVML库
    device_count = pynvml.nvmlDeviceGetCount() # 获取GPU数量
    available_indexs = []  # 可用的GPU列表
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i) # 获取GPU句柄
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle) # 获取显存信息
        free_mem = mem_info.free / 1024**2
        if free_mem >= 23000:  # 如果GPU显存数大于23000MB，就加入可用列表
            available_indexs.append(str(i))
    pynvml.nvmlShutdown()  # 关闭NVML库
    available_num = len(available_indexs)
    return available_num, available_indexs

def run_script_n_GPUs(files,save_path,root_dir,python_path,script_path,model_pth,Available_GPUS):
    split_num = len(files)
    combined_log_path = f'{save_path}/all.log'
    process_list = []  # 用于记录process数量
    log_list = []
    for i in range(split_num):
        json_path = files[i]
        log_path = f'{save_path}/sub{i+1}.log'
        os.chdir(root_dir)
        with open(log_path, "w") as outfile:
            process = subprocess.Popen(
                [python_path, "-u", script_path, "--input_json", json_path, "--model_pth", model_pth],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=Available_GPUS[i]),
                stdout=outfile,
                stderr=subprocess.STDOUT
            )
        process_list.append(process)
        log_list.append(log_path)
        print(f"Process {i+1} started, PID:", process.pid)
    for process_i in process_list:  # 等待n个进程结束
        process_i.wait()
    with open(combined_log_path, 'w') as combined_file:  # 合并日志文件
        for log_path in log_list:
            with open(log_path, 'r') as f:
                combined_file.write(f.read() + "\n")
    return combined_log_path


def split_file_into_subfiles(path, split_num, splited_dir):
    with open(path,'r') as file:  # 读取原始文件
        data = json.load(file) 
    keys = list(data.keys())  # 获取所有键名的列表
    total_len = len(data)
    chunk_size = math.ceil(total_len / split_num)
    chunks = [keys[i:i + chunk_size] for i in range(0, total_len, chunk_size)]  # 切分原始文件
    saved_files = []
    for i, chunk_keys in enumerate(chunks):  # 保存每个子JSON文件
        chunk_data = {key: data[key] for key in chunk_keys}  # 根据键提取子字典
        output_file = os.path.join(splited_dir, f'chunk_{i + 1}.json')
        saved_files.append(output_file)
        print(output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=4)
    print(f"切分完成，共生成 {len(chunks)} 个子json文件。")
    return saved_files


def caculate_acc_score(pth):
    '''读取合并后的日志文件，输出acc值'''
    with open(pth, 'r') as file:
        log_data = file.readlines()
    v_list=[]
    label_list=['Dysplastic cerebellar gangliocytoma',
        'Chondrosarcoma',
        'Choroid plexus papilloma',
        'Pediatric-type diffuse high-grade gliomas',
        'Glioneuronal and neuronal tumors',
        'Ependymoma',
        'Hemangioma',
        'Solitary fibrous tumour',
        'Pleomorphic xanthoastrocytoma',
        'Diffuse midline glioma',
        'Embryonal tumour with multilayered rosettes',
        'Chordoid glioma',
        'Pineal parenchymal tumour of intermediate differentiation',
        'germ cell tumor of pineal region',
        'Central neurocytoma',
        'Angiocentric glioma',
        'Craniopharyngioma',
        'Rosai Dorfman disease',
        'Pilocytic astrocytoma',
        'Hemangioblastoma',
        'Medulloblastoma',
        'vascular malformations',
        'Polymorphous low-grade neuroepithelial tumor of the young',
        'Chordoma',
        'Subependymal giant cell astrocytoma',
        'Ewing sarcoma',
        'High-grade astrocytoma with piloid features',
        'Pituicytoma',
        'neuroblastoma',
        'Lymphomas',
        'Pineoblastoma',
        'Germinoma',
        'Schwannoma',
        'Dysembryoplastic neuroepithelial tumor',
        'meningioma',
        'Langerhans cell histiocytosis',
        ' Pineocytoma',
        'vascular malformation',
        'Ganglioglioma',
        'Neurofibroma',
        'Atypical choroid plexus papilloma',]
    wrong_patient_list=[]
    big_class=['Meningioma',
            "Cranial",
            "Mesenchymal",
            "Hematolymphoid",
            'pineal',
            'Melanocytic',
            "Embryonal",
            "Tumors of the sellar region",
            "Germ cell",
            "Choroid plexus",
            "Glioma",
            "Metastase"]
    acc_list=[]

    cur_patient_list=[]
    final_total_num=0
    final_total_acc=0
    total_num=0
    total_acc=0
    for tumor in big_class:
        num=0
        acc=0
        precision=0
        num_precision=0
        for i,line in enumerate(log_data):
            if tumor.lower() in line.lower() and 'label' in line.lower() and i+1 < len(log_data) and 'final' in log_data[i+1].lower() :
                num=num+1
                if tumor.lower() in log_data[i+1].lower():
                    acc=acc+1
                    if 'Brain_Tumor_data' in log_data[i-3]:
                        line_list=ast.literal_eval(log_data[i-3])
                        cur_patient=line_list[0].split('/')[-3]
                        cur_patient_list.append(cur_patient)
                else:
                    if 'mnt/7T' in log_data[i-3]:
                        line_list=ast.literal_eval(log_data[i-3])
                        cur_patient=line_list[0].split('/')[-3]
                        wrong_patient_list.append(cur_patient)
            if tumor.lower() in line.lower() and 'answer' in line.lower():
                num_precision=num_precision+1
                if tumor.lower() in log_data[i-1].lower():
                    precision=precision+1
        if num==0:
            continue    
        total_acc=total_acc+acc
        total_num=total_num+num
        print("{} acc:{:2f},acc patient:{}, num:{}".format(tumor,acc/num,acc,num))
        acc_list.append(acc/num)
    random.shuffle(cur_patient_list)
    # print(wrong_patient_list,len(wrong_patient_list))
    # print(total_acc,total_num)
    final_acc = total_acc/total_num
    print('final_acc:{:2f}'.format(final_acc))
    return final_acc

# def my_collate(batch):
#     keys = batch[0].keys()
#     collated_batch = {key: [] for key in keys}
    
#     for sample in batch:
#         for key, value in sample.items():
#             collated_batch[key].append(value)
            
    # for key, value in collated_batch.items():
    #     print(value[0])
    #     if isinstance(value[0], str):
    #         # 如果是字符串类型的数据，不进行转换
    #         continue
    #     collated_batch[key] = torch.tensor(value)
    # return collated_batch

@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id,valid_dataset=None):
        self.config = cfg
        self.job_id = job_id

        self.task = task
        self.datasets = datasets
        
        
        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0
        
        if valid_dataset!=None:
            self.valid_dataset=valid_dataset['xiangya_validation']['train']
            # self.valid_dataset=datasets['xiangya_validation']['train']
        else:
            self.valid_dataset=None
        if self.valid_dataset!=None:
            if self.use_distributed:
                sampler = DistributedSampler(
                    self.valid_dataset,
                    shuffle=True,
                    num_replicas=get_world_size(),
                    rank=get_rank(),
                )
                self.valid_dataloader=DataLoader(
                    self.valid_dataset,
                    batch_size=1,
                    sampler=sampler,
                )
            else:
                self.valid_dataloader=DataLoader(
                    self.valid_dataset,
                    batch_size=1,
                    # sampler=sampler,
                )
        
        # self.setup_seeds()
        self.setup_output_dir()

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu], find_unused_parameters=True
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            num_parameters = 0
            p_wd, p_non_wd = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                print(n)
                if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                    p_non_wd.append(p)
                else:
                    p_wd.append(p)
                num_parameters += p.data.nelement()
            logging.info("number of trainable parameters: %d" % num_parameters)
            optim_params = [
                {
                    "params": p_wd,
                    "weight_decay": float(self.config.run_cfg.weight_decay),
                },
                {"params": p_non_wd, "weight_decay": 0},
            ]
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=float(self.config.run_cfg.init_lr),
                weight_decay=float(self.config.run_cfg.weight_decay),
                betas=(0.9, beta2),
            )

        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = self.max_epoch
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", None)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)
            iters_per_epoch = self.config.run_cfg.get("iters_per_epoch", None)

            if iters_per_epoch is None:
                try:
                    iters_per_epoch = len(self.dataloaders['train'])
                except (AttributeError, TypeError):
                    iters_per_epoch = 10000

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                iters_per_epoch=iters_per_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            batch_sizes = {dataset_name: getattr(self.config.datasets_cfg, dataset_name).batch_size
                           for dataset_name in self.datasets.keys()}
            datasets, batch_sizes = reorg_datasets_by_split(self.datasets, batch_sizes)
            self.datasets = datasets
            # self.datasets = concat_datasets(datasets)

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            batch_sizes = [batch_sizes[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            print("batch sizes", batch_sizes)
            
            collate_fns = []
            sample_list=[]
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    if hasattr(dataset[0],"get_sampler"):
                        # print(1,type(dataset))
                        sample_list.append(dataset[0].get_sampler())
                    else:
                        sample_list.append(None)
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                    # collate_fns.append([my_collate for d in dataset])
                else:
                    if hasattr(dataset,"get_sampler"):
                        # print(2,type(dataset))
                        sample_list.append(dataset.get_sampler())
                    else:
                        sample_list.append(None)
                    collate_fns.append(getattr(dataset, "collater", None))
                    # collate_fns.append(my_collate)
            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                sample_list=sample_list
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def train_loader(self):
        # for k,v in self.dataloaders.items():
        #     print(k)
        train_dataloader = self.dataloaders["train"]
        # smote = SMOTE(sampling_strategy='auto', random_state=42)

        return train_dataloader

    def setup_output_dir(self):
        lib_root = Path(registry.get_path("library_root"))
        output_dir=Path("/disk1/Data/Medical-Ours/Brain_Tumor_data/ckpt/brain_llm") / self.config.run_cfg.output_dir / self.job_id
        # output_dir = lib_root / self.config.run_cfg.output_dir / self.job_id
        # output_dir = lib_root / self.config.run_cfg.output_dir
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            # training phase
            # print(len(self.train_loader))
            # if self.valid_dataset!=None and cur_epoch%1==0 and cur_epoch>0:  # epoch n valid
            #     # self.dataloaders
            #     self.task.evaluation(self.model, self.valid_dataloader, cur_epoch=cur_epoch)
            #     self.train_loader.sample_shuffle(cur_epoch)#random.shuffle(dataloader)
            #     # self.task.evaluation_v2(self.model, self.valid_dataloader, cur_epoch=cur_epoch)
            # self._save_checkpoint(cur_epoch, is_best=False)
            # 进行evaluation
            is_eval_multi_gpus = False  # True False
            if is_eval_multi_gpus:
                # 先保存模型到指定临时位置
                model_no_ddp = self.unwrap_dist_model(self.model)
                param_grad_dic = {
                    k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
                }
                state_dict = model_no_ddp.state_dict()
                for k in list(state_dict.keys()):
                    if k in param_grad_dic.keys() and not param_grad_dic[k]:
                        # delete parameters that do not require gradient
                        del state_dict[k]
                save_obj = {
                    "model": state_dict,
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.config.to_dict(),
                    "scaler": self.scaler.state_dict() if self.scaler else None,
                    "epoch": cur_epoch,
                }
                save_to = os.path.join(
                    "/home/haoran/Yanzhaoshi/MiniGPT-4/Yanzhao_scripts_w_data/training_pth_tmp/",
                    "checkpoint_tmp_.pth",
                )
                # logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
                torch.save(save_obj, save_to)
                print(f"Save Current ckpt {cur_epoch} to tmp: {save_to}")
                

                ### Start 一些基础参数配置 ###
                available_num, available_indexs = detect_GPUs()  # 自动获取GPU状态
                if available_num > 6:  # 限制GPU使用数量，留两张给别的同学，8-2=6
                    available_num = 6
                NumberSplit = available_num  # 自动选择并行的数量
                Available_GPUS = available_indexs[:available_num]  # 检测可使用的GPU
                # Testing_full_json = "/mnt/7T/yinong/crop/Brain_Tumor_data/Brain_Tumor_reformat/integrate/swap_train_test/final_test_1006.json"  # 测试集json
                Testing_full_json = "/home/haoran/Yanzhaoshi/MiniGPT-4/Yanzhao_scripts_w_data/val_sets/final_val_1006_4.json"  # 验证集json
                # Testing_full_json = "/home/haoran/Yanzhaoshi/MiniGPT-4/Yanzhao_scripts_w_data/final_test_1006_toy.json"  # toy版原始json（只有9个样本供调试）
                model_pth = save_to
                splited_dir = '/home/haoran/Yanzhaoshi/MiniGPT-4/Yanzhao_scripts_w_data/splited_jsons'  # 临时保存的测试集子jsons
                script_path = f'/home/haoran/Yanzhaoshi/MiniGPT-4/eval_sub_test_yanzhao.py' # 每个进程调用的python文件
                python_path = '/home/haoran/anaconda3/envs/new_foundation/bin/python'  # 指定 Conda 环境中的 Python 解释器路径
                root_dir = '/home/haoran/Yanzhaoshi/MiniGPT-4'  # 根目录
                log_save_path = "/home/haoran/Yanzhaoshi/MiniGPT-4/logs/test"  # 日志和结果输出路径
                
                Date, Version = datetime_str, "v1" # 配置一些版本信息
                save_path = create_dir(log_save_path, Date, Version)
                ### End 一些基础参数配置 ###

                if not NumberSplit == 0:
                    try:
                        # 首先，切分原始json文件到n份，并把它们存储在临时路径
                        files = split_file_into_subfiles(Testing_full_json, NumberSplit, splited_dir)
                        # 其次，并行训练split后的testset，由不同卡执行，最终返回合并的log
                        combined_log_path = run_script_n_GPUs(files,save_path,root_dir,python_path,script_path,model_pth,Available_GPUS)  # files,save_path,root_dir,python_path,script_path,model_pth,Available_GPUS
                        print("Logs combined in:", combined_log_path)
                        # 最后，计算ACC值 combined_log_path = "/home/haoran/Yanzhaoshi/MiniGPT-4/logs/test/2024.1010_v1/Test_20241010_v1_all.log"
                        values = caculate_acc_score(combined_log_path)
                        print(f"第{cur_epoch}周期的val结果为acc {values}")
                    except:
                        print("特殊原因验证失败，skip掉本周期的val")
                        pass
                else:
                    print("无空闲GPU，skip掉本周期的val")

            if not self.evaluate_only:
                logging.info("Start training")
                train_stats = self.train_epoch(cur_epoch)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0:
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch
                    )
                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics

                                self._save_checkpoint(cur_epoch, is_best=True)

                            val_log.update({"best_epoch": best_epoch})
                            self.log_stats(val_log, split_name)

            else:
                # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                if not self.evaluate_only and (cur_epoch+1) % 1==0:
                    self._save_checkpoint(cur_epoch, is_best=False)

            if self.evaluate_only:
                break

            if self.config.run_cfg.distributed:
                dist.barrier()

        # testing phase
        test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
        self.evaluate(cur_epoch=test_epoch, skip_reload=self.evaluate_only)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        # TODO In validation, you need to compute loss as well as metrics
        # TODO consider moving to model.before_evaluation()
        model = self.unwrap_dist_model(self.model)
        if not skip_reload and cur_epoch == "best":
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )
        results = self.task.evaluation(model, data_loader)

        if results is not None:
            return self.task.after_evaluation(
                val_result=results,
                split_name=split_name,
                epoch=cur_epoch,
            )

    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        sample_list,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn,sample):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset) or isinstance(
                dataset, wds.DataPipeline
            ):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler

                if self.use_distributed:
                    sampler = DistributedSampler(
                        dataset,
                        shuffle=is_train,
                        num_replicas=get_world_size(),
                        rank=get_rank(),
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    if sample:
                        sampler=sample
                    else:
                        sampler = None

                loader = DataLoader(
                    dataset,
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn,sample in zip(
            datasets, batch_sizes, is_trains, collate_fns,sample_list
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                if hasattr(dataset[0], 'sample_ratio') and dataset_ratios is None:
                    dataset_ratios = [d.sample_ratio for d in dataset]
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz[i], is_train, collate_fn[i],sample)
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn,sample)

            loaders.append(loader)

        return loaders

    @main_process
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "epoch": cur_epoch,
        }
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else cur_epoch),
        )
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        try:
            model.load_state_dict(checkpoint["model"])
        except RuntimeError as e:
            logging.warning(
                """
                Key mismatch when loading checkpoint. This is expected if only part of the model is saved.
                Trying to load the model with strict=False.
                """
            )
            model.load_state_dict(checkpoint["model"], strict=False)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        message = self.unwrap_dist_model(self.model).load_state_dict(state_dict,strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        print("resume the checkpoint")
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @main_process
    def log_stats(self, stats, split_name):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
