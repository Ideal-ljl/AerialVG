# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler


from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import  BestMetricHolder
import util.misc as utils
from tqdm import tqdm
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from datasets.odvg import build_odvg
from util.utils import clean_state_dict
from model import build_model
from datasets.odvg import ODVGDataset,make_coco_transforms




def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    # dataset parameters

    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true')

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--dino', default= True,action='store_true')
    parser.add_argument('--top_k',type=int, default= 5)
    parser.add_argument('--batch',type=int, default= 4)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--local-rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    return parser



def load_model(args):
    model = build_model(args)
    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()

    return model

def convert_xyxy(box):
    new_box = torch.zeros_like(box)
    new_box[:, :, 0] = box[:, :, 0] - box[:, :, 2] / 2
    new_box[:, :, 1] = box[:, :, 1] - box[:, :, 3] / 2
    new_box[:, :, 2] = box[:, :, 0] + box[:, :, 2] / 2
    new_box[:, :, 3] = box[:, :, 1] + box[:, :, 3] / 2
    return new_box
def compute_iou(gt_box, candidate_boxes):
    """
    计算 IoU。

    :param gt_box: Ground truth boxes, shape [bs, 4] (相对坐标)
    :param candidate_boxes: 候选 boxes, shape [bs, nq, 4] (相对坐标)
    :return: (first_iou, max_iou) - 第一候选框的 IoU 和最大 IoU
    """
    # 将 gt_box 和 candidate_boxes 进行广播，计算交集和并集
    gt_box = gt_box.unsqueeze(1)  # 形状变为 [bs, 1, 4]
    gt_box = convert_xyxy(gt_box)
    candidate_boxes =  convert_xyxy(candidate_boxes)
    # 计算交集
    inter_x1 = torch.max(gt_box[:, :, 0], candidate_boxes[:, :, 0])  # x_min
    inter_y1 = torch.max(gt_box[:, :, 1], candidate_boxes[:, :, 1])  # y_min
    inter_x2 = torch.min(gt_box[:, :, 2], candidate_boxes[:, :, 2])  # x_max
    inter_y2 = torch.min(gt_box[:, :, 3], candidate_boxes[:, :, 3])  # y_max
    
    # 计算交集的宽和高
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    
    # 计算交集面积
    inter_area = inter_width * inter_height
    
    # 计算 gt_box 和 candidate_boxes 的面积
    gt_area = (gt_box[:, :, 2] - gt_box[:, :, 0]) * (gt_box[:, :, 3] - gt_box[:, :, 1])
    candidate_area = (candidate_boxes[:, :, 2] - candidate_boxes[:, :, 0]) * (candidate_boxes[:, :, 3] - candidate_boxes[:, :, 1])
    
    # 计算并集面积
    union_area = gt_area + candidate_area - inter_area
    
    # 计算 IoU
    iou = inter_area / (union_area + 1e-6)  # 防止除以零

    # 获取第一个候选框的 IoU 和最大 IoU
    first_iou = iou[:, 0]  # 第一个候选框的 IoU
    max_iou = iou.max(dim=1).values  # 每个样本的最大 IoU

    return first_iou, max_iou

def eval_vg(model, data_loader_val, topk, device):
    all_first_iou = []
    all_best_iou = []
    model.eval()
    model = model.to(device)  

    for samples, targets in tqdm(data_loader_val, desc="Evaluating", unit="batch"):
        caption = [t["anno"] for t in targets]

        gt_box = torch.stack([t["boxes"][0] for t in targets]).to(device)

        image = samples.to(device)

        with torch.no_grad():
            outputs = model(image, captions=caption)
        
        logits = outputs["pred_logits"].sigmoid().max(dim=2)[0]  # (bs, nq)
        boxes = outputs["pred_boxes"]  # (bs, nq, 4)

        # 使用 topk 选择每个样本中最后一个维度最大值的前 k 个
        top_values, top_indices = logits.topk(topk, dim=1)  # (bs, k), (bs, k)

        # 创建新的张量来存储选定的 boxes
        boxes_selected = boxes[torch.arange(boxes.size(0)).unsqueeze(1), top_indices]  # 直接选择


        sorted_indices = top_values.argsort(dim=1, descending=True)
        top_values_sorted = top_values.gather(1, sorted_indices)
        boxes_selected_sorted = boxes_selected.gather(1, sorted_indices.unsqueeze(-1).expand(-1, -1, 4)).to(device)

        # 计算 IoU
        first_iou, best_iou = compute_iou(gt_box, boxes_selected_sorted)

        all_first_iou.append(first_iou)
        all_best_iou.append(best_iou)

    # 合并所有批次的 IoU 值
    all_first_iou = torch.cat(all_first_iou)
    all_best_iou = torch.cat(all_best_iou)

    # 计算比例
    first_iou_above_threshold = (all_first_iou > 0.5).float()
    best_iou_above_threshold = (all_best_iou > 0.5).float()

    # 计算整体比例
    first_iou_ratio = first_iou_above_threshold.mean().item()
    best_iou_ratio = best_iou_above_threshold.mean().item()

    return first_iou_ratio, best_iou_ratio


def main(args):
    # print(args.dino)

    utils.setup_distributed(args)


    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)

    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))


    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'test.log'), distributed_rank=args.rank, color=False, name="detr")

    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))


    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    logger.debug("build model ... ...")

    model = load_model(args)
    wo_class_error = False
    model.to(device)
    logger.debug("build model, done.")


    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model._set_static_graph()
        model_without_ddp = model.module



    dataset_val = ODVGDataset("./dataset/vg/images","./dataset/vg/annotation/vg_test_odvg.jsonl",None, transforms=make_coco_transforms('test'))


    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)

    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


    top_1, top_k = eval_vg(model,data_loader_val,topk=args.top_k,device=device)
    logger.info(f'ckpt:{args.pretrain_model_path}, Top_1: {top_1}, Top_k: {top_k}')
 



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
    
