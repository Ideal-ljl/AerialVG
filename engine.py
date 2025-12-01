# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import to_device
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.cocogrounding_eval import CocoGroundingEvaluator

from datasets.panoptic_eval import PanopticEvaluator

def convert_xyxy(box):
    new_box = torch.zeros_like(box)
    new_box[:, :, 0] = box[:, :, 0] - box[:, :, 2] / 2
    new_box[:, :, 1] = box[:, :, 1] - box[:, :, 3] / 2
    new_box[:, :, 2] = box[:, :, 0] + box[:, :, 2] / 2
    new_box[:, :, 3] = box[:, :, 1] + box[:, :, 3] / 2
    return new_box

def iou_loss(candidate_boxes, scores, gt_box):
    """
    Compute the maximum score for boxes with IoU > 0.5 with the ground-truth box.

    Args:
        candidate_boxes (Tensor): Predicted boxes with shape [bs, nq, 4].
        scores (Tensor): Scores of predicted boxes with shape [bs, nq].
        gt_box (Tensor): Ground-truth box with shape [bs, 4].

    Returns:
        Tensor: Computed loss value.
    """
    bs, nq, _ = candidate_boxes.shape
    gt_box = gt_box.unsqueeze(1)  # Expand to [bs, 1, 4]

    # Convert to [x_min, y_min, x_max, y_max] format if needed
    gt_box = convert_xyxy(gt_box)
    candidate_boxes = convert_xyxy(candidate_boxes)

    # Calculate intersection
    inter_x1 = torch.max(gt_box[:, :, 0], candidate_boxes[:, :, 0])
    inter_y1 = torch.max(gt_box[:, :, 1], candidate_boxes[:, :, 1])
    inter_x2 = torch.min(gt_box[:, :, 2], candidate_boxes[:, :, 2])
    inter_y2 = torch.min(gt_box[:, :, 3], candidate_boxes[:, :, 3])

    # Calculate intersection area
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_width * inter_height

    # Calculate areas of GT and candidate boxes
    gt_area = (gt_box[:, :, 2] - gt_box[:, :, 0]) * (gt_box[:, :, 3] - gt_box[:, :, 1])
    candidate_area = (candidate_boxes[:, :, 2] - candidate_boxes[:, :, 0]) * (candidate_boxes[:, :, 3] - candidate_boxes[:, :, 1])

    # Calculate union area
    union_area = gt_area + candidate_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)  # [bs, nq]

    # Mask for boxes with IoU > 0.5
    mask = iou > 0.5  # Shape: [bs, nq]
    losses = []

    # Iterate over each batch
    for b in range(bs):
        if mask[b].any():
            # Select scores of boxes with IoU > 0.5
            selected_scores = scores[b][mask[b]]
            # Get maximum score
            max_score = selected_scores.max()
            # Use negative of the maximum score as the loss to maximize it during training
            # losses.append(-torch.log(max_score + 1e-6))
            losses.append(-max_score )
        else:
            # If no boxes have IoU > 0.5, add a default penalty (e.g., 1.0)
            losses.append(torch.tensor(1.0, device=candidate_boxes.device))

    # Compute average loss over the batch
    return torch.stack(losses).mean()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)


    model.train()

    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        # print(type(targets))
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        # 补充实验
        # gt_box = torch.stack([t["boxes"][0] for t in targets]).to(device)
        
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples, captions=captions)
            loss_dict = criterion(outputs, targets, cap_list, captions)


           
            # relation_loss = iou_loss(outputs["pred_boxes"], outputs["pred_logits"].sigmoid().max(dim=2)[0],gt_box)
            #  # 将新损失加入 loss_dict
            # loss_dict['new_loss'] = relation_loss

            weight_dict = criterion.weight_dict
            # 新损失权重
            # if 'new_loss' not in weight_dict:
            #     weight_dict['new_loss'] = 1.0  # 设置新损失的权重，比如 1.0，可根据需求调整


            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    
    coco_evaluator = CocoGroundingEvaluator(base_ds, iou_types, useCats=useCats)


    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only

    if args.use_coco_eval:
        from pycocotools.coco import COCO
        coco = COCO(args.coco_val_path)

        # 获取所有类别
        category_dict = coco.loadCats(coco.getCatIds())
        cat_list = [item['name'] for item in category_dict]
    else:
        cat_list=args.label_list
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        bs = samples.tensors.shape[0]
        input_captions = [caption] * bs
        with torch.cuda.amp.autocast(enabled=args.amp):

            outputs = model(samples, captions=input_captions)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:



            for i, (tgt, res) in enumerate(zip(targets, results)):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                _res_bbox = res['boxes']
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
       

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]



    return stats, coco_evaluator


