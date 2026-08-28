from __future__ import print_function, division

import sys
import time
import torch
from torch import nn

from helper.util import AverageMeter, accuracy, reduce_tensor


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    model.train()

    batch_time = AverageMeter()
    losses_moe = AverageMeter()
    losses_x1 = AverageMeter()
    losses_x2 = AverageMeter()
    losses_x4 = AverageMeter()
    losses_total = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_x1 = AverageMeter()
    top5_x1 = AverageMeter()
    top1_x2 = AverageMeter()
    top5_x2 = AverageMeter()
    top1_x4 = AverageMeter()
    top5_x4 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size
    train_phase = "预训" if hasattr(model, '_original_forward') or (
                hasattr(model, 'module') and hasattr(model.module, '_original_forward')) else "联合训练"
    end = time.time()

    for idx, batch_data in enumerate(train_loader):
        if opt.dali is None:
            images, labels = batch_data
        else:
            images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        output_moe, xs, _, _ = model(images, use_moe=True)  # xs = [x1, x2, x4]
        x1, x2, x4 = xs


        loss_moe = criterion(output_moe, labels)
        loss_x1 = criterion(x1, labels)
        loss_x2 = criterion(x2, labels)
        loss_x4 = criterion(x4, labels)


        loss_total = loss_moe + opt.alpha * (loss_x1 + loss_x2 + loss_x4)


        losses_moe.update(loss_moe.item(), images.size(0))
        losses_x1.update(loss_x1.item(), images.size(0))
        losses_x2.update(loss_x2.item(), images.size(0))
        losses_x4.update(loss_x4.item(), images.size(0))
        losses_total.update(loss_total.item(), images.size(0))

        # ===================Metrics=====================

        metrics_moe = accuracy(output_moe, labels, topk=(1, 5))
        top1.update(metrics_moe[0].item(), images.size(0))
        top5.update(metrics_moe[1].item(), images.size(0))

        metrics_x1 = accuracy(x1, labels, topk=(1, 5))
        top1_x1.update(metrics_x1[0].item(), images.size(0))
        top5_x1.update(metrics_x1[1].item(), images.size(0))
        metrics_x2 = accuracy(x2, labels, topk=(1, 5))
        top1_x2.update(metrics_x2[0].item(), images.size(0))
        top5_x2.update(metrics_x2[1].item(), images.size(0))
        metrics_x4 = accuracy(x4, labels, topk=(1, 5))
        top1_x4.update(metrics_x4[0].item(), images.size(0))
        top5_x4.update(metrics_x4[1].item(), images.size(0))

        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        # ========== 修改日志：打印每个分类器的损失 ==========
        if idx % opt.print_freq == 0:
            print('{0} Epoch: [{1}][{2}/{3}]\t'
                  'GPU {4}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss_Total {losses_total.avg:.4f}\t'
                  'Loss_MOE {losses_moe.avg:.4f}\t'
                  'Loss_x1 {losses_x1.avg:.4f}\t'
                  'Loss_x2 {losses_x2.avg:.4f}\t'
                  'Loss_x4 {losses_x4.avg:.4f}\t'
                  'Acc@1 (MOE) {top1.avg:.3f}\t'
                  'Acc@1 (x1) {top1_x1.avg:.3f}\t'
                  'Acc@1 (x2) {top1_x2.avg:.3f}\t'
                  'Acc@1 (x4) {top1_x4.avg:.3f}'.format(
                train_phase, epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                losses_total=losses_total, losses_moe=losses_moe,
                losses_x1=losses_x1, losses_x2=losses_x2, losses_x4=losses_x4,
                top1=top1, top1_x1=top1_x1, top1_x2=top1_x2, top1_x4=top1_x4))
            sys.stdout.flush()


    return (top1.avg, top5.avg, losses_total.avg,
            top1_x1.avg, top5_x1.avg, losses_x1.avg,
            top1_x2.avg, top5_x2.avg, losses_x2.avg,
            top1_x4.avg, top5_x4.avg, losses_x4.avg)


def validate_vanilla(val_loader, model, criterion, opt):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):

            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                    idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg


# def validate_distill(val_loader, module_list, criterion, opt):
#     """validation"""
#
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#
#     # switch to evaluate mode
#     for module in module_list:
#         module.eval()
#
#     model_s = module_list[0]
#     model_t = module_list[-1]
#     n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size
#
#     with torch.no_grad():
#         end = time.time()
#         for idx, batch_data in enumerate(val_loader):
#
#             if opt.dali is None:
#                 images, labels = batch_data
#             else:
#                 images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()
#
#             if opt.gpu is not None:
#                 images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
#             if torch.cuda.is_available():
#                 labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
#
#             # compute output
#             if opt.distill == 'simkd':
#                 feat_s, _ = model_s(images, is_feat=True)
#                 feat_t, _ = model_t(images, is_feat=True)
#                 feat_t = [f.detach() for f in feat_t]
#                 cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
#                 model_t.get_feat_modules()[-1]
#                 _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
#             else:
#                 output = model_s(images)
#             loss = criterion(output, labels)
#             losses.update(loss.item(), images.size(0))
#
#             # ===================Metrics=====================
#             metrics = accuracy(output, labels, topk=(1, 5))
#             top1.update(metrics[0].item(), images.size(0))
#             top5.update(metrics[1].item(), images.size(0))
#             batch_time.update(time.time() - end)
#
#             if idx % opt.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'GPU: {2}\t'
#                       'Time: {batch_time.avg:.3f}\t'
#                       'Loss {loss.avg:.4f}\t'
#                       'Acc@1 {top1.avg:.3f}\t'
#                       'Acc@5 {top5.avg:.3f}'.format(
#                     idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
#                     top1=top1, top5=top5))
#
#     if opt.multiprocessing_distributed:
#         # Batch size may not be equal across multiple gpus
#         total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
#         count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
#         total_metrics = reduce_tensor(total_metrics, 1)  # here world_size=1, because they should be summed up
#         count_metrics = reduce_tensor(count_metrics, 1)
#         ret = []
#         for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
#             ret.append(s / (1.0 * n))
#         return ret
#
#     return top1.avg, top5.avg, losses.avg









def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-2].eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-2]
    T_cls1 = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()


    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()
    # ==========================================================================

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.dali is None:
            if opt.distill in ['crd']:
                images, labels, index, contrast_idx = data
            else:
                images, labels = data
        else:
            images, labels = data[0]['data'], data[0]['label'].squeeze().long()

        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]
            logits_t_origin = logit_t
            logits_t1 = T_cls1(feat_t[-2])

        logits_t_list = [
            logits_t_origin,
            logits_t1
        ]

        logits_s_list = [
            logit_s[0],
            logit_s[1],
        ]


        avg_logit_t = sum(logits_t_list) / len(logits_t_list)
        avg_logit_s = sum(logits_s_list) / len(logits_s_list)

        cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else \
            model_t.get_feat_modules()[-1]


        loss_cls = 0.0
        loss_div = 0.0

        for s_logit, t_logit in zip(logits_s_list, logits_t_list):
            loss_cls += criterion_cls(s_logit, labels)
            loss_div += criterion_div(s_logit, t_logit)
        # ==========================================================================

        loss_cls_avg = criterion_cls(avg_logit_s, labels)
        loss_div_avg = criterion_div(avg_logit_s, avg_logit_t)
        # ==========================================================================
        loss_cls += loss_cls_avg
        loss_div += loss_div_avg

        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'ldrld':
            loss_kd = criterion_kd(logit_s, logit_t, labels)
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)
        elif opt.distill == 'srrl':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
        else:
            raise NotImplementedError(opt.distill)


        if opt.distill == 'ldrld':
            loss = opt.cls * loss_kd
        else:
            loss = opt.cls * (loss_cls) + opt.div * (loss_div) + opt.beta * loss_kd

        losses.update(loss.item(), images.size(0))


        metrics1 = accuracy(logits_s_list[0], labels, topk=(1, 5))
        metrics2 = accuracy(logits_s_list[1], labels, topk=(1, 5))
        metrics_avg = accuracy(sum(logits_s_list) / len(logits_s_list), labels, topk=(1, 5))


        top1_1.update(metrics1[0].item(), images.size(0))
        top5_1.update(metrics1[1].item(), images.size(0))

        top1_2.update(metrics2[0].item(), images.size(0))
        top5_2.update(metrics2[1].item(), images.size(0))

        top1.update(metrics_avg[0].item(), images.size(0))
        top5.update(metrics_avg[1].item(), images.size(0))
        # ==========================================================================
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc1-1: {top1_1.avg:.3f}\t'
                  'Acc1-2: {top1_2.avg:.3f}\t'
                  'Acc1-Avg: {top1.avg:.3f}\t'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses,
                top1_1=top1_1, top1_2=top1_2, top1=top1,
                batch_time=batch_time))
            sys.stdout.flush()


    return top1_1.avg, top1_2.avg, top1.avg, top5_1.avg, top5_2.avg, top5.avg, losses.avg


def validate_distill(val_loader, module_list, criterion, opt):
    """validation: 仅验证学生模型，输出2分支+平均全部指标"""

    batch_time = AverageMeter()
    losses = AverageMeter()


    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_1 = AverageMeter()
    top5_1 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()

    # 评估模式
    for module in module_list:
        module.eval()

    # 仅使用学生模型
    model_s = module_list[0]
    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            if opt.gpu is not None:
                images = images.cuda(opt.gpu, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu, non_blocking=True)


            logits_s_list = model_s(images)
            avg_output = sum(logits_s_list) / len(logits_s_list)
            # ==============================================================

            # 损失
            loss = criterion(avg_output, labels)
            losses.update(loss.item(), images.size(0))


            metrics1 = accuracy(logits_s_list[0], labels, topk=(1, 5))
            metrics2 = accuracy(logits_s_list[1], labels, topk=(1, 5))
            metrics_avg = accuracy(avg_output, labels, topk=(1, 5))

            # 更新指标
            top1_1.update(metrics1[0].item(), images.size(0))
            top5_1.update(metrics1[1].item(), images.size(0))
            top1_2.update(metrics2[0].item(), images.size(0))
            top5_2.update(metrics2[1].item(), images.size(0))
            top1.update(metrics_avg[0].item(), images.size(0))
            top5.update(metrics_avg[1].item(), images.size(0))

            batch_time.update(time.time() - end)

            # 打印信息
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc1-1: {top1_1.avg:.3f}\t'
                      'Acc1-2: {top1_2.avg:.3f}\t'
                      'Acc1-Avg: {top1.avg:.3f}\t'.format(
                    idx, n_batch, opt.gpu, loss=losses,
                    top1_1=top1_1, top1_2=top1_2, top1=top1,
                    batch_time=batch_time))
                sys.stdout.flush()

    if opt.multiprocessing_distributed:
        metrics = torch.tensor([
            top1_1.avg, top1_2.avg, top1.avg,
            top5_1.avg, top5_2.avg, top5.avg,
            losses.avg
        ]).cuda(opt.gpu, non_blocking=True)
        reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
        return reduced.tolist()

    return top1_1.avg, top1_2.avg, top1.avg, top5_1.avg, top5_2.avg, top5.avg, losses.avg
