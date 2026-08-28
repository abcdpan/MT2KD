"""
the general training framework (modified: 移除学生模型 + 冻结教师 + 训练自定义分类器 + 单独展示x1/x2分支指标)
"""

from __future__ import print_function

import os

from models.util import Classification

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import re
import argparse
import time

import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger

# from dataset.imagenet_dali import get_dali_data_loader
from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader,  get_dataloader_sample

from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate
from distiller_zoo import DistillKL

# ===================== 训练函数 =====================
def train_classifier(epoch, train_loader, teacher, classifier, criterion_ce, criterion_kl, optimizer, opt):
    """
    仅训练分类器：教师冻结提特征/logit，分类器接收最后一层特征，计算CE+KL损失
    返回：x1_top1, x1_top5, x2_top1, x2_top5, 总损失（不再平均）
    """
    classifier.train()  # 分类器训练模式
    teacher.eval()      # 教师强制eval，冻结BN/Dropout

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_x1, top5_x1 = AverageMeter(), AverageMeter()
    top1_x2, top5_x2 = AverageMeter(), AverageMeter()

    end = time.time()
    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    for idx, batch_data in enumerate(train_loader):
        # 1. 数据加载
        if opt.dali is None:
            images, labels = batch_data
        else:
            images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

        # 设备迁移
        if opt.gpu is not None:
            images = images.cuda(opt.gpu, non_blocking=True)
            labels = labels.cuda(opt.gpu, non_blocking=True)

        # 2. 教师前向（无梯度）
        with torch.no_grad():
            feat_list, logit_list = teacher(images, is_feat=True)
            t_last_feat = feat_list[-2]  # 教师最后一层特征
            t_logit = logit_list     # 教师最终输出logit

        # # 3. 分类器前向
        # cls_x1, cls_x2 = classifier(t_last_feat)
        #
        # # 4. 损失计算：CE(x1/labels) + CE(x2/labels) + KL(x1/t_logit) + KL(x2/t_logit)
        # loss_ce_x1 = criterion_ce(cls_x1, labels)
        # loss_ce_x2 = criterion_ce(cls_x2, labels)
        # loss_kl_x1 = criterion_kl(cls_x1, t_logit)
        # loss_kl_x2 = criterion_kl(cls_x2, t_logit)

        # 3. 分类器前向
        cls_x1 = classifier(t_last_feat)

        # 4. 损失计算：CE(x1/labels) + CE(x2/labels) + KL(x1/t_logit) + KL(x2/t_logit)
        loss_ce_x1 = criterion_ce(cls_x1, labels)
        loss_kl_x1 = criterion_kl(cls_x1, t_logit)

        # 总损失：CE损失 + KL损失（可通过参数调整权重，这里默认1:1）
        # loss = (loss_ce_x1 + loss_ce_x2) + (loss_kl_x1 + loss_kl_x2)
        loss = (loss_ce_x1) + (loss_kl_x1)
        losses.update(loss.item(), images.size(0))

        # 5. 指标统计（x1/x2的Top1/Top5）
        acc1_x1, acc5_x1 = accuracy(cls_x1, labels, topk=(1,5))
        top1_x1.update(acc1_x1.item(), images.size(0))
        top5_x1.update(acc5_x1.item(), images.size(0))

        # acc1_x2, acc5_x2 = accuracy(cls_x2, labels, topk=(1,5))
        # top1_x2.update(acc1_x2.item(), images.size(0))
        # top5_x2.update(acc5_x2.item(), images.size(0))

        acc1_x2, acc5_x2 = accuracy(cls_x1, labels, topk=(1,5))
        top1_x2.update(acc1_x2.item(), images.size(0))
        top5_x2.update(acc5_x2.item(), images.size(0))

        # 6. 反向传播（仅更新分类器）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. 日志打印（展示各分支指标，无平均）
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'GPU {3}\tTime {batch_time.avg:.3f}\tLoss {loss.avg:.4f}\t'
                  'Acc@1(x1) {x1:.3f}\tAcc@5(x1) {x1_5:.3f}\t'
                  'Acc@1(x2) {x2:.3f}\tAcc@5(x2) {x2_5:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                x1=top1_x1.avg, x1_5=top5_x1.avg, x2=top1_x2.avg, x2_5=top5_x2.avg))

    # 返回各分支指标，不平均
    return top1_x1.avg, top5_x1.avg, top1_x2.avg, top5_x2.avg, losses.avg

# ===================== 验证函数 =====================
def validate_classifier(val_loader, teacher, classifier, criterion_ce, criterion_kl, opt):
    """
    验证分类器：无梯度，逻辑与训练一致，返回各分支指标
    """
    classifier.eval()
    teacher.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1_x1, top5_x1 = AverageMeter(), AverageMeter()
    top1_x2, top5_x2 = AverageMeter(), AverageMeter()

    end = time.time()
    n_batch = len(val_loader) if opt.dali is None else (val_loader._size + opt.batch_size - 1) // opt.batch_size

    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            # 1. 数据加载
            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            # 设备迁移
            if opt.gpu is not None:
                images = images.cuda(opt.gpu, non_blocking=True)
                labels = labels.cuda(opt.gpu, non_blocking=True)

            # 2. 教师前向
            feat_list, logit_list = teacher(images, is_feat=True)
            t_last_feat = feat_list[-2]
            t_logit = logit_list

            # # 3. 分类器前向
            # cls_x1, cls_x2 = classifier(t_last_feat)
            #
            # # 4. 损失计算
            # loss_ce_x1 = criterion_ce(cls_x1, labels)
            # loss_ce_x2 = criterion_ce(cls_x2, labels)
            # loss_kl_x1 = criterion_kl(cls_x1, t_logit)
            # loss_kl_x2 = criterion_kl(cls_x2, t_logit)
            # loss = (loss_ce_x1 + loss_ce_x2) + (loss_kl_x1 + loss_kl_x2)
            # losses.update(loss.item(), images.size(0))
            # 3. 分类器前向
            cls_x1 = classifier(t_last_feat)

            # 4. 损失计算
            loss_ce_x1 = criterion_ce(cls_x1, labels)
            loss_kl_x1 = criterion_kl(cls_x1, t_logit)
            loss = (loss_ce_x1) + (loss_kl_x1)
            losses.update(loss.item(), images.size(0))

            # 5. 指标统计
            acc1_x1, acc5_x1 = accuracy(cls_x1, labels, topk=(1,5))
            top1_x1.update(acc1_x1.item(), images.size(0))
            top5_x1.update(acc5_x1.item(), images.size(0))

            acc1_x2, acc5_x2 = accuracy(cls_x1, labels, topk=(1,5))
            top1_x2.update(acc1_x2.item(), images.size(0))
            top5_x2.update(acc5_x2.item(), images.size(0))

            # 6. 日志打印（展示各分支指标，无平均）
            batch_time.update(time.time() - end)
            end = time.time()
            if idx % opt.print_freq == 0:
                print('Val: [{0}/{1}]\t'
                      'GPU {2}\tTime {batch_time.avg:.3f}\tLoss {loss.avg:.4f}\t'
                      'Acc@1(x1) {x1:.3f}\tAcc@5(x1) {x1_5:.3f}\t'
                      'Acc@1(x2) {x2:.3f}\tAcc@5(x2) {x2_5:.3f}'.format(
                    idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                    x1=top1_x1.avg, x1_5=top5_x1.avg, x2=top1_x2.avg, x2_5=top5_x2.avg))

    # 分布式指标汇总（返回各分支指标，不平均）
    if opt.multiprocessing_distributed:
        total_metrics = torch.tensor([
            top1_x1.sum, top5_x1.sum, top1_x2.sum, top5_x2.sum, losses.sum
        ]).to(opt.gpu)
        count_metrics = torch.tensor([
            top1_x1.count, top5_x1.count, top1_x2.count, top5_x2.count, losses.count
        ]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1)
        count_metrics = reduce_tensor(count_metrics, 1)

        # 分布式下返回各分支原始指标
        x1_top1 = (total_metrics[0]/count_metrics[0]).item()
        x1_top5 = (total_metrics[1]/count_metrics[1]).item()
        x2_top1 = (total_metrics[2]/count_metrics[2]).item()
        x2_top5 = (total_metrics[3]/count_metrics[3]).item()
        avg_loss = (total_metrics[4]/count_metrics[4]).item()
        return x1_top1, x1_top5, x2_top1, x2_top5, avg_loss

    # 非分布式返回各分支原始指标
    return top1_x1.avg, top5_x1.avg, top1_x2.avg, top5_x2.avg, losses.avg

# ===================== 加载教师并冻结 =====================
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=30, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')
    parser.add_argument('--path_t', type=str, default='', help='teacher model snapshot')
    parser.add_argument('--cls_channels', type=int, default=256, help='分类器输入通道数（教师最后一层特征通道数）')
    parser.add_argument('--cls_size', type=int, default=1, help='')

    # distillation
    parser.add_argument('--trial', type=str, default='001', help='trial id')
    parser.add_argument('--kd_T', type=float, default=9, help='temperature for KD distillation')

    # multiprocessing
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    opt = parser.parse_args()

    # 调整学习率
    opt.lr_decay_epochs = list(map(int, opt.lr_decay_epochs.split(',')))

    # 调整模型保存路径
    opt.model_t = get_teacher_name(opt.path_t)
    opt.model_name = f'Classifier_T_{opt.model_t}_{opt.dataset}_T{opt.kd_T}_{opt.trial}'
    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    # 保存路径
    opt.model_path = './save/classifier/models'
    opt.tb_path = './save/classifier/tensorboard'
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    os.makedirs(opt.tb_folder, exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    split_symbol = '~' if os.name == 'nt' else ':'
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]

def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model (freeze all params)')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}

    state_dict = torch.load(model_path, map_location=map_location)
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)

    # 冻结教师所有参数
    for param in model.parameters():
        param.requires_grad = False
    model.eval()  # eval模式
    print('==> teacher model frozen completely')
    return model

# ===================== 工具类 =====================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # 修复：确保output是2维的 [B, num_classes]
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        if len(target.shape) > 1:
            target = target.squeeze()

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ===================== 主函数修改 =====================
best_x1_acc = 0
best_x2_acc = 0
total_time = time.time()

def main():
    opt = parse_option()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_x1_acc, best_x2_acc, total_time
    opt.gpu = int(gpu) if gpu is not None else 0
    opt.gpu_id = opt.gpu

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    # 分布式设置
    if opt.multiprocessing_distributed:
        opt.rank = gpu
        dist.init_process_group(backend='nccl', init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    # 确定性训练
    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)

    # 加载教师模型（冻结）
    n_cls = 100 if opt.dataset == 'cifar100' else 1000
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)

    # 初始化自定义分类器
    model_t.cuda(opt.gpu)
    with torch.no_grad():
        # 使用实际的batch_size创建dummy data，避免维度错误
        dummy_data = torch.randn(opt.batch_size, 3, 32, 32).cuda(opt.gpu) if opt.dataset == 'cifar100' else torch.randn(opt.batch_size, 3, 224, 224).cuda(opt.gpu)
        feat_list, _ = model_t(dummy_data, is_feat=True)
        t_last_feat_channels = feat_list[-2].shape[1]
    print(f'==> 自动获取教师最后一层特征通道数：{t_last_feat_channels}')
    classifier = Classification(
        channels=t_last_feat_channels,
        num_classes=n_cls,
        size=opt.cls_size
    ).cuda(opt.gpu)

    # 定义损失函数
    criterion_ce = nn.CrossEntropyLoss().cuda(opt.gpu)  # 分类CE损失
    criterion_kl = DistillKL(opt.kd_T).cuda(opt.gpu)    # KL散度损失

    # 优化器仅传入分类器参数
    optimizer = optim.SGD(classifier.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # 分布式适配
    if opt.multiprocessing_distributed:
        classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[opt.gpu])
        model_t = torch.nn.parallel.DistributedDataParallel(model_t, device_ids=[opt.gpu])

    # 数据加载
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                            num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset,
                                                                              batch_size=opt.batch_size,
                                                                              num_workers=opt.num_workers,
                                                                              multiprocessing_distributed=opt.multiprocessing_distributed)
        else:
            pass
    else:
        raise NotImplementedError(opt.dataset)

    # TensorBoard日志
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # 验证教师准确率（可选）- 修复维度问题
    if not opt.skip_validation:
        try:
            teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_ce, opt)
            if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
                print(f'teacher accuracy: Top1={teacher_acc:.3f}')
        except Exception as e:
            print(f"验证教师模型时出现错误：{e}，跳过教师模型验证")

    # 训练主循环
    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed and opt.dali is None:
            train_sampler.set_epoch(epoch)

        # 调整学习率
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training classifier... Epoch: {}".format(epoch))

        # 训练分类器
        time1 = time.time()
        train_x1_top1, train_x1_top5, train_x2_top1, train_x2_top5, train_loss = train_classifier(
            epoch, train_loader, model_t, classifier, criterion_ce, criterion_kl, optimizer, opt
        )
        time2 = time.time()

        # 分布式指标汇总
        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_x1_top1, train_x1_top5, train_x2_top1, train_x2_top5, train_loss]).cuda(opt.gpu)
            reduced = reduce_tensor(metrics, opt.world_size)
            train_x1_top1, train_x1_top5, train_x2_top1, train_x2_top5, train_loss = reduced.tolist()

        # 训练日志打印各分支指标
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, GPU {}, Train - '
                  'x1_Acc@1 {:.3f}, x1_Acc@5 {:.3f}, '
                  'x2_Acc@1 {:.3f}, x2_Acc@5 {:.3f}, '
                  'Loss {:.4f}, Time {:.2f}'.format(
                epoch, opt.gpu, train_x1_top1, train_x1_top5,
                train_x2_top1, train_x2_top5, train_loss, time2 - time1))

            # TensorBoard记录各分支指标
            logger.log_value('train_x1_acc1', train_x1_top1, epoch)
            logger.log_value('train_x1_acc5', train_x1_top5, epoch)
            logger.log_value('train_x2_acc1', train_x2_top1, epoch)
            logger.log_value('train_x2_acc5', train_x2_top5, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        # 验证分类器
        print('GPU %d validating classifier' % (opt.gpu))
        val_x1_top1, val_x1_top5, val_x2_top1, val_x2_top5, val_loss = validate_classifier(
            val_loader, model_t, classifier, criterion_ce, criterion_kl, opt
        )

        # DALI数据加载器重置
        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        # 验证日志打印各分支指标
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' ** Val - '
                  'x1_Acc@1 {:.3f}, x1_Acc@5 {:.3f}, '
                  'x2_Acc@1 {:.3f}, x2_Acc@5 {:.3f}, '
                  'Loss {:.4f}'.format(
                val_x1_top1, val_x1_top5, val_x2_top1, val_x2_top5, val_loss))

            # TensorBoard记录验证各分支指标
            logger.log_value('val_x1_acc1', val_x1_top1, epoch)
            logger.log_value('val_x1_acc5', val_x1_top5, epoch)
            logger.log_value('val_x2_acc1', val_x2_top1, epoch)
            logger.log_value('val_x2_acc5', val_x2_top5, epoch)
            logger.log_value('val_loss', val_loss, epoch)

            # 保存各分支最佳模型
            is_best = False
            current_best_info = ""
            if val_x1_top1 > best_x1_acc:
                best_x1_acc = val_x1_top1
                is_best = True
                current_best_info += f"x1最佳Acc@1更新至{best_x1_acc:.3f} "
            if val_x2_top1 > best_x2_acc:
                best_x2_acc = val_x2_top1
                is_best = True
                current_best_info += f"x2最佳Acc@1更新至{best_x2_acc:.3f}"

            if is_best:
                state = {
                    'epoch': epoch,
                    'classifier': classifier.module.state_dict() if opt.multiprocessing_distributed else classifier.state_dict(),
                    'best_x1_acc': best_x1_acc,
                    'best_x2_acc': best_x2_acc,
                    'feat_channels': t_last_feat_channels,
                    'cls_size': opt.cls_size
                }
                save_file = os.path.join(opt.save_folder, 'classifier_best.pth')
                # 保存各分支验证指标
                test_metrics = {
                    'val_loss': val_loss,
                    'val_x1_acc1': val_x1_top1,
                    'val_x1_acc5': val_x1_top5,
                    'val_x2_acc1': val_x2_top1,
                    'val_x2_acc5': val_x2_top5,
                    'best_x1_acc': best_x1_acc,
                    'best_x2_acc': best_x2_acc,
                    'epoch': epoch
                }
                save_dict_to_json(test_metrics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                print(f'saving the best classifier! {current_best_info}')
                torch.save(state, save_file)

    # 训练结束打印各分支最佳指标
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        print(f'分类器训练完成 - 最佳x1_Acc@1: {best_x1_acc:.3f}, 最佳x2_Acc@1: {best_x2_acc:.3f}')
        # 保存参数
        save_state = {k: v for k, v in opt._get_kwargs()}
        num_params = (sum(p.numel() for p in classifier.parameters())/1000000.0)
        save_state['Classifier params(M)'] = num_params
        save_state['Total time(h)'] = (time.time() - total_time)/3600.0
        save_state['Best x1 Acc@1'] = best_x1_acc
        save_state['Best x2 Acc@1'] = best_x2_acc
        save_dict_to_json(save_state, os.path.join(opt.save_folder, "parameters.json"))

# ===================== validate_vanilla =====================
def validate_vanilla(val_loader, model, criterion, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    end = time.time()
    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            if opt.dali is None:
                images, labels = batch_data
            else:
                images, labels = batch_data[0]['data'], batch_data[0]['label'].squeeze().long()

            # 设备迁移
            if opt.gpu is not None:
                images = images.cuda(opt.gpu, non_blocking=True)
                labels = labels.cuda(opt.gpu, non_blocking=True)

            # 前向传播
            feat_list, logit_list = model(images, is_feat=True)
            output = logit_list

            # 确保output是 [batch_size, num_classes] 维度
            if len(output.shape) == 1:
                # 增加batch维度：[100] -> [1, 100]
                output = output.unsqueeze(0)
                print(f"警告：Output维度异常，已修复为 {output.shape}")
            # 确保labels是 [batch_size] 一维张量
            if len(labels.shape) > 1:
                labels = labels.squeeze()
                print(f"警告：Labels维度异常，已修复为 {labels.shape}")

            # 调试：打印维度信息
            if idx == 0:
                print(f"调试信息 - Batch {idx}:")
                print(f"  Images shape: {images.shape}")
                print(f"  Output shape: {output.shape}")
                print(f"  Labels shape: {labels.shape}")

            # 计算损失
            loss = criterion(output, labels)

            # 计算准确率
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # 更新时间
            batch_time.update(time.time() - end)
            end = time.time()

    # 分布式汇总
    if opt.multiprocessing_distributed:
        reduced_loss = reduce_tensor(torch.tensor(losses.avg).cuda(opt.gpu), 1)
        reduced_top1 = reduce_tensor(torch.tensor(top1.avg).cuda(opt.gpu), 1)
        reduced_top5 = reduce_tensor(torch.tensor(top5.avg).cuda(opt.gpu), 1)
        return reduced_top1.item(), reduced_top5.item(), reduced_loss.item()

    return top1.avg, top5.avg, losses.avg

if __name__ == '__main__':
    main()