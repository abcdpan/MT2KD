"""
the general training framework
"""

from __future__ import print_function

import os
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
from models.util import ConvReg, Classification

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader,  get_dataloader_sample
# from dataset.imagenet_dali import get_dali_data_loader

from MT2KD.loops_ours import train_distill as train, validate_vanilla, validate_distill
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate

from crd.criterion import CRDLoss
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, VIDLoss

split_symbol = '~' if os.name == 'nt' else ':'

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # basic
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')
    parser.add_argument('--model_s', type=str, default='resnet8x4_multiCls')
    parser.add_argument('--path_t', type=str, default='', help='teacher model snapshot')
    parser.add_argument('--path_cls1', type=str, default='', help='teacher cls1')
    parser.add_argument('--path_cls2', type=str, default='', help='teacher cls2')
    parser.add_argument('--path_cls3', type=str, default='', help='teacher cls3')
    parser.add_argument('--path_cls4', type=str, default='', help='teacher cls4')
    parser.add_argument('--path_cls5', type=str, default='', help='teacher cls5')
    parser.add_argument('--path_cls6', type=str, default='', help='teacher cls6')
    parser.add_argument('--path_cls7', type=str, default='', help='teacher cls7')


    # distillation
    parser.add_argument('--trial', type=str, default='01', help='trial id')
    parser.add_argument('--size', type=int, default=1, help='pool_size')
    parser.add_argument('--kd_T', type=float, default=2, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', 'vid',
                                                                      'crd', 'semckd','srrl', 'simkd','ldrld'])
    parser.add_argument('-c', '--cls', type=float, default=0.1, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=9.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('-f', '--factor', type=int, default=2, help='factor size of SimKD')
    parser.add_argument('-s', '--soft', type=float, default=1.0, help='attention scale of SemCKD')
    parser.add_argument('--k', type=int, default=7, help='k')


    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # multiprocessing
    parser.add_argument('--dali', type=str, choices=['cpu', 'gpu'], default=None)
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    opt = parser.parse_args()

    # set different learning rates for these MobileNet/ShuffleNet models
    if opt.model_s in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/ours/models'
    opt.tb_path = './save/ours/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_r', '{}_a', '{}_b', '{}_{}'])
    opt.model_name = model_name_template.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                opt.cls, opt.div, opt.beta, opt.trial)

    if opt.dali is not None:
        opt.model_name += '_dali:' + opt.dali

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model


def load_classifier(classifier_path_list, ClassificationCls, n_cls, gpu, opt, channels=256, size=4):
    print('==> loading classifiers')
    # 存储加载后的分类器
    classifier_list = []

    for idx, classifier_path in enumerate(classifier_path_list):
        try:
            # 实例化单个分类器
            classifier = ClassificationCls(channels=channels, num_classes=n_cls, size=size)

            # 加载权重
            checkpoint = torch.load(classifier_path, map_location=f'cuda:{gpu}')
            cls_state_dict = checkpoint.get('classifier', checkpoint)

            # 处理module.前缀
            new_state_dict = {}
            for k, v in cls_state_dict.items():
                new_state_dict[k.replace('module.', '')] = v

            # 加载权重
            classifier.load_state_dict(new_state_dict, strict=False)

            # 配置分类器
            classifier.cuda(gpu)
            classifier.eval()
            for param in classifier.parameters():
                param.requires_grad = False

            # 添加到分类器列表
            classifier_list.append(classifier)

            # 打印加载信息
            if not opt.multiprocessing_distributed or opt.rank % opt.ngpus_per_node == 0:
                print(f"✅ 分类器{idx + 1}加载完成：{classifier_path}")
                print(f"   配置 - channels: {channels}, num_classes: {n_cls}, size: {size}")

        except Exception as e:
            # 异常处理：单个分类器加载失败时提示并终止
            raise RuntimeError(f"❌ 分类器{idx + 1}加载失败（路径：{classifier_path}）：{str(e)}")

    # 校验加载数量是否匹配
    if len(classifier_list) != len(classifier_path_list):
        raise ValueError(f"分类器加载数量不匹配！预期{len(classifier_path_list)}个，实际{len(classifier_list)}个")

    return classifier_list



best_acc = 0
total_time = time.time()
def main():

    opt = parse_option()

    # ASSIGN CUDA_ID
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
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(12345)
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(12345)

    # model
    n_cls = {
        'cifar100': 100,
        'imagenet': 1000,
    }.get(opt.dataset, None)

    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    try:
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)


    # 加载分类器
    classifier_path1 = opt.path_cls1
    classifier_path2 = opt.path_cls2
    classifier_path3 = opt.path_cls3
    classifier_path4 = opt.path_cls4
    classifier_path5 = opt.path_cls5
    classifier_path6 = opt.path_cls6
    classifier_path7 = opt.path_cls7
    classifier_path_list = [classifier_path1,classifier_path2
                            ]
    channels = feat_t[-2].shape[1]

    classifier_list = load_classifier(
        classifier_path_list=classifier_path_list,
        ClassificationCls=Classification,  # 传入类本身，而非实例
        n_cls=n_cls,                      # 补充n_cls参数
        gpu=opt.gpu,                      # 传入正确的GPU ID（int）
        opt=opt,                          # 传入opt对象
        channels=channels,                # 传入自动获取的通道数
        size=opt.size                     # 传入池化尺寸
    )


    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)


    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    # elif opt.distill == 'ldrld':
    #     criterion_kd = LDRLD(opt.div, opt.beta, k=opt.k, temperature=opt.kd_T, ce_loss_weight=opt.cls)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        trainable_list.append(criterion_kd)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        if opt.dataset == 'cifar100':
            opt.n_data = 50000
        else:
            opt.n_data = 1281167
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    # elif opt.distill == 'semckd':
    #     s_n = [f.shape[1] for f in feat_s[1:-1]]
    #     t_n = [f.shape[1:-1]]
    #     criterion_kd = SemCKDLoss()
    #     self_attention = SelfA(opt.batch_size, s_n, t_n, opt.soft)
    #     module_list.append(self_attention)
    #     trainable_list.append(self_attention)
    # elif opt.distill == 'srrl':
    #     s_n = feat_s[-1].shape[1]
    #     t_n = feat_t[-1].shape[1]
    #     model_fmsr = SRRL(s_n= s_n, t_n=t_n)
    #     criterion_kd = nn.MSELoss()
    #     module_list.append(model_fmsr)
    #     trainable_list.append(model_fmsr)
    # elif opt.distill == 'simkd':
    #     s_n = feat_s[-2].shape[1]
    #     t_n = feat_t[-2].shape[1]
    #     model_simkd = SimKD(s_n= s_n, t_n=t_n, factor=opt.factor)
    #     criterion_kd = nn.MSELoss()
    #     module_list.append(model_simkd)
    #     trainable_list.append(model_simkd)
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)

    # 分类器添加到module_list（教师模型后）
    module_list.append(model_t)
    module_list.extend(classifier_list)

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay,
                          nesterov=True)

    if torch.cuda.is_available():
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data, _, train_sampler = get_dataloader_sample(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                                num_workers=opt.num_workers,
                                                                                is_sample=True,
                                                                                k=opt.nce_k,
                                                                                multiprocessing_distributed=opt.multiprocessing_distributed)
            else:
                train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        multiprocessing_distributed=opt.multiprocessing_distributed)
        else:
            pass
    else:
        raise NotImplementedError(opt.dataset)

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if not opt.skip_validation:
        teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt)
        if opt.dali is not None:
            val_loader.reset()
        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')

    # routine
    for epoch in range(1, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            if opt.dali is None:
                train_sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        # 训练
        train_acc1_1, train_acc1_2, train_acc_avg, \
        train_acc5_1, train_acc5_2, train_acc5_avg, \
        train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            # 分布式同步训练指标
            metrics = torch.tensor([
                train_acc1_1, train_acc1_2, train_acc_avg,
                train_acc5_1, train_acc5_2, train_acc5_avg,
                train_loss
            ]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            (
                train_acc1_1, train_acc1_2, train_acc_avg,
                train_acc5_1, train_acc5_2, train_acc5_avg,
                train_loss
            ) = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            # 打印训练指标
            print(' * Epoch {}, GPU {}\n'
                  '   Train Branch1 Acc@1: {:.3f} | Train Branch2 Acc@1: {:.3f}\n'
                  '   Train Avg Acc@1: {:.3f} | Train Avg Acc@5: {:.3f} | Time: {:.2f}s'.format(
                epoch, opt.gpu,
                train_acc1_1, train_acc1_2,
                train_acc_avg, train_acc5_avg, time2 - time1))
            # 训练日志
            logger.log_value('train_acc_branch1', train_acc1_1, epoch)
            logger.log_value('train_acc_branch2', train_acc1_2, epoch)
            logger.log_value('train_acc_avg', train_acc_avg, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        print('GPU %d validating' % (opt.gpu))
        # 验证
        test_acc1_1, test_acc1_2, test_acc_avg, \
        test_acc5_1, test_acc5_2, test_acc5_avg, \
        test_loss = validate_distill(val_loader, module_list, criterion_cls, opt)

        # 分布式同步验证指标
        if opt.multiprocessing_distributed:
            metrics = torch.tensor([
                test_acc1_1, test_acc1_2, test_acc_avg,
                test_acc5_1, test_acc5_2, test_acc5_avg,
                test_loss
            ]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            (
                test_acc1_1, test_acc1_2, test_acc_avg,
                test_acc5_1, test_acc5_2, test_acc5_avg,
                test_loss
            ) = reduced.tolist()

        if opt.dali is not None:
            train_loader.reset()
            val_loader.reset()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            # 打印验证指标
            print(' ** Test Epoch {}, GPU {}\n'
                  '   Test Branch1 Acc@1: {:.3f} | Test Branch2 Acc@1: {:.3f}\n'
                  '   Test Avg Acc@1: {:.3f} | Test Avg Acc@5: {:.3f}'.format(
                epoch, opt.gpu,
                test_acc1_1, test_acc1_2,
                test_acc_avg, test_acc5_avg))

            # 验证日志
            logger.log_value('test_acc_branch1', test_acc1_1, epoch)
            logger.log_value('test_acc_branch2', test_acc1_2, epoch)
            logger.log_value('test_acc_avg', test_acc_avg, epoch)
            logger.log_value('test_loss', test_loss, epoch)
            logger.log_value('test_acc_top5_avg', test_acc5_avg, epoch)

            # 最优模型保存
            if test_acc_avg > best_acc:
                best_acc = test_acc_avg
                state = {
                    'epoch': epoch,
                    'model': model_s.state_dict(),
                    'best_acc': best_acc,
                }
                if opt.distill == 'simkd':
                    state['proj'] = trainable_list[-1].state_dict()
                save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))

                test_merics = {
                    'test_loss': test_loss,
                    'test_acc_avg': test_acc_avg,
                    'test_acc_branch1': test_acc1_1,
                    'test_acc_branch2': test_acc1_2,
                    'test_acc_top5_avg': test_acc5_avg,
                    'epoch': epoch
                }

                save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                print('saving the best model!')
                torch.save(state, save_file)

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        print('best accuracy:', best_acc)
        save_state = {k: v for k, v in opt._get_kwargs()}
        num_params = (sum(p.numel() for p in model_s.parameters())/1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] =  (time.time() - total_time)/3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json")
        save_dict_to_json(save_state, params_json_path)

if __name__ == '__main__':
    main()