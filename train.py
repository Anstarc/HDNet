import numpy as np
from tqdm import tqdm
import math
import torchvision.transforms as transform
from torch.cuda.amp import autocast, GradScaler
from torch.nn import SyncBatchNorm

from encoding import utils
from encoding.utils import DataLoaderX
from encoding.nn import OIMLoss
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model
from encoding.utils import TensorboardSummary

from option import Options
from utils import *


class Trainer():
    def __init__(self, args):
        self.args = args
        self.cuda_device = get_cuda_device()
        self.cuda_device_num = len(self.cuda_device)
        print(self.cuda_device)
        self.cuda_device = range(self.cuda_device_num)

        args.log_name = str(args.checkname)
        self.logger = utils.create_logger(args.result_run_dir, args.log_name)

        self.summary = TensorboardSummary(args.result_run_dir)
        self.writer = self.summary.create_summary()

        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size, 'logger': self.logger,
                       'scale': args.scale}
        if args.final_exp:
            test_mode = 'testval'
            test_bs = 1
        else:
            test_mode = 'val'
            test_bs = args.batch_size

        trainset = get_segmentation_dataset(args.dataset, split='train', mode='train',
                                            root=args.data_folder, **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode=test_mode,
                                           root=args.data_folder, **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = DataLoaderX(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = DataLoaderX(testset, batch_size=test_bs,
                                         drop_last=False, shuffle=False, **kwargs)
        self.val_names = []

        if self.args.dataset == 'pcontext_detail':
            for name in self.valloader.dataset.ids:
                self.val_names.append(name['file_name'].split('/')[-1].split('.')[0])

        else:
            for name in self.valloader.dataset.images:
                self.val_names.append(name.split('/')[-1].split('.')[0])


        self.nclass = trainset.num_class
        # model
        if self.cuda_device_num == 1:
            bn = torch.nn.BatchNorm2d
        else:
            bn = SyncBatchNorm
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone,
                                       aux=args.aux, se_loss=args.se_loss,
                                       norm_layer=bn,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid,
                                       stride=args.stride,
                                       multi_dilation=args.multi_dilation,
                                       root=args.pretrained_home+'/pretrain_models',
                                       skin=args.skin, ft=args.ft)

        # optimizer using different LR
        params_list = []
        if hasattr(model, 'pretrained'):
            if model.pretrained is not None:
                params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]

        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*args.lr_times})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*args.lr_times})
        if hasattr(model, 'get_parameters'):
            pl = model.get_parameters()
            for p in pl:
                params_list.append({'params': p, 'lr': args.lr * args.lr_times})

        if hasattr(model, 'get_ft_params'):
            for p in model.get_ft_params():
                params_list.append({'params': p, 'lr': args.lr*args.lr_times})

        if args.model == 'skinny':
            params_list = [{'params': model.parameters(), 'lr': args.lr},]
        if hasattr(model, 'decoder'):
            params_list.append({'params': model.decoder.parameters(), 'lr': args.lr*args.lr_times})

        if args.ft and args.skin:
            params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr * 0.5},
                           {'params': model.head.parameters(), 'lr': args.lr * args.lr_times * 2}]
            if args.aux:
                for d in model.decoder[:-2]:
                    params_list.append({'params': d.parameters(), 'lr': args.lr * args.lr_times*2})
                params_list.append({'params': model.decoder[-1].parameters(), 'lr': args.lr*args.lr_times})
                params_list.append({'params': model.decoder[-2].parameters(), 'lr': args.lr*args.lr_times})

        optimizer = torch.optim.SGD(params_list,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay)

        if args.use_oim:
            self.oim = OIMLoss(512, self.nclass, para={'gamma':args.gamma, 'total_epoch':args.epochs}, bg=args.bg, start_epoch=args.oim_start, ingore=-1)
            self.lut = torch.zeros(self.nclass, 512, requires_grad=False).cuda()
            torch.nn.init.kaiming_uniform_(self.lut, a=math.sqrt(5))
        else:
            self.oim = None
            args.oim_start = 1000

        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.scaler = GradScaler()
            if self.oim is not None:
                self.oim = self.oim.cuda()

        # finetune from a trained model
        if args.ft:
            args.start_epoch = 0
            checkpoint = torch.load(args.ft_resume)
            if args.skin:
                keys = []
                for k in checkpoint['state_dict'].keys():
                    keys.append(k)
                for k in keys:
                    if 'decoder.5' in k or 'decoder.4' in k or 'head' in k:
                        del checkpoint['state_dict'][k]

            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.ft_resume, checkpoint['epoch']))
        # resuming checkpoint
        if args.resume:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader), logger=self.logger,
                                            lr_step=args.lr_step, warmup_epochs=args.warmup)
        self.best_pred = 0.0

    def training(self, epoch):
        train_loss, ce_loss = init_loss(2)
        self.model.train()
        tbar = tqdm(self.trainloader)
        num_img_tr = len(self.trainloader)

        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()

            with autocast():
                outputs, dist, attention, losses = self.model(image.cuda(), target.cuda(), epoch)
                losses = gather_loss(losses)
                if self.oim is not None:
                    losses.append(self.oim(self.lut, dist, target.cuda(), outputs[0], epoch))
                loss = weighted_sum(losses, [1.0, 0.1])

            if self.args.model == 'emanet':
                with torch.no_grad():
                    mu = dist.mean(dim=0, keepdim=True)
                    momentum = 0.9
                    self.model.module.emau.mu *= momentum
                    self.model.module.emau.mu += mu * (1 - momentum)

            if not self.args.not_train_seg:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += loss.item()
                ce_loss += losses[0].item()

            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.summary.write_loss([loss.item(), losses[0].item()], ['total','ce'], i + num_img_tr * epoch, 'iter', 'train')

        train_loss, ce_loss = avg_loss([train_loss, ce_loss], len(tbar))
        self.logger.info('Train loss: %.3f' % (train_loss))
        self.summary.write_loss([train_loss, ce_loss], ['total','ce'], epoch, 'epoch', 'train')

        # save checkpoint every 10 epoch
        filename = "checkpoint.pth.tar"
        is_best = False
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
            }, self.args, is_best, filename)


    def validation(self, epoch):

        # Fast test during the training
        def eval_batch(image, target, epoch):
            with autocast():
                outputs, dist, attention, losses = self.model(image.cuda(), target.cuda(), epoch)
                losses = gather_loss(losses)
                if self.oim is not None:
                    losses.append(self.oim(self.lut, dist, target.cuda(), outputs, epoch))

            pred = outputs.data
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred, target)
            inter, union = utils.batch_intersection_union(pred, target, self.nclass)
            return correct, labeled, inter, union, pred, target, losses, dist, attention, outputs

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label, total_loss, ce_loss = init_loss(6)
        if self.nclass == 2:
            total_tp, total_pos, total_real = init_loss(3)
        tbar = tqdm(self.valloader, desc='\r')

        for i, (image, target) in enumerate(tbar):
            with torch.no_grad():
                correct, labeled, inter, union, pred, target, losses, dist, attention, outputs = eval_batch(self.model, image, target, epoch)
                loss = weighted_sum(losses, [1.0, 0.1], gradients=False)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            total_loss += loss
            ce_loss += losses[0]
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()


            if self.nclass == 2:
                mIoU = IoU[-1]
                tp, pos, real = utils.batch_f1_score(pred, target, self.nclass)
                total_tp += tp
                total_pos += pos
                total_real += real
                prec = 1.0 * total_tp / (total_pos + np.spacing(1))
                recall = 1.0 * total_tp / (total_real + np.spacing(1))

                f1 = 2 * prec * recall / (prec + recall + np.spacing(1))

                tbar.set_description(
                    'CE Loss: %.3f, mIoU: %.3f, f1: %.3f' % (losses[0], mIoU, f1))
            else:
                tbar.set_description(
                    'CE Loss: %.3f, mIoU: %.3f' % (losses[0], mIoU))

            if self.cuda_device_num == 1 and i < 40:
                for j in range(image.data.shape[0]):
                    if epoch == self.args.epochs - 1:
                        input_image = image[j:j + 1]
                    else:
                        input_image = None
                    self.summary.visualize_per_image_stages(self.writer, self.args.dataset, input_image, target[j:j+1], outputs, epoch,
                                                 self.val_names[i * image.data.shape[0]+j], j)

        print(f'IoU: {IoU}')
        total_loss, ce_loss = avg_loss([total_loss, ce_loss], len(tbar))
        self.logger.info('CE Loss: %.3f, mIoU: %.4f' % (ce_loss, mIoU))
        self.summary.write_loss([total_loss,ce_loss], ['total','ce'], epoch, 'epoch', 'val')
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        if self.nclass == 2:
            self.writer.add_scalar('val/F1', f1, epoch)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            state_dict = self.model.state_dict()
            for i in list(state_dict.keys()):
                if 'post_process_layer' in i:
                    state_dict.pop(i)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


if __name__ == "__main__":
    args = Options().parse()

    if args.cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    trainer.logger.info(['Starting Epoch:', str(args.start_epoch)])
    trainer.logger.info(['Total Epoches:', str(args.epochs)])

    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.validation(epoch)