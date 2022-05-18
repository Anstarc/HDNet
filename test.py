import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options
from utils import *

ids = ['2008_000050', '2008_000052', '2008_000054', '2008_000090', '2008_000254', '2008_000257', '2008_000424', '2008_000533',
       '2008_000659', '2008_000804', '2008_000848', '2008_001041', '2008_001062', '2008_001380', '2008_001531', '2008_001544',
       '2008_002207', '2008_002283', '2008_003336', '2008_004907', '2008_005764', '2008_006646', '2008_007247', '2008_008437',
       '2009_000919', '2009_001370', '2009_001413', '2009_001673', '2009_002078', '2008_008011', '2009_000335', '2008_006151',
       '2008_007190', '2008_005257', '2008_005808', '2008_004910', '2008_005139', '2008_004592', '2008_004852', '2008_003826',
       '2008_004221', '2008_002589', '2008_002965', '2008_001669', '2008_002234', '2008_000942', '2008_001445', '2008_000418',
       '2008_000783', '2008_000149', '2008_000195', '2008_000064', '2010_000524', '2008_004910', '2009_003973', '2009_003247']

four2two = False

def test(args):
    # output folder
    outdir = '%s/test/'%(args.out_dir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
                                           transform=input_transform, root=args.data_folder)
    else:#set split='test' for test set
        testset = get_segmentation_dataset(args.dataset, split='val', mode='vis',
                                           transform=input_transform, root=args.data_folder)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    names = []
    if args.dataset == 'pcontext_detail':
        for name in test_data.dataset.ids:
            names.append(name['file_name'].split('/')[-1].split('.')[0])
    elif args.dataset == 'skinny' or args.dataset == 'dark':
        for name in test_data.dataset.images:
            names.append(name.split('/')[-1].split('.')[0])
    elif args.dataset == 'coco' or args.dataset == 'cityscapes' or args.dataset == 'ade20k':
        names = test_data.dataset.images

    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux=args.aux,
                                       se_loss=args.se_loss, norm_layer=torch.nn.BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation,
                                       root=args.pretrained_home+'/pretrain_models',
                                       skin=args.skin, ft=args.ft,
                                       eval=True)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        print(args.resume)
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    # print(model)
    num_class = testset.num_class
    # for 4 class dark to 2 class skinny
    if four2two:
        evaluator = MultiEvalModule(model, 4, multi_scales=args.multi_scales).cuda()
    else:
        evaluator = MultiEvalModule(model, testset.num_class, multi_scales=args.multi_scales).cuda()
    evaluator.eval()

    tbar = tqdm(test_data)
    def eval_batch(image, dst, evaluator, eval_mode, name):
        if eval_mode:
            # evaluation mode on validation set
            targets = dst
            outputs = evaluator.parallel_forward(image)

            # for 4 class dark to 2 class skinny
            if four2two:
                num_class = 2
            else:
                num_class = testset.num_class

            batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
            if num_class == 2:
                total_tp, total_pos, total_real = 0, 0, 0
            for output, target in zip(outputs, targets):
                correct, labeled = utils.batch_pix_accuracy(output.data.cpu(), target, four2two)
                inter, union = utils.batch_intersection_union(
                    output.data.cpu(), target, num_class, four2two)
                batch_correct += correct
                batch_label += labeled
                batch_inter += inter
                batch_union += union

                if num_class == 2:
                    tp, pos, real = utils.batch_f1_score(output, target, num_class)
                    total_tp += tp
                    total_pos += pos
                    total_real += real

            if num_class == 2:
                return batch_correct, batch_label, batch_inter, batch_union, total_tp, total_pos, total_real
            return batch_correct, batch_label, batch_inter, batch_union

    total_inter, total_union, total_correct, total_label = \
        np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    if num_class == 2:
        total_tp, total_pos, total_real = np.int64(0), np.int64(0), np.int64(0)
    for i, (image, dst) in enumerate(tbar):
        image, dst = image[0].unsqueeze_(0).cuda(), dst[0].unsqueeze_(0).cuda()
        if torch_ver == "0.3":
            image = Variable(image, volatile=True)
            correct, labeled, inter, union = eval_batch(image, dst, evaluator, args.eval, names[i])
        else:
            with torch.no_grad():
                if num_class == 2:
                    correct, labeled, inter, union, tp, pos, real = eval_batch(image, dst, evaluator, args.eval, names[i])
                else:
                    correct, labeled, inter, union = eval_batch(image, dst, evaluator, args.eval, names[i])
        pixAcc, mIoU, IoU = 0, 0, 0
        if args.eval:
            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()

            if num_class == 2:
                total_tp += tp
                total_pos += pos
                total_real += real
                prec = 1.0 * total_tp / (total_pos + np.spacing(1))
                recall = 1.0 * total_tp / (total_real + np.spacing(1))

                f1 = 2 * prec * recall / (prec + recall + np.spacing(1))

                tbar.set_description(
                    'F1: %.4f, IoU: %.4f' % (f1, IoU[-1]))
            else:
                tbar.set_description(
                    'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
    return pixAcc, mIoU, IoU, num_class

def eval_multi_models(args):
    if args.resume_dir is None or not os.path.isdir(args.resume_dir):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))
    for resume_file in os.listdir(args.resume_dir):
        if os.path.splitext(resume_file)[1] == '.tar':
            args.resume = os.path.join(args.resume_dir, resume_file)
            assert os.path.exists(args.resume)
            if not args.eval:
                test(args)
                continue
            pixAcc, mIoU, IoU, num_class = test(args)
        
            txtfile = args.resume
            txtfile = txtfile.replace('pth.tar', 'txt')
            if not args.multi_scales:
                txtfile = txtfile.replace('.txt', 'result_mIoU_%.4f.txt'%mIoU)
            else:
                txtfile = txtfile.replace('.txt', 'multi_scale_result_mIoU_%.4f.txt'%mIoU)
            fh = open(txtfile, 'w')
            print("================ Summary IOU ================\n")
            for i in range(0,num_class):
                print("%3d: %.4f\n" %(i,IoU[i]))
                fh.write("%3d: %.4f\n" %(i,IoU[i]))
            print("Mean IoU over %d classes: %.4f\n" % (num_class, mIoU))
            print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
            fh.write("Mean IoU over %d classes: %.4f\n" % (num_class, mIoU))
            fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
            fh.close()
    print('Evaluation is finished!!!')

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    eval_multi_models(args)
