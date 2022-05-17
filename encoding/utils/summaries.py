import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from encoding.datasets.utils import decode_seg_map_sequence, decode_segmap

import numpy as np
import cv2


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.mean = torch.Tensor([.485, .456, .406])
        self.std = torch.Tensor([.229, .224, .225])

    def create_summary(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return self.writer

    def write_loss(self, loss, name, step, type, group):
        '''
        :param step: the x coordinate of tb graph
        :param type: iter or epoch
        :param group: tb group (train or val)
        '''
        assert len(loss) <= len(name)
        for i in range(len(loss)):
            self.writer.add_scalar(group + '/loss_' + type + '_' + name[i], loss[i], step)


    def visualize_per_image_stages(self, writer, dataset, image, target, output, global_step, img_name, batch_index):
        '''

        :param writer:
        :param dataset:
        :param image: [batch, 3, height, width]
        :param target: [batch, height, width]
        :param output:
        :param global_step:
        :param img_name:
        :param batch_index:
        :return:
        '''
        if 'pcontext' not in dataset and 'skinny' not in dataset and 'dark' not in dataset:
            return

        grid_output = []
        if type(output) == 'list':
            for i in range(len(output)):
                pred = torch.max(output[i][batch_index:batch_index + 1], 1)[1]
                # print(pred.size())
                grid_output.append(pred)
        else:
            pred = torch.max(output[batch_index:batch_index + 1], 1)[1]
            # print(pred.size())
            grid_output.append(pred)
        grid_output.append(target[batch_index:batch_index + 1])
        # print(target[batch_index:batch_index + 1].size())

        grid_output = torch.cat(grid_output, dim=0).detach().cpu().numpy()
        grid_output = decode_seg_map_sequence(grid_output, dataset=dataset).float()

        if image is not None:
            input = image[batch_index:batch_index + 1].clone()
            input = (input * self.std.view(1,-1,1,1) + self.mean.view(1,-1,1,1)).cpu()
            grid_output = torch.cat([grid_output, input], dim=0)

        grid_image = make_grid(grid_output, 4, normalize=False, range=(0, 255))
        writer.add_image(img_name + '/Stages', grid_image, global_step)

    def visualize_image(self, writer, dataset, image, target, output, global_step, img_name, dist=None, attention=None, stage_num=0):
        if dataset != 'pcontext':
            return
        # image = image.float()
        # target = target.float()
        # output = output.float()
        # dist = dist.float()
        # attention = attention.float()
        if stage_num == 0:
            if global_step == 0:
                grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image(img_name+'/Image', grid_image, global_step)
            grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image(img_name+'/Groundtruth_label', grid_image, global_step)

            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image(img_name+'/Predicted_label', grid_image, global_step)

            if dist is not None or attention is not None:
                inputs = make_grid(image[:3].clone().cpu().data, 3, normalize=False, range=(0,255))
                inputs = inputs.permute((1,2,0))
                inputs = inputs * self.std + self.mean
                inputs = inputs.numpy().astype(np.uint8)

                target = target.float()
                if dist is not None:
                    pass
                if attention is not None:
                    self.vis_attention_map(attention, writer, global_step, img_name, scale=100, blend=False, inputs=inputs)


        else:
            if global_step == 0:
                grid_image = make_grid(image[:1].clone().cpu().data, 1, normalize=True)
                writer.add_image(img_name+'/Image', grid_image, global_step)
                grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:1], 1).detach().cpu().numpy(),
                                                               dataset=dataset), 1, normalize=False, range=(0, 255))
                writer.add_image(img_name+'/Groundtruth_label', grid_image, global_step)

            preds = []
            for i in range(stage_num):
                preds.append(output[i][:1])
            pred = torch.cat(preds, dim=0)
            grid_image = make_grid(decode_seg_map_sequence(torch.max(pred, 1)[1].detach().cpu().numpy(),
                                                           dataset=dataset), 3, normalize=False, range=(0, 255))
            writer.add_image(img_name+'/Predicted_label', grid_image, global_step)


    def vis_attention_map(self, attention, writer, global_step, img_name, scale=1.0, blend=False, inputs=None):
        if isinstance(attention, list):
            attentions = []
            for a in attention:
                attentions.append(a[:3].clone().cpu())
            attention = torch.cat(attentions, dim=0)
        else:
            attention = attention[:3].clone().cpu()

        attention = make_grid(attention, 3, normalize=False, range=(0, 255) )[0]
        attention = attention.detach().numpy()
        attention = attention - attention.min()

        if blend:
            attention_map, attention_map_blend = self.create_heatmap_from_numpy(attention, scale=scale, blend=blend, inputs=inputs)
            writer.add_image(img_name+'/Attention_Map', attention_map , global_step)
            writer.add_image(img_name+'/Attention_Blend', attention_map_blend , global_step)
        else:
            attention_map = self.create_heatmap_from_numpy(attention, scale=scale, blend=blend, inputs=inputs)
            writer.add_image(img_name+'/Attention_Map', attention_map , global_step)

        # attention_map = self.create_heatmap_from_numpy(attention, 'mul5', blend=blend, inputs=inputs)
        # writer.add_image('Attention Map/5x', attention_map , global_step)


    def create_heatmap_from_numpy(self, image, scale=1.0, blend=False, inputs=None):
        # input: [H, W] ndarray
        # output: [C, H, W] tensor with JET color
        image_u1 = image * scale
        image_u1[ image_u1 > 255 ] = 255
        image_u1[ image_u1 < 0   ] = 0

        image_u1 = image_u1.astype(np.uint8)

        heatmap = cv2.applyColorMap(image_u1, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(heatmap.transpose((2, 0, 1)))

        if blend:
            heatmap_blend = cv2.cv2.resize(heatmap, (inputs.shape[1],inputs.shape[0]), interpolation=cv2.INTER_NEAREST)
            assert inputs is not None

            # for row in range(inputs.shape[0]):
            #     for list in range(inputs.shape[1]):
            #         for c in range(inputs.shape[2]):
            #             pv = inputs[row, list, c]
            #             inputs[row, list, c] = 255 - pv

            heatmap_blend = cv2.addWeighted(inputs, 0.2, heatmap_blend, 0.9, 0)

            tensor_blend = torch.from_numpy(heatmap_blend.transpose((2, 0, 1)))
            return tensor, tensor_blend


        return tensor

    def save_results(self, dataset, output, target, attention, img_name, p_size):
        self.p_size = p_size
        l = len(p_size)

        if attention is not None:
            assert len(attention) == 2*l+1

            # pools
            self.vis_pool(attention[0:l], 'avg', img_name)
            self.vis_pool(attention[l:2*l], 'max', img_name)

            # attention
            a = self.reshape(attention[-1])
            amin = a.min()
            amax = a.max()
            a = (a - amin) * 255 / (amax - amin)
            name = img_name + '_a'
            self.wirte_png(a, name)
            # self.vis_pool(attention[-2:-1], 'a', img_name)

        if dataset == 'pcontext':
            # pred
            pred = torch.max(output, 1)[1].detach().cpu().squeeze_().numpy()
            pred = decode_segmap(pred, dataset=dataset, div255=False).astype(np.uint8)
            cv2.imwrite(img_name+'_p.png', cv2.cvtColor(pred,cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            gt = target.squeeze_().cpu().numpy() + 1
            gt = decode_segmap(gt, dataset=dataset, div255=False, unlabeled=True).astype(np.uint8)
            cv2.imwrite(img_name+'_gt.png', cv2.cvtColor(gt,cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    def save_seg_vis(self, img_name, output, target, dataset):
        if 'pcontext' in dataset:
            dataset = 'pcontext'
        pred = torch.max(output, 1)[1].detach().cpu().squeeze_().numpy()
        pred = decode_segmap(pred, dataset=dataset, div255=False).astype(np.uint8)
        cv2.imwrite(img_name + '_p.png', cv2.cvtColor(pred, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

        gt = target.squeeze_().cpu().numpy() + 1
        gt = decode_segmap(gt, dataset=dataset, div255=False, unlabeled=True).astype(np.uint8)
        cv2.imwrite(img_name + '_gt.png', cv2.cvtColor(gt, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

    def vis_pool(self, pools, type, name):
        vmax, vmin = -100, 100
        for att in pools[1:]:
            vvmax = float(att.min())
            vvmin = float(att.max())
            if vvmax > vmax:
                vmax = vvmax
            if vmin > vvmin:
                vmin = vvmin

        # vmax, vmin = 1, -1
        scale = 255.0 / (vmax - vmin)

        for i in range(len(pools)):
            if i == 0 and self.p_size[0] == 1:
                continue
            a = self.reshape(pools[i].cpu())
            a = (a - vmin) * scale
            self.wirte_png(a, name+'_'+type+str(self.p_size[i]))

    def reshape(self, x):
        if len(x.shape) == 4:
            a = x[0].permute(1, 2, 0).data.numpy()
        elif len(x.shape) == 3:
            a = x.permute(1, 2, 0).data.numpy()
        else:
            raise NotImplementedError
        return a

    def wirte_png(self, img, name):
        img[img > 255] = 255
        img[img < 0] = 0
        img = img.astype(np.uint8)
        heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(name + '.png', heatmap, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


