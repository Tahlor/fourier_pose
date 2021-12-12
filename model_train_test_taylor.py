import sys
import os
import numpy as np
import cv2
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from collections import OrderedDict
from data_loaders import Ego2Hands
from models.CPM import cpm_model
from utils import *
import params


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, dest='config')
    parser.add_argument('--use_seg', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--adapt', action='store_true', default=False)
    parser.add_argument('--save_outputs', action='store_true', default=False)
    return parser.parse_args()
    
def construct_model_2d(args, config):
    if config.model_2d_name == params.MODEL_NAME_CPM:
        model = cpm_model.CPM(k=config.num_keypoints)
    else:
        raise Exception("Error, model {} not implemented".format(config.model_2d_name))

    pretrained_path = get_model_2d_save_path(args, config)
    print("model_2d #params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load pretrained model
    if args.eval:
        print("Loading {}".format(pretrained_path))
        state_dict = torch.load(pretrained_path)['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    return model

def train_model_2d(args, config, model_2d, seq_i = -1):
    print("Training for 2d model")
    cudnn.benchmark = True
    
    # Data loader
    train_loader = None
    if config.dataset == 'ego2hands':
        hand_dataset = Ego2Hands.Ego2HandsData(args, config, seq_i)
        train_loader = torch.utils.data.DataLoader(hand_dataset,
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    else:
        raise Exception("Error, unknown dataset: {}".format(config.dataset))
    
    base_lr = config.base_lr_2d
    policy_parameter = config.policy_parameter_2d
    if config.model_2d_name == params.MODEL_NAME_CPM:
        parameters_2d, multiple_2d = get_cpm_parameters(model_2d, config, is_default=False)
    else:
        parameters_2d = model_2d.parameters()
    optimizer_2d = torch.optim.Adam(parameters_2d, base_lr)
    lr_scheduler_2d = torch.optim.lr_scheduler.StepLR(optimizer_2d, step_size = policy_parameter.step_size, gamma = policy_parameter.gamma)
    model_2d.train()
    
    out_2d_path = "outputs/{}/train_2d".format(config.dataset)
    os.makedirs(out_2d_path, exist_ok = True)
    model_2d_dir_path, model_2d_save_path = get_model_2d_save_path(args, config)
    os.makedirs(model_2d_dir_path, exist_ok = True)

    # Criterions
    criterion_mse = nn.MSELoss().cuda()

    # Measures
    losses_2d_meter_list = [AverageMeter() for i in range(config.num_2d_stages)]
    iters = 0
    heat_weight = 32*32*21
    max_iter = config.max_iter_2d
    
    # Training starts
    while iters < max_iter:
        for i, (img_input_tensor, heatmpas_gt_tensor) in enumerate(train_loader):
            iters += 1
            if iters > max_iter:
                break
            img_batch_size = img_input_tensor.size(0)
            
            # Segmentation module       
            img_input_var = torch.autograd.Variable(img_input_tensor.cuda())
            heatmaps_gt_var = torch.autograd.Variable(heatmpas_gt_tensor.cuda())
             
            heatmaps_stage1, heatmaps_stage2, heatmaps_stage3, heatmaps_stage4, heatmaps_stage5, heatmaps_stage_final = model_2d(img_input_var)
            
            loss_stage1 = criterion_mse(heatmaps_stage1, heatmaps_gt_var) * heat_weight
            loss_stage2 = criterion_mse(heatmaps_stage2, heatmaps_gt_var) * heat_weight
            loss_stage3 = criterion_mse(heatmaps_stage3, heatmaps_gt_var) * heat_weight
            loss_stage4 = criterion_mse(heatmaps_stage4, heatmaps_gt_var) * heat_weight
            loss_stage5 = criterion_mse(heatmaps_stage5, heatmaps_gt_var) * heat_weight
            loss_stage6 = criterion_mse(heatmaps_stage_final, heatmaps_gt_var) * heat_weight
            
            losses_2d_meter_list[0].update(float(loss_stage1), img_batch_size)
            losses_2d_meter_list[1].update(float(loss_stage2), img_batch_size)
            losses_2d_meter_list[2].update(float(loss_stage3), img_batch_size)
            losses_2d_meter_list[3].update(float(loss_stage4), img_batch_size)
            losses_2d_meter_list[4].update(float(loss_stage5), img_batch_size)
            losses_2d_meter_list[5].update(float(loss_stage6), img_batch_size)

            loss_2d_total = loss_stage1 + loss_stage2 + loss_stage3 + loss_stage4 + loss_stage5 + loss_stage6

            optimizer_2d.zero_grad()
            loss_2d_total.backward()
            optimizer_2d.step()
            
            lr_scheduler_2d.step()
                
            # Display info
            if iters % config.display_interval == 0:
                print("Train Iteration: {}".format(iters))
                print("Learning rate: {}".format(lr_scheduler_2d.get_last_lr()))
                for layer_i, loss_2d_meter in enumerate(losses_2d_meter_list):
                    print('Loss_2d_stage{} = {loss.avg: .4f}'.format(layer_i, loss=loss_2d_meter))
                
                # Visualize Outputs
                if args.save_outputs:
                    img_input_np = img_input_var.cpu().data.numpy().transpose(0,2,3,1)
                    heatmaps_output_np = heatmaps_stage_final.cpu().data.numpy().transpose(0,2,3,1)
                    heatmaps_gt_np = heatmpas_gt_tensor.cpu().data.numpy().transpose(0,2,3,1)
                    seq_status = "seq_{}".format(seq_i) if seq_i != -1 else ""
                    for batch_i, (img_input_i, heatmaps_output_i, heatmaps_gt_i) in enumerate(zip(img_input_np, heatmaps_output_np, heatmaps_gt_np)):
                        cv2.imwrite(os.path.join(out_2d_path, "{}_{}_{}img_gray.png".format(iters, batch_i, seq_status)), (img_input_i[:,:,0]*255.0 + 128.0).astype(np.uint8))#+128.0
                        cv2.imwrite(os.path.join(out_2d_path, "{}_{}_{}img_edge.png".format(iters, batch_i, seq_status)), (img_input_i[:,:,1]*255.0 + 128.0).astype(np.uint8))#+128.0
                        cv2.imwrite(os.path.join(out_2d_path, "{}_{}_{}img_seg.png".format(iters, batch_i, seq_status)), (img_input_i[:,:,2]*255.0 + 128.0).astype(np.uint8))#+128.0
                        heatmaps_gt_combined_i = np.max(heatmaps_gt_i[:, :, 1:], axis=2)
                        cv2.imwrite(os.path.join(out_2d_path, "{}_{}_{}heatmaps_gt.png".format(iters, batch_i, seq_status)), (heatmaps_gt_combined_i*255.0).astype(np.uint8))
                        heatmaps_output_combined_i = np.max(heatmaps_output_i[:, :, 1:], axis=2)
                        cv2.imwrite(os.path.join(out_2d_path, "{}_{}_{}heatmaps_output.png".format(iters, batch_i, seq_status)), (heatmaps_output_combined_i*255.0).astype(np.uint8))
                        kpts_i = get_kpts(np.expand_dims(heatmaps_gt_i.transpose(2, 0, 1), 0), img_h=img_input_i.shape[0], img_w=img_input_i.shape[1], num_keypoints=config.num_keypoints)
                        hand_vis_i = paint_kpts(None, (cv2.cvtColor((img_input_i[:,:,0]*256.0+128.0).astype(np.uint8), cv2.COLOR_GRAY2RGB)).astype(np.uint8), kpts_i)
                        cv2.imwrite(os.path.join(out_2d_path, "{}_{}_{}hand_vis.png".format(iters, batch_i, seq_status)), (hand_vis_i).astype(np.uint8))
                
                # Clear meters
                for loss_meter in losses_2d_meter_list:
                    loss_meter.reset()
            
            # Save models
            if iters % config.save_interval == 0:
                print("Saving latest model at {}".format(model_2d_save_path))
                save_model({
                     'iter': iters,
                     'state_dict': model_2d.state_dict(),
                }, is_best = False, is_last = False, filename = model_2d_save_path)
    # Save models
    print("Saving finished model at {}".format(model_2d_save_path))
    save_model({
         'iter': iters,
         'state_dict': model_2d.state_dict(),
    }, is_best = False, is_last = True, filename = model_2d_save_path)
    

if __name__ == '__main__':
    args = parse()
    config = Config(args.config)
    model_2d = construct_model_2d(args, config)
    if not args.eval:
        if not args.adapt:
            train_model_2d(args, config, model_2d)
        else:
            for seq_i in range(1, config.num_seqs + 1):
                train_model_2d(args, config, model_2d, seq_i)
    #else:
    #    test_model_2d(args, config, model_2d)