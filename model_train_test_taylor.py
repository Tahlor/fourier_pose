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
from data_loaders import Ego2Hands_taylor
from models.CPM import cpm_model_basic
from utils import *
import params

EPS = 1e-6
TESTING = False

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
        model = cpm_model_basic.CPM(k=config.num_keypoints, baseline=config.experiment=="_BASELINE")
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
    #model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    return model.cuda()

def train_model_2d(args, config, model_2d, seq_i = -1):
    print("Training for 2d model")
    cudnn.benchmark = True
    
    # Data loader
    train_loader = None
    if config.dataset == 'ego2hands':
        hand_dataset = Ego2Hands_taylor.Ego2HandsData(args, config, seq_i)
        train_loader = torch.utils.data.DataLoader(hand_dataset,
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
        hand_dataset.set_B_gauss(model_2d.B_gauss)
    else:
        raise Exception("Error, unknown dataset: {}".format(config.dataset))
    
    base_lr = config.base_lr_2d
    policy_parameter = config.policy_parameter_2d
    if config.model_2d_name == params.MODEL_NAME_CPM and False:
        parameters_2d, multiple_2d = get_cpm_parameters(model_2d, config, is_default=False)
    else:
        parameters_2d = model_2d.parameters()
    optimizer_2d = torch.optim.Adam(parameters_2d, base_lr)

    if "step_size" in policy_parameter:
        lr_scheduler_2d = torch.optim.lr_scheduler.StepLR(optimizer_2d, **policy_parameter, )
    elif "patience"  in policy_parameter:
        policy_parameter["factor"] = policy_parameter["gamma"]
        del policy_parameter["gamma"]
        lr_scheduler_2d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2d, **policy_parameter)
    model_2d.train()
    
    out_2d_path = "outputs/{}/train_2d".format(config.dataset+config.experiment)
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
    losses = []
    # Training starts
    while iters < max_iter:
        for i, (item) in enumerate(train_loader):
            if not item["valid"].all().item():
                print("BAD INDEX", item["index"][torch.where(item["valid"]==False)])
                continue
            bs=item["image"].shape[0]
            iters += 1
            if iters * bs > max_iter:
                break
            img = item["image"].cuda(); proj = item["proj"]; coords = item["coords"]
            preds = model_2d(img)
            if config.experiment!="_BASELINE":
                if False:
                    preds2 = preds.cpu().reshape(bs, 21, -1, 2).clamp(min=-1 + EPS, max=1 - EPS)
                    preds_sin = torch.arcsin(preds2[:,:,:,0]) / (2*np.pi)
                    preds_cos = torch.arccos(preds2[:,:,:,1]) / (2*np.pi)
                    p = torch.stack([preds_sin, preds_cos], dim=3)
                    _proj = proj.unsqueeze(-1).tile(2)
                else:
                    p = preds.cpu().reshape(bs, 21, -1, 2)
                    _proj = item["sin_cos"]
                if torch.isnan(p).any():
                    input("PROBLEM")
                loss = criterion_mse(p, _proj)
            elif config.experiment=="_BASELINE":
                p = preds.cpu().reshape(bs, 21, 2)
                _proj = coords
                loss = criterion_mse(p, _proj)
            if torch.isnan(loss):
                input("UGH")

            # solutions = regressor.fit(np.tile(B_gauss,21), p.detach().numpy()[0])
            # p.shape = torch.Size([20, 21, 32, 2])
            # B_gauss.shape
            # (32, 2)
            optimizer_2d.zero_grad()
            loss.backward()
            optimizer_2d.step()
            lr_scheduler_2d.step(loss.item()/bs)
            losses.append(loss.item()/bs)
            # b=0
            # for joint in range(1,21):
            #     best_guess = cpm_model_basic.calc_best_guess(p[b,joint].detach().numpy(), model_2d.B_gauss)
            #     actual = cords[b, joint]
            #     criterion_mse(best_guess, actual)
            if TESTING:
                print(loss)
                break

            # Display info
            if iters % config.display_interval - 1 == 0:
                print("Train Iteration: {}".format(iters))
                #print("Learning rate: {}".format(lr_scheduler_2d.get_last_lr()))
                print("LOSS:", np.average(losses))
                losses = []
            if iters in [1,10,50,100,200]:
                print(iters)
                plot_tensor(p)
                plot_tensor(_proj)

            # Save models
            if iters % config.save_interval == 0:
                print("Saving latest model at {}".format(model_2d_save_path))
                save_model({
                     'iter': iters,
                     'state_dict': model_2d.state_dict(),
                     'B_gauss': model_2d.B_gauss
                }, is_best = False, is_last = False, filename = model_2d_save_path)
    # Save models
    print("Saving finished model at {}".format(model_2d_save_path))
    save_model({
         'iter': iters,
         'state_dict': model_2d.state_dict(),
    }, is_best = False, is_last = True, filename = model_2d_save_path)
    
def plot_tensor(f):
    from matplotlib import pyplot as plt
    plt.hist(f.detach().numpy().reshape(-1))
    plt.show()

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

