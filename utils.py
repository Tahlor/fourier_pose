import os
import math
import numpy as np
import torch
import cv2
import random
from easydict import EasyDict as edict
import yaml
import params

def Config(filename):
    with open(filename, 'r') as f:
        parser = edict(yaml.load(f))
    for x in parser:
        print('{}: {}'.format(x, parser[x]))
    return parser

class AverageMeter(object):
    """ Computes ans stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_model_2d_save_path(args, config):
    model_dir_path = os.path.join(params.MODEL_SAVE_DIR_PATH, config.dataset, "2d")
    model_name = '{}_{}_2d_pretrained.pth.tar'.format(config.dataset, config.model_2d_name)
    model_save_path = os.path.join(model_dir_path, model_name)
    return model_dir_path, model_save_path
    
def get_cpm_parameters(model, config, is_default=True):
    if is_default:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': config.base_lr_2d},
            {'params': lr_2, 'lr': config.base_lr_2d * 2.},
            {'params': lr_4, 'lr': config.base_lr_2d * 4.},
            {'params': lr_8, 'lr': config.base_lr_2d * 8.}]
    return params, [1., 2., 4., 8.]
    
def seg_augmentation_wo_kpts(img, seg, energy):
    img_h, img_w = img.shape[:2]
    fg_mask = seg.copy()
    
    coords1 = np.where(fg_mask)
    img_top, img_bot = np.min(coords1[0]), np.max(coords1[0])
    
    shift_range_ratio = 0.2
    # down shift
    down_shift = True if not fg_mask[0, :].any() else False
    if down_shift:
        down_space = int((img_h - img_top)*shift_range_ratio)
        old_bot = img_h
        down_offset = random.randint(0, down_space)
        old_bot -= down_offset

        old_top = 0
        cut_height = old_bot - old_top

        new_bot = img_h
        new_top = new_bot - cut_height
    else:
        old_bot, old_top = img_h, 0
        new_bot, new_top = old_bot, old_top
    
    coords2 = np.where(fg_mask[old_top:old_bot,:])
    img_left, img_right = np.min(coords2[1]), np.max(coords2[1])
    
    # Left shift or right shift    
    left_shift = True if not fg_mask[old_top:old_bot, -1].any() else False
    right_shift = True if not fg_mask[old_top:old_bot, 0].any() else False
    if left_shift and right_shift:
        if random.random() > 0.5:
            right_shift = False
        else:
            left_shift = False
            
    if left_shift:
        left_space = int(img_right*shift_range_ratio)
        old_left = 0
        left_offset = random.randint(0, left_space)
        old_left += left_offset
        
        old_right = img_w
        cut_width = old_right - old_left
        
        new_left = 0
        new_right = new_left + cut_width
        
    if right_shift:
        right_space = int((img_w - img_left)*shift_range_ratio)
        old_right = img_w
        right_offset = random.randint(0, right_space)
        old_right -= right_offset
        
        old_left = 0
        cut_width = old_right - old_left
        
        new_right = img_w
        new_left = new_right - cut_width
    
    if not (left_shift or right_shift):
        old_left, old_right = 0, img_w
        new_left, new_right = old_left, old_right

    img_new = np.zeros_like(img)
    seg_new = np.zeros_like(seg)
    energy_new = np.zeros_like(energy)

    img_new[new_top:new_bot, new_left:new_right] = img[old_top:old_bot, old_left:old_right]
    seg_new[new_top:new_bot, new_left:new_right] = seg[old_top:old_bot, old_left:old_right]
    energy_new[new_top:new_bot, new_left:new_right] = energy[old_top:old_bot, old_left:old_right]
    return img_new, seg_new, energy_new
    
def seg_augmentation_w_kpts(img, seg, energy, kpts_2d_glob):
    img_h, img_w = img.shape[:2]
    fg_mask = seg > 0
    # Convert kpts 2d from ratio to coordinates
    kpts_2d_glob = kpts_2d_glob.copy()
    kpts_2d_glob *= np.array([img_h, img_w])
    
    # Find bounding box for joints in img global
    
    kpt_top, kpt_bot = int(np.amin(kpts_2d_glob[:,0])), int(np.amax(kpts_2d_glob[:,0]))
    kpt_left, kpt_right = int(np.amin(kpts_2d_glob[:,1])), int(np.amax(kpts_2d_glob[:,1]))

    # Randomly shift the image left, right and down
    margin = 20

    # Find vertical area to crop
    down_shift = True if not fg_mask[0, :].any() else False
    down_shift_val = 0
    if down_shift:
        down_space = img_h - kpt_bot
        old_bot = kpt_bot + margin if down_space > margin else img_h
        # Shift down
        down_offset = random.randint(0, img_h - old_bot)
        old_bot += down_offset
        down_shift_val = img_h - old_bot

        old_top = 0
        cut_height = old_bot - old_top
        new_bot = img_h
        new_top = new_bot - cut_height
    else:
        old_bot, old_top = img_h, 0
        new_bot, new_top = old_bot, old_top

    # Left shift or right shift    
    left_shift = True if not fg_mask[:, -1].any() else False
    right_shift = True if not fg_mask[:, 0].any() else False
    if left_shift and right_shift:
        if random.random() > 0.5:
            right_shift = False
        else:
            left_shift = False
           
    left_shift_val = 0
    if left_shift:
        left_space = kpt_left
        old_left = kpt_left - margin if left_space > margin else 0
        # Shift left
        left_offset = random.randint(0, old_left)
        old_left -= left_offset
        left_shift_val = old_left
        
        old_right = img_w
        cut_width = old_right - old_left

        new_left = 0
        new_right = new_left + cut_width

    right_shift_val = 0
    if right_shift:
        right_space = img_w - kpt_right
        old_right = kpt_right + margin if right_space > margin else img_w
        # Shift right
        right_offset = random.randint(0, img_w - old_right)
        old_right += right_offset
        right_shift_val = img_w - old_right
        
        old_left = 0
        cut_width = old_right - old_left
        new_right = img_w
        new_left = new_right - cut_width
    
    if not (left_shift or right_shift):
        side_shift_val = 0
        old_left, old_right = 0, img_w
        new_left, new_right = old_left, old_right
    else:
        side_shift_val = -left_shift_val if left_shift else right_shift_val

    img_new = np.zeros_like(img)
    seg_new = np.zeros_like(seg)
    energy_new = np.zeros_like(energy)

    #print("old {}, {} , {}, {}".format(old_top, old_bot, old_left, old_right))
    #print("new {}, {} , {}, {}".format(new_top, new_bot, new_left, new_right))
    #print("diff {}, {}".format(down_shift_val, side_shift_val))
    #print("{}, {}".format(img_h, img_w))
    offset_np = np.array([float(down_shift_val), float(side_shift_val)])
    kpts_2d_glob += offset_np
    kpts_2d_glob /= np.array([img_h, img_w])
    #print(offset_np)
    img_new[new_top:new_bot, new_left:new_right] = img[old_top:old_bot, old_left:old_right]
    seg_new[new_top:new_bot, new_left:new_right] = seg[old_top:old_bot, old_left:old_right]
    energy_new[new_top:new_bot, new_left:new_right] = energy[old_top:old_bot, old_left:old_right]
    return img_new, seg_new, energy_new, kpts_2d_glob#offset_np
    
def crop_hand_for_2d(img_top_hand, img_bot_hand, energy, seg, kpts_2d_glob, img_size=224, fix_scale = False, size_th = 5000, side_th = 10, side=1):
    img = img_top_hand.copy()
    img_h, img_w = img_top_hand.shape[0], img_top_hand.shape[1]
    
    if not fix_scale:
        scale_low, scale_high = 1.0, 1.7
    else:
        scale_low, scale_high = 1.55, 1.55
    
    energy_mask = energy > (np.amax(energy)/2)
    if kpts_2d_glob is None:
        if energy_mask.sum() < size_th:
            return None, None, None, None
        if energy_mask[:, 0].sum() > side_th or energy_mask[:, -1].sum() > side_th or energy_mask[0, :].sum() > side_th or energy_mask[-1, :].sum() > side_th:
            return None, None, None, None
    coords = np.where(energy_mask)

    row_min, row_max, col_min, col_max = np.min(coords[0]), np.max(coords[0]), np.min(coords[1]), np.max(coords[1])
    row_range = row_max - row_min
    col_range = col_max - col_min

    max_range = max(row_range, col_range)#min(max(row_range, col_range), min_range)
    #print("row_min: {}".format(row_min))
    #print("row_range: {}".format(row_range))
    #print("col_min: {}".format(col_min))
    #print("col_range: {}".format(col_range))
   
    #print("max_range: {}".format(max_range))
    mid_point = np.array([row_min + row_range/2.0, col_min + col_range/2.0])
    # Randomize croped window
    new_max_range = random.uniform(scale_low, scale_high) * max_range
    offset_point = mid_point - new_max_range/2.0
    # Make sure offset point is no less than (0,0)
    #!offset_point = np.maximum(offset_point, np.array([0.0, 0.0]))
    #!offset_point = np.minimum(offset_point, np.array([img_h - max_range, img_w - max_range]))

    #row_col_room = max((new_max_range - max_range)/4.0, 0)
    #row_offset = random.uniform(0.0, row_col_room)
    #col_offset = random.uniform(0.0, row_col_room)
    row_shift_room = max((new_max_range - row_range)/4.0, 0)
    row_offset = random.uniform(0.0, row_shift_room)
    col_shift_room = max((new_max_range - col_range)/4.0, 0)
    col_offset = random.uniform(0.0, col_shift_room)
    #row_offset = (new_max_range - max_range)/4.0#! if offset_point[0] > 0.0 else 0.0
    #col_offset = (new_max_range - max_range)/4.0#! if offset_point[1] > 0.0 else 0.0

    #!row_start = min(int(offset_point[0] + row_offset), img_h - 50)
    #!col_start = min(int(offset_point[1] + col_offset), img_w - 50)
    row_start = int(offset_point[0] + row_offset)
    col_start = int(offset_point[1] + col_offset)

    crop_range = max_range + (new_max_range - max_range)/2.0
    crop_range = min(crop_range, min(img_h, img_w) - 1)
    # Make sure crop range doesn't extend over image size
    #!crop_range = min(img.shape[0] - row_start, crop_range)
    #!crop_range = min(img.shape[1] - col_start, crop_range)

    row_end = int(row_start + crop_range)
    col_end = int(col_start + crop_range)

    # Adjust if out of image frame
    bounding_box = np.array([row_start, row_end, col_start, col_end], dtype=np.int32)
    if row_start < 0:
        # shift down
        bounding_box[:2] += (0 - row_start)
    if row_end > (img_h - 1):
        # shift up
        bounding_box[:2] -= (row_end - (img_h - 1))
    if col_start < 0:
        # shift right
        bounding_box[2:] += (0 - col_start)
    if col_end > (img_w - 1):
        # shift down
        bounding_box[2:] -= (col_end - (img_w - 1))
    row_start, row_end, col_start, col_end = bounding_box

    img_top_cropped = img_top_hand[row_start:row_end, col_start:col_end]
    img_bot_cropped = img_bot_hand[row_start:row_end, col_start:col_end] if img_bot_hand is not None else None
    seg_cropped = seg[row_start:row_end, col_start:col_end] if seg is not None else None
    #edge_cropped = edge[row_start:row_end, col_start:col_end] if edge is not None else None
    kpts_2d_can = None
    if kpts_2d_glob is not None:
        kpts_2d_glob *= np.array([img_h, img_w])
        kpts_2d_can = (kpts_2d_glob - np.array([row_start, col_start]))/crop_range

    #print(img_new.shape)
    
    #cv2.imwrite("outputs/test0.png", img_new)
    
    img_top_resized = cv2.resize(img_top_cropped, (img_size, img_size)).astype(np.float32)
    img_bot_resized = cv2.resize(img_bot_cropped, (img_size, img_size)).astype(np.float32) if img_bot_cropped is not None else None
    seg_resized = cv2.resize(seg_cropped, (img_size, img_size)).astype(np.float32) if seg_cropped is not None else None
    
    if side == 0:
        img_top_resized = cv2.flip(img_top_resized, 1)
        img_bot_resized = cv2.flip(img_bot_resized, 1) if img_bot_resized is not None else None
        seg_resized = cv2.flip(seg_resized, 1) if seg_resized is not None else None

    return img_top_resized, img_bot_resized, seg_resized, kpts_2d_can
    
def normalize_tensor(tensor, mean, std):
    for t in tensor:
        t.sub_(mean).div_(std)
    return tensor
    
def gaussian_kernel(size_w, size_h, center_x, center_y, sigma, z = -2.0):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    #spread = 1.0 - ((z + 2.0) / 4.0) # min z = -2.0, spread between [0,1]
    if z is None:
        spread = 1.0
        sigma = sigma#10.0
    elif z >= 0:
        spread = 1.0
    else:
        spread = max(1.0 + z/2.0, 0.5)#max(1.0 - (z/2.0), 0.5)
    return np.exp(-D2 / 2.0 / sigma / sigma * spread)
    
def generate_heatmaps(img_shape, stride, kpt_2d, kpt_3d, combined=False, sigma=2.0, is_ratio=False, replace_wrist=False):
    kpt_2d = kpt_2d.copy()
    kpt_3d = kpt_3d.copy() if kpt_3d is not None else None 
        
    height, width = img_shape[:2]
    if is_ratio:
        kpt_2d *= np.array([height, width])

    if replace_wrist:
        kpt_2d[0] = np.mean(kpt_2d[[0, 9], :], 0)

    heatmaps = np.zeros((int(height / stride), int(width / stride), len(kpt_2d) + 1), dtype=np.float32)
    #sigma = 3.0
    for i in range(len(kpt_2d)):
        y = int(kpt_2d[i][0]) * 1.0 / stride
        x = int(kpt_2d[i][1]) * 1.0 / stride
        z = kpt_3d[i][2] * 1.0 * 8.0 / stride if kpt_3d is not None else None
        heat_map = gaussian_kernel(size_h=height / stride, size_w=width / stride, center_x=x, center_y=y, sigma=sigma, z = z)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        heatmaps[:, :, i + 1] = heat_map

    if not combined:
        heatmaps[:, :, 0] = 1.0 - np.max(heatmaps[:, :, 1:], axis=2)  # for background
    else:
        heatmaps = np.max(heatmaps[:, :, 1:], axis=2)  # combine all foregrounds
    if not np.any(kpt_2d):
        heatmaps.fill(0.0)
    return heatmaps
    
def get_kpts(maps, img_h, img_w, num_keypoints, flip = False):
    # maps (1,?,img_h,img_w)
    if not isinstance(maps, np.ndarray):
        maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]
    kpts = [None for i in range(num_keypoints)]
    for idx, m in enumerate(map_6[1:]):
        h, w = np.unravel_index(m.argmax(), m.shape)
        col = int(w * img_w / m.shape[1])
        row = int(h * img_h / m.shape[0])
        kpts[idx] = [row,col] if not flip else [row, img_w - col]
    
    #a = np.amax(map_6, axis=0) * (200.0 / np.amax(map_6))
    #cv2.imwrite("outputs/test.png", a)  
    return kpts
    
def paint_kpts(img_path, img, kpts, circle_size = 1):
    colors = params.colors
    # To be continued
    limbSeq = params.limbSeq_ganhand
    
    im = cv2.imread(img_path) if img is None else img.copy()
    # draw points
    for k, kpt in enumerate(kpts):
        row = int(kpt[0])
        col = int(kpt[1])
        if k in [0, 4, 8, 12, 16, 20]:
            r = circle_size
        else:
            r = 1
        cv2.circle(im, (col, row), radius=r, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [X0, Y0] = kpts[limb[0]]
        [X1, Y1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_im, polygon, colors[i])#
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

    #cv2.imshow('test_example', im)
    #cv2.waitKey(0)
    #cv2.imwrite('test_example.png', im)
    return im
    
def save_model(state, is_best, is_last, filename):
    if is_last:
        torch.save(state, filename + '_pretrained.pth.tar')
    else:
        if is_best:
            torch.save(state, filename + '_best.pth.tar')
        else:
            torch.save(state, filename + '_latest.pth.tar')