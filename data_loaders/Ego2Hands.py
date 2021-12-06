import os
import sys
import numpy as np
import torch
import torch.utils.data as data
import cv2
import random
import math
from utils import *
import params

def random_bg_augment(img, img_path = "", bg_adapt = False, brightness_aug = True, flip_aug = True):
    if brightness_aug:
        if bg_adapt:
            brightness_mean = int(np.mean(img))
            brightness_val = random.randint(brightness_mean - 50, brightness_mean + 50)
            img = change_mean_brightness(img, None, brightness_val, 20, img_path)
        else:
            brightness_val = random.randint(35, 220)
            img = change_mean_brightness(img, None, brightness_val, 20, img_path)
    
    img = img.astype("uint8")
    
    if flip_aug:
        do_flip = bool(random.getrandbits(1))
        if do_flip:
            img = cv2.flip(img, 1)
    return img

def resize_bg(fg_shape, bg_img, bg_adapt):
    fg_h, fg_w = fg_shape[:2]
    if not bg_adapt:
        bg_h, bg_w = bg_img.shape[:2]
        
        if bg_h < fg_h or bg_w < fg_w:
            fb_h_ratio = float(fg_h)/bg_h
            fb_w_ratio = float(fg_w)/bg_w
            bg_resize_ratio = max(fb_h_ratio, fb_w_ratio)
            bg_img = cv2.resize(bg_img, (int(math.ceil(bg_img.shape[1]*bg_resize_ratio)), int(math.ceil(bg_img.shape[0]*bg_resize_ratio))))
        bg_h, bg_w = bg_img.shape[:2]

        # Get row/col offsets
        bg_h_offset_range = max(bg_h - fg_h, 0)
        bg_w_offset_range = max(bg_w - fg_w, 0)

        bg_h_offset = random.randint(0, bg_h_offset_range)
        bg_w_offset = random.randint(0, bg_w_offset_range)
        bg_img = bg_img[bg_h_offset:bg_h_offset+fg_h, bg_w_offset:bg_w_offset+fg_w, :3]
    else:
        bg_img = cv2.resize(bg_img, (fg_w, fg_h))
    return bg_img

def crop_bg(bg_img, crop_shape):
    bg_h, bg_w = bg_img.shape[:2]
    crop_h, crop_w = crop_shape
    #print("{}, {}".format(bg_img.shape, crop_shape))
    row_start = random.randint(0, bg_h - crop_h)
    row_end = row_start + crop_h
    col_start = random.randint(0, bg_w - crop_w)
    col_end = col_start + crop_w
    bg_img_cropped = bg_img[row_start:row_end, col_start:col_end]
    return bg_img_cropped

def add_alpha_image_to_bg(alpha_img, bg_img):
    alpha_s = np.repeat((alpha_img[:,:,3]/255.0)[:,:,np.newaxis], 3, axis=2)
    alpha_l = 1.0 - alpha_s
    combined_img = np.multiply(alpha_s ,alpha_img[:,:,:3]) + np.multiply(alpha_l, bg_img)
    return combined_img
    
def add_alpha_border(hand_img):
    fg_mask = (hand_img[:,:,-1] == 0).astype(np.uint8)
    fg_mask = cv2.dilate(fg_mask, np.ones((3, 3)))
    alpha_mask = fg_mask * 255
    alpha_mask = 255 - cv2.GaussianBlur(alpha_mask, (7, 7), 0)
    #alpha_mask[np.logical_not(fg_mask)] = 255
    hand_img[:,:,-1] = alpha_mask
    hand_seg = alpha_mask > 200
    hand_all_seg = alpha_mask > 0
    return hand_img, hand_seg, hand_all_seg

def merge_hands(img_top_hand, img_bot_hand, bg_img, bg_adapt, bg_resize = True):
    if img_top_hand is not None and img_bot_hand is not None:
        img_bot_hand, _, _ = add_alpha_border(img_bot_hand)
        img_top_hand, _, _ = add_alpha_border(img_top_hand)
        bg_img_resized = resize_bg(img_bot_hand.shape, bg_img, bg_adapt) if bg_resize else bg_img
        combined_hand_img = add_alpha_image_to_bg(img_bot_hand, bg_img_resized)
        combined_hand_img = add_alpha_image_to_bg(img_top_hand, combined_hand_img)
    else:
        img_top_hand, _, _ = add_alpha_border(img_top_hand)
        bg_img_resized = resize_bg(img_top_hand.shape, bg_img, bg_adapt) if bg_resize else bg_img
        combined_hand_img = add_alpha_image_to_bg(img_top_hand, bg_img_resized)
    return combined_hand_img, bg_img_resized
    
def merge_hands_no_bg(img_top_hand, img_bot_hand):
    assert img_top_hand is not None
    if img_bot_hand is not None:
        img_bot_hand, _, _ = add_alpha_border(img_bot_hand)
        img_top_hand, _, _ = add_alpha_border(img_top_hand)
    else:
        img_top_hand, _, _ = add_alpha_border(img_top_hand)
    return img_top_hand, img_bot_hand
    
def merge_hands_cropped(img_top_hand, img_bot_hand, bg_img):
    bg_img_cropped = crop_bg(bg_img, img_top_hand.shape[:2])
    if img_bot_hand is not None:
        combined_hand_img = add_alpha_image_to_bg(img_bot_hand, bg_img_cropped)
        combined_hand_img = add_alpha_image_to_bg(img_top_hand, combined_hand_img)
    else:
        combined_hand_img = add_alpha_image_to_bg(img_top_hand, bg_img_cropped)
    return combined_hand_img
    
def change_mean_brightness(img, seg, brightness_val, jitter_range = 20, img_path = ""):
    if seg is not None:
        old_mean_val = np.mean(img[seg])
    else:
        old_mean_val = np.mean(img)
    assert old_mean_val != 0, "ERROR: {} has mean of 0".format(img_path)
    new_mean_val = brightness_val + random.uniform(-jitter_range/2, jitter_range/2)
    img *= (new_mean_val/old_mean_val)
    img = np.clip(img, 0, 255)
    return img
    
def random_smoothness(img, smooth_rate = 0.3):
    smooth_rate_tick = smooth_rate/5
    rand_val = random.random()
    if rand_val < smooth_rate:
        if rand_val < smooth_rate_tick:
            kernel_size = 3
        elif rand_val < smooth_rate_tick*2:
            kernel_size = 5
        elif rand_val < smooth_rate_tick*3:
            kernel_size = 7
        elif rand_val < smooth_rate_tick*4:
            kernel_size = 9
        else:
            kernel_size = 11
        img[:,:,:3] = cv2.blur(img[:,:,:3], (kernel_size, kernel_size))
    return img

def read_ego2hands_files(args, config, all_data, seq_i = -1):
    if args.eval:
        assert seq_i != -1
        img_path_list = []
        kpts_2d_glob_path_list = []
        ego2hands_eval_seq_dir = os.path.join(config.dataset_eval_dir, 'eval_seq{}_imgs')
        for root, dirs, files in os.walk(ego2hands_eval_seq_dir):
            for file_name in files:
                if file_name.endswith('.png') and '_' not in file_name:
                    img_path = os.path.join(root, file_name)
                    img_path_list.append(img_path)
                    kpts_2d_path = os.path.join(root, file_name.replace('.png', '_kpts_2d_glob_r.npy'))
                    kpts_2d_glob_path_list.append(kpts_2d_path)
        return img_path_list, kpts_2d_glob_path_list
    else:
        ego2hands_root_dir = config.dataset_train_dir
        if not all_data:
            img_path_list = []
            energy_path_list = []
            kpts_2d_glob_path_list = []
            kpts_3d_can_path_list = []
            mano_path_list = []
            for root, dirs, files in os.walk(ego2hands_root_dir):
                for file_name in files:
                    if file_name.endswith(".png") and "energy" not in file_name and "vis" not in file_name:
                        kpts_2d_path = os.path.join(root, "kpts_2d_glob.npy")
                        if os.path.exists(kpts_2d_path):
                            img_path = os.path.join(root, file_name)
                            img_path_list.append(img_path)
                            energy_path = img_path.replace(".png", "_energy.png")
                            energy_path_list.append(energy_path)
                            kpts_2d_glob_path_list.append(kpts_2d_path)
                            mano_path = os.path.join(root, 'mano_stage3.npy')
                            if not os.path.exists(mano_path):
                                mano_path = os.path.join(root, 'mano_stage2.npy')
                            mano_path_list.append(mano_path)
                            kpts_3d_can_path = os.path.join(root, 'kpts_3d_can_stage3.npy')
                            if not os.path.exists(kpts_3d_can_path):
                                kpts_3d_can_path = os.path.join(root, 'kpts_3d_can_stage2.npy')
                            kpts_3d_can_path_list.append(kpts_3d_can_path)
            return img_path_list, energy_path_list, kpts_2d_glob_path_list, mano_path_list, kpts_3d_can_path_list
        else:
            img_path_list = []
            energy_path_list = []
            for root, dirs, files in os.walk(ego2hands_root_dir):
                for file_name in files:
                    if file_name.endswith(".png") and "energy" not in file_name and "vis" not in file_name:
                        img_path = os.path.join(root, file_name)
                        img_path_list.append(img_path)
                        energy_path = img_path.replace(".png", "_energy.png")
                        energy_path_list.append(energy_path)
            return img_path_list, energy_path_list
    
def read_bg_data(args, config, bg_adapt, seq_i):
    if not bg_adapt:
        root_bg_dir = config.bg_all_dir
    else:
        root_bg_dir = os.path.join(config.dataset_eval_dir, "eval_seq{}_bg".format(seq_i))
        
    # backgrounds
    bg_path_list = []
    for root, dirs, files in os.walk(root_bg_dir):
        for file_name in files:
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                bg_path_list.append(os.path.join(root, file_name))
    return bg_path_list

def get_random_brightness_for_scene(args, config, bg_adapt, seq_i):
    dark_lighting_set = [5]
    normal_lighting_set = [1, 3, 4, 6, 7]
    bright_lighting_set = [2, 8]
    brightness_map = {"dark": (0, 55), "normal": (55, 200), "bright": (55, 255)}
    if not bg_adapt:
        return random.randint(15, 240)
        #random.randint(15, 240)
    else:
        if True:#not args.custom:
            if seq_i in dark_lighting_set:
                return random.randint(*brightness_map["dark"])
            elif seq_i in normal_lighting_set:
                return random.randint(*brightness_map["normal"])
            elif seq_i in bright_lighting_set:
                return random.randint(*brightness_map["bright"])
        else:
            assert config.custom_scene_brightness != "", "Error: custom scene brightness not set. Please set \"custom_scene_brightness\" in the config file."
            assert config.custom_scene_brightness in brightness_map, "Error: unrecognized brightness {} (valid options [\"dark\", \"normal\", \"bright\"]".format(config.custom_scene_brightness)
            return random.randint(*brightness_map[config.custom_scene_brightness])

LEFT_IDX = 1
RIGHT_IDX = 2

class Ego2HandsData(data.Dataset):
    def __init__(self, args, config, seq_i = -1):
        self.args = args
        self.config = config
        self.bg_adapt = args.adapt
        self.seq_i = seq_i
        self.input_edge = True
        self.bg_list = read_bg_data(self.args, self.config, self.bg_adapt, self.seq_i)
        self.img_path_all_list = None
        if not self.args.eval:
            self.img_path_all_list, self.energy_path_all_list = read_ego2hands_files(self.args, self.config, all_data = True)
            self.img_path_list, self.energy_path_list, self.kpts_2d_glob_path_list, self.mano_path_list, self.kpts_3d_can_path_list = read_ego2hands_files(self.args, self.config, all_data = False)
        else:   
            self.img_path_list, self.kpts_2d_glob_path_list = read_ego2hands_files(self.args, self.config)
                
        self.img_h, self.img_w = params.IMG_H, params.IMG_W
        self.crop_size = params.HAND_2D_CROP_SIZE
        self.crop_stride = params.HAND_2D_CROP_STRIDE
        self.valid_hand_seg_th = params.VALID_HAND_SEG_TH
        self.valid_energy_th_ratio = params.VALID_ENERGY_TH_RATIO
        self.EMPTY_IMG_ARRAY = np.zeros((1, 1))
        self.EMPTY_BOX_ARRAY = np.zeros([0, 0, 0, 0])
        #self.jpeg = TurboJPEG('/home/alex/anaconda3/envs/pose_env/lib/libturbojpeg.so')#('/usr/lib/libturbojpeg.so')
        self.mean_dct = np.reshape(np.load("files_saved/dct_mean_192.npy")[:64], (1, 1, 64))
        self.std_dct = np.reshape(np.load("files_saved/dct_std_192.npy")[:64], (1, 1, 64))
        self.dct_size = 56
        print("Loading finished")
        if self.img_path_all_list is not None:
            print("#hand imgs all: {}".format(len(self.img_path_all_list)))
        print("#hand imgs: {}".format(len(self.img_path_list)))
        print("#bg imgs: {}".format(len(self.bg_list)))

    def __getitem__(self, index):
        if not self.args.eval:
            # Left hand
            left_i = random.randint(0, len(self.img_path_all_list) - 1)
            left_img = cv2.imread(self.img_path_all_list[left_i], cv2.IMREAD_UNCHANGED)
            assert left_img is not None, "Error, image not found: {}".format(self.img_path_all_list[left_i])
            left_img = left_img.astype(np.float32)
            left_img = cv2.resize(left_img, (self.img_w, self.img_h))
            left_img = cv2.flip(left_img, 1)
            left_seg = left_img[:,:,-1] > 128
            left_energy = cv2.imread(self.energy_path_all_list[left_i], 0)
            left_energy = cv2.resize(left_energy, (self.img_w, self.img_h)).astype(np.float32)/255.0
            left_energy = cv2.flip(left_energy, 1)
            left_img_orig = left_img.copy()
            # Brightness Augmentation
            brightness_val = get_random_brightness_for_scene(self.args, self.config, self.bg_adapt, self.seq_i)
            left_img = change_mean_brightness(left_img, left_seg, brightness_val, 20, self.img_path_all_list[left_i])
            left_img = random_smoothness(left_img)

            # Right hand
            right_i = random.randint(0, len(self.img_path_list) - 1)
            right_img_init = cv2.imread(self.img_path_list[right_i], cv2.IMREAD_UNCHANGED)
            assert right_img_init is not None, "Error, image not found: {}".format(self.img_path_list[right_i])
            right_img_init = right_img_init.astype(np.float32)
            right_img_init = cv2.resize(right_img_init, (self.img_w, self.img_h))
            right_seg_init = right_img_init[:,:,-1] > 128
            right_energy_init = cv2.imread(self.energy_path_list[right_i], 0)
            right_energy_init = cv2.resize(right_energy_init, (self.img_w, self.img_h)).astype(np.float32)/255.0
            right_kpts_2d_glob_init = np.load(self.kpts_2d_glob_path_list[right_i])
            mano_init = np.load(self.mano_path_list[right_i])[-45:]
            kpts_3d_can_init = np.load(self.kpts_3d_can_path_list[right_i])
            kpts_3d_can_selected = (kpts_3d_can_init[[5, 17],:] - kpts_3d_can_init[0]).reshape(-1)
            right_img_orig = right_img_init.copy()
            #right_img, right_seg, right_energy = seg_augmentation_wo_kpts(right_img, right_seg, right_energy)
            # Brightness Augmentation
            right_img_init = change_mean_brightness(right_img_init, right_seg_init, brightness_val, 20, self.img_path_list[right_i])
            right_img_init = random_smoothness(right_img_init)

            # Find background
            bg_img = None
            while(bg_img is None):
                bg_i = random.randint(0, len(self.bg_list) - 1)
                bg_img = cv2.imread(self.bg_list[bg_i]).astype(np.float32)
                if bg_img.shape[0] < self.crop_size or bg_img.shape[1] < self.crop_size:
                    bg_img = None
                    continue
                if not self.bg_adapt:
                    bg_img = random_bg_augment(bg_img, self.bg_list[bg_i], bg_adapt = self.bg_adapt)
                else:
                    bg_img = random_bg_augment(bg_img, self.bg_list[bg_i], bg_adapt = self.bg_adapt, flip_aug = False)
                bg_img = random_smoothness(bg_img)
            #print("Selected bg shape ", bg_img.shape)
                
            # Find a well-merged sample
            merge_is_good = False
            merge_count = 0
            while not merge_is_good:
                merge_mode = random.randint(0, 8)
                merge_count += 1
                if merge_count >= 100:
                    print("Stuck in merge, mode = {}".format(merge_mode))
                    print(self.img_path_all_list[left_i])
                    print(self.img_path_all_list[right_i])
                    sys.exit()
                # Left hand augmentation with random translation
                left_img, left_seg, left_energy = seg_augmentation_wo_kpts(left_img, left_seg, left_energy)
                # Right hand augmentation with random translation
                right_img, right_seg, right_energy, right_kpts_2d_glob = seg_augmentation_w_kpts(right_img_init, right_seg_init, right_energy_init, right_kpts_2d_glob_init)
                right_energy_mask = right_energy > 0.5
                right_energy_sum = np.sum(right_energy_mask)
                
                if merge_mode < 8:
                    if np.sum(left_energy) > np.sum(right_energy):
                        # left hand first
                        merge_mode = 0
                    else:
                        # right hand first
                        merge_mode = 4
                
                # Merge hands
                if merge_mode < 4:
                    # left hand top, right hand bottom
                    #img_merged, bg_img_resized = merge_hands(left_img, right_img, bg_img, self.bg_adapt)
                    img_top_hand, img_bot_hand = merge_hands_no_bg(left_img, right_img)
                    seg_merged = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                    seg_merged[right_seg] = RIGHT_IDX
                    seg_merged[left_seg] = LEFT_IDX
                    # Check for hand with insufficient size
                    right_energy_visible = np.logical_and(seg_merged == RIGHT_IDX, right_energy_mask)
                    #merge_count += 1
                    if right_energy_visible.sum() >= right_energy_sum*self.valid_energy_th_ratio:
                        merge_is_good = True
                elif merge_mode >= 4 and merge_mode < 8:
                    # left hand bottom, right hand top
                    #img_merged, bg_img_resized = merge_hands(right_img, left_img, bg_img, self.bg_adapt)
                    img_top_hand, img_bot_hand = merge_hands_no_bg(right_img, left_img)
                    seg_merged = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                    seg_merged[left_seg] = LEFT_IDX
                    seg_merged[right_seg] = RIGHT_IDX
                    merge_is_good = True
                    # Check for hand with insufficient size
                    #left_mask = seg_merged == LEFT_IDX
                    #if left_mask.sum() < self.valid_hand_seg_th:
                    #    seg_merged[left_mask] = 0
                    #    left_energy.fill(0.0)
                elif merge_mode == 8:
                    # drop left hand, right hand only
                    #img_merged, bg_img_resized = merge_hands(right_img, None, bg_img, self.bg_adapt)
                    img_top_hand, img_bot_hand = merge_hands_no_bg(right_img, None)
                    seg_merged = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                    seg_merged[right_seg] = RIGHT_IDX
                    left_energy.fill(0.0)
                    merge_is_good = True
                
            # Obtain cropped hand and heatmaps
            img_top_cropped, img_bot_cropped, seg_cropped, kpts_2d_cropped = crop_hand_for_2d(img_top_hand, img_bot_hand, right_energy, (seg_merged == RIGHT_IDX).astype(np.uint8)*255, right_kpts_2d_glob, img_size=self.crop_size)
            img_cropped = merge_hands_cropped(img_top_cropped, img_bot_cropped, bg_img)
            img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
            edge_cropped = cv2.Canny(img_cropped.astype(np.uint8), 25, 100).astype(np.float32)
            
            # transform to frequency domain
            img_cropped[seg_cropped < 128] = 0
            img_cropped_freq = img_cropped#img2freq(img_cropped)
            edge_cropped_freq = edge_cropped#img2freq(edge_cropped)#
            seg_cropped_freq = seg_cropped#img2freq(seg_cropped)#
            
            if self.args.use_seg:
                img_input = np.expand_dims(img_cropped_freq, 0)#np.stack((img_cropped_freq, edge_cropped_freq, seg_cropped_freq), 0)
                #img_input = img_cropped_freq
                #img_input = (img_input - self.mean_dct) / self.std_dct
                #img_input = cv2.resize(img_input, (self.dct_size, self.dct_size))
                #img_input = img_input.transpose(2, 0, 1)
                #print(img_input.shape)
                #print(img_input)
                img_input_rgb = np.expand_dims(img_cropped, -1)#np.stack((img_cropped, edge_cropped, seg_cropped), -1)
            else:
                #img_input = np.stack((img_cropped_freq, edge_cropped_freq, np.zeros_like(edge_cropped_freq)), 0)
                img_input_rgb = np.stack((img_cropped, edge_cropped, np.zeros_like(edge_cropped)), -1)
            heatmaps = generate_heatmaps((self.crop_size, self.crop_size), self.crop_stride, kpts_2d_cropped, None, is_ratio=True)
                
            # Prepare tensors
            img_input_tensor = torch.from_numpy(img_input).float()#normalize_tensor(torch.from_numpy(img_input), 128.0, 256.0)
            img_input_rgb_tensor = torch.from_numpy(img_input_rgb)
            heatmaps_tensor = torch.from_numpy(heatmaps.transpose(2, 0, 1))
            pose_np = np.concatenate((mano_init, kpts_3d_can_selected))
            mano_tensor = torch.from_numpy(pose_np)

            return img_input_tensor, heatmaps_tensor, img_input_rgb_tensor, mano_tensor
        else:
            # Prepare image
            img_real_test = cv2.imread(self.img_path_list[index]).astype(np.float32)
            img_real_test = cv2.resize(img_real_test, (self.img_w, self.img_h))
            img_real_orig = img_real_test.copy()
            img_real_test = cv2.cvtColor(img_real_test, cv2.COLOR_RGB2GRAY)
            
            # For input edge map
            #img_edge = cv2.Canny(img_real_test.astype(np.uint8), 25, 100).astype(np.float32)
            #img_real_test = np.stack((img_real_test, img_edge), -1)
            
            # Prepare segmentation gt
            seg_gt_test = (cv2.imread(self.seg_path_list[index], 0)/50).astype(np.uint8)
            seg_gt_test = cv2.resize(seg_gt_test, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)

            # Prepare energy
            energy_l_gt = cv2.resize(cv2.imread(self.energy_l_path_list[index]), (self.img_w, self.img_h)).astype(np.float32)/255.0
            box_l_np = get_bounding_box_from_energy(energy_l_gt, close_op = False)
            energy_r_gt = cv2.resize(cv2.imread(self.energy_r_path_list[index]), (self.img_w, self.img_h)).astype(np.float32)/255.0
            box_r_np = get_bounding_box_from_energy(energy_r_gt, close_op = False)

            # Crop left hand for 2D
            #print("Left")
            #print(self.img_path_list[index])
            #print(np.histogram(energy_l_gt))
            img_l_cropped, _, seg_l_cropped, _ = crop_hand_for_2d(img_real_test, None, energy_l_gt, (seg_gt_test == LEFT_IDX).astype(np.uint8)*255, None, img_size=self.crop_size, side = 0)
            if img_l_cropped is not None:
                edge_l_cropped = cv2.Canny(img_l_cropped.astype(np.uint8), 25, 100).astype(np.float32)
                img_l_input = np.stack((img_l_cropped, edge_l_cropped, seg_l_cropped), 0)
                img_l_input_tensor = normalize_tensor(torch.from_numpy(img_l_input), 128.0, 256.0)
            else:
                img_l_input_tensor = self.EMPTY_IMG_ARRAY;
            # Crop right hand for 2D
            #print("Right")
            #print(self.img_path_list[index])
            #print(np.histogram(energy_r_gt))
            img_r_cropped, _, seg_r_cropped, _ = crop_hand_for_2d(img_real_test, None, energy_r_gt, (seg_gt_test == RIGHT_IDX).astype(np.uint8)*255, None, img_size=self.crop_size)
            if img_r_cropped is not None:
                edge_r_cropped = cv2.Canny(img_r_cropped.astype(np.uint8), 25, 100).astype(np.float32)
                img_r_input = np.stack((img_r_cropped, edge_r_cropped, seg_r_cropped), 0)
                img_r_input_tensor = normalize_tensor(torch.from_numpy(img_r_input), 128.0, 256.0)
            else:
                img_r_input_tensor = self.EMPTY_IMG_ARRAY;
            
            return img_l_input_tensor, img_r_input_tensor
           
    def __len__(self):
        return len(self.img_path_list)

