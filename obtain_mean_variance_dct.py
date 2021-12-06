import os
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
from jpeg2dct.numpy import load, loads

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

def read_ego2hands_files():
    ego2hands_root_dir = "/home/alex/Documents/Data/Ego2Hands/train"
    img_path_list = []
    energy_path_list = []
    kpts_2d_glob_path_list = []
    for root, dirs, files in os.walk(ego2hands_root_dir):
        for file_name in files:
            if file_name.endswith(".png") and "_" not in file_name:
                kpts_2d_path = os.path.join(root, "kpts_2d_glob.npy")
                if os.path.exists(kpts_2d_path):
                    img_path = os.path.join(root, file_name)
                    img_path_list.append(img_path)
                    energy_path = img_path.replace(".png", "_energy.png")
                    energy_path_list.append(energy_path)
                    kpts_2d_glob_path_list.append(kpts_2d_path)
    return img_path_list, energy_path_list, kpts_2d_glob_path_list

img_path_list, energy_path_list, kpts_2d_glob_path_list = read_ego2hands_files()
img_h, img_w = params.IMG_H, params.IMG_W
crop_size = params.HAND_2D_CROP_SIZE
num_samples = 100000
dct_size = 56
mean_sum = np.zeros(64*3)
std_sum = np.zeros(64*3)
for i in range(num_samples):
    right_i = random.randint(0, len(img_path_list) - 1)
    right_img_init = cv2.imread(img_path_list[right_i], cv2.IMREAD_UNCHANGED)
    right_img_init = right_img_init.astype(np.float32)
    right_img_init = cv2.resize(right_img_init, (img_w, img_h))
    right_seg_init = right_img_init[:,:,-1] > 128
    right_energy_init = cv2.imread(energy_path_list[right_i], 0)
    right_energy_init = cv2.resize(right_energy_init, (img_w, img_h)).astype(np.float32)/255.0
    right_kpts_2d_glob_init = np.load(kpts_2d_glob_path_list[right_i])

    right_img_orig = right_img_init.copy()
    brightness_val = random.randint(15, 240)
    right_img_init = change_mean_brightness(right_img_init, right_seg_init, brightness_val, 20, img_path_list[right_i])
    right_img_init = random_smoothness(right_img_init)

    bg_img = np.zeros((img_h, img_w), dtype=np.uint8)
    
    right_img, right_seg, right_energy, right_kpts_2d_glob = seg_augmentation_w_kpts(right_img_init, right_seg_init, right_energy_init, right_kpts_2d_glob_init)
    img_top_hand, img_bot_hand = merge_hands_no_bg(right_img, None)
    
    img_top_cropped, img_bot_cropped, seg_cropped, kpts_2d_cropped = crop_hand_for_2d(img_top_hand, img_bot_hand, right_energy, right_seg.astype(np.uint8)*255, right_kpts_2d_glob, img_size=crop_size)
    img_top_cropped = cv2.cvtColor(img_top_cropped, cv2.COLOR_RGB2GRAY)
    img_top_cropped[seg_cropped < 128] = 0
    tmp_save_path = "temp/hand_cropped.jpg"
    cv2.imwrite(tmp_save_path, img_top_cropped)
    
    dct_y, dct_cb, dct_cr = load(tmp_save_path)

    dct_y = cv2.resize(dct_y, (dct_size, dct_size))
    dct_cb = cv2.resize(dct_cb, (dct_size, dct_size))
    dct_cr = cv2.resize(dct_cr, (dct_size, dct_size))

    dct_combined = np.concatenate((dct_y, dct_cb, dct_cr), -1)
    dct_combined = dct_combined.transpose(2, 0, 1).reshape(64*3, -1)
    mean_sum += np.mean(dct_combined, -1)
    std_sum += np.std(dct_combined, -1)
    print("{}, mean = {}, std = {}".format(i, mean_sum[0], std_sum[0]))
mean_sum /= num_samples
std_sum /= num_samples
np.save("temp/dct_mean_192.npy", mean_sum)
np.save("temp/dct_std_192.npy", std_sum)
print("finished.")
