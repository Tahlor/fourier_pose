import os
import sys
import random
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

def read_ego2hands_files():
    ego2hands_root_dir = '/home/alex/Documents/Data/Ego2Hands/train'
    img_path_list = []
    energy_path_list = []
    kpts_2d_glob_path_list = []
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
    return img_path_list, energy_path_list, kpts_2d_glob_path_list

def read_bg_data():
    root_bg_dir = '/home/alex/Documents/Data/backgrounds'
    # backgrounds
    bg_path_list = []
    for root, dirs, files in os.walk(root_bg_dir):
        for file_name in files:
            if file_name.endswith(".jpg") or file_name.endswith(".png"):
                bg_path_list.append(os.path.join(root, file_name))
    return bg_path_list

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

def merge_hands_no_bg(img_top_hand, img_bot_hand):
    assert img_top_hand is not None
    if img_bot_hand is not None:
        img_bot_hand, _, _ = add_alpha_border(img_bot_hand)
        img_top_hand, _, _ = add_alpha_border(img_top_hand)
    else:
        img_top_hand, _, _ = add_alpha_border(img_top_hand)
    return img_top_hand, img_bot_hand

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

def merge_hands_cropped(img_top_hand, img_bot_hand, bg_img):
    bg_img_cropped = crop_bg(bg_img, img_top_hand.shape[:2])
    if img_bot_hand is not None:
        combined_hand_img = add_alpha_image_to_bg(img_bot_hand, bg_img_cropped)
        combined_hand_img = add_alpha_image_to_bg(img_top_hand, combined_hand_img)
    else:
        combined_hand_img = add_alpha_image_to_bg(img_top_hand, bg_img_cropped)
    return combined_hand_img

def img2freq(img):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]) + 0.000000001)
    magnitude_spectrum = magnitude_spectrum/np.amax(magnitude_spectrum)*255
    return magnitude_spectrum

# Macro
IMG_H = 288
IMG_W = 512
CROP_SIZE = 224

# Read data
img_path_list, energy_path_list, kpts_2d_glob_path_list = read_ego2hands_files()
bg_list = read_bg_data()

# Load data
right_i = 0
right_img_init = cv2.imread(img_path_list[right_i], cv2.IMREAD_UNCHANGED)
right_img_init = right_img_init.astype(np.float32)
right_img_init = cv2.resize(right_img_init, (IMG_W, IMG_H))
right_seg_init = right_img_init[:,:,-1] > 128
right_energy_init = cv2.imread(energy_path_list[right_i], 0)
right_energy_init = cv2.resize(right_energy_init, (IMG_W, IMG_H)).astype(np.float32)/255.0
right_kpts_2d_glob_init = np.load(kpts_2d_glob_path_list[right_i])

# Find background
"""
bg_img = None
while(bg_img is None):
    bg_i = random.randint(0, len(bg_list) - 1)
    bg_img = cv2.imread(bg_list[bg_i]).astype(np.float32)
    if bg_img.shape[0] < CROP_SIZE or bg_img.shape[1] < CROP_SIZE:
        bg_img = None
        continue
    bg_img = random_bg_augment(bg_img, bg_list[bg_i], bg_adapt = False, flip_aug = False)
    bg_img = random_smoothness(bg_img)
"""
#print(bg_img.shape)
#print(bg_img.dtype)
bg_img = np.zeros((1000, 2000, 3), dtype=np.uint8)

# Translation augmentation
right_img, right_seg, right_energy, right_kpts_2d_glob = seg_augmentation_w_kpts(right_img_init, right_seg_init, right_energy_init, right_kpts_2d_glob_init)

# Image composition
img_top_hand, img_bot_hand = merge_hands_no_bg(right_img, None)
img_top_cropped, img_bot_cropped, seg_cropped, kpts_2d_cropped = crop_hand_for_2d(img_top_hand, img_bot_hand, right_energy, right_seg.astype(np.uint8)*255, right_kpts_2d_glob, img_size=CROP_SIZE)

img_cropped_no_bg = cv2.cvtColor(img_top_cropped, cv2.COLOR_RGB2GRAY)
img_cropped_no_bg[seg_cropped < 128] = 0
img_cropped = merge_hands_cropped(img_top_cropped, img_bot_cropped, bg_img)
img_cropped = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
edge_cropped = cv2.Canny(img_cropped.astype(np.uint8), 25, 100).astype(np.float32)

# Save init RGB image
cv2.imwrite('test_dir/img_rgb_no_bg.png', img_cropped_no_bg)
cv2.imwrite('test_dir/img_rgb_init.png', img_cropped)
cv2.imwrite('test_dir/img_edge_init.png', edge_cropped)

# Frequency domain
img_no_bg_cropped_freq = img2freq(img_cropped_no_bg)
img_cropped_freq = img2freq(img_cropped)
edge_cropped_freq = img2freq(edge_cropped)
seg_cropped_freq = img2freq(seg_cropped)

cv2.imwrite('test_dir/freq_img_rgb_no_bg.png', img_no_bg_cropped_freq)
cv2.imwrite('test_dir/freq_img_rgb.png', img_cropped_freq)
cv2.imwrite('test_dir/freq_img_edge.png', edge_cropped_freq)
cv2.imwrite('test_dir/freq_img_seg.png', seg_cropped_freq)


