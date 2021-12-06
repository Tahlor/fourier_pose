import os
import sys
import cv2
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from utils import *

IMG_H = 288
IMG_W = 512
CROP_SIZE = 224
DISPLAY_MODE_IMG = 'img'
DISPLAY_MODE_SEG = 'seg'
DISPLAY_MODE_EDGE = 'edge'
SHIFT_HOR_MAX = 100
SHIFT_VER_MAX = 100
COLOR_WHITE = '#ffffff'
BUTTON_H = 2
BUTTON_W = 10

class interactive_hand_app:
    def __init__(self):
        # prepare UI
        self.init_window()

    def run_app(self):
        self.window.mainloop()

    def init_window(self):
        # Create tkinter UI
        self.window = tk.Tk()
        self.window.title('Interactive Hand App')
        self.load_data_paths()
        self.img_i = 0
        # Load 2D model
        self.model_2d = construct_model_2d()
        self.model_2d.eval()
        # Set up UI panels
        self.window.panel_top_hor = None
        self.annotation_mode = None
        self.init_panels()

    def init_panels(self):
        # Top horizontal panel
        if self.window.panel_top_hor is not None:
            self.window.panel_top_hor.pack_forget()
            del self.window.panel_top_hor
        self.window.panel_top_hor = tk.PanedWindow(self.window, orient = tk.HORIZONTAL)
        self.window.panel_top_hor.pack(side = tk.LEFT, anchor = tk.NW)
        self.window.panel_image_display_rgb = None
        self.window.panel_image_display_freq = None
        self.window.panel_image_display_pred = None
        self.window.panel_option_ver = None
        self.shift_ver = 0
        self.shift_hor = 0
        self.init_image_panel()
        self.init_option_panel()
        self.update_image()

    def init_image_panel(self):
        # Clear existing panel if exists
        if self.window.panel_image_display_rgb is not None:
            self.window.panel_image_display_rgb.pack_forget()
            del self.window.panel_image_display_rgb
        self.window.panel_image_display_rgb = tk.PanedWindow(self.window.panel_top_hor)
        self.window.panel_image_display_rgb.pack(side=tk.LEFT)
        
        # Clear existing panel if exists
        if self.window.panel_image_display_freq is not None:
            self.window.panel_image_display_freq.pack_forget()
            del self.window.panel_image_display_freq
        self.window.panel_image_display_freq = tk.PanedWindow(self.window.panel_top_hor)
        self.window.panel_image_display_freq.pack(side=tk.LEFT)
        
        # Clear existing panel if exists
        if self.window.panel_image_display_pred is not None:
            self.window.panel_image_display_pred.pack_forget()
            del self.window.panel_image_display_pred
        self.window.panel_image_display_pred = tk.PanedWindow(self.window.panel_top_hor)
        self.window.panel_image_display_pred.pack(side=tk.LEFT)
        
        # Prepare image
        #self.img_path = 
        #self.seq_name, self.img_name = get_nyu_img_name_info(self.img_path)
        #print('image path: {}'.format(self.img_path))
        #self.img_orig = cv2.imread(self.img_path)
        #self.img_orig = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2RGB)
        #self.img_shape_orig = self.img_orig.shape
        #self.img_orig, self.img_input, self.img_display,  _, self.img_padded = self.get_mask_tuple(self.img_orig, is_img = True)
        #self.scale2input = float(self.img_input.shape[0])/self.img_display.shape[0]
        self.prepare_hand_image()
        self.set_augmentaton(None)
        self.model_2d_pred()
        self.display_mode = DISPLAY_MODE_IMG
        self.set_display_images()
        # Prepare gt
        #self.gt_path = self.gt_path_list[self.img_i]
        #print("gt path: {}".format(self.gt_path))
        #self.gt_mask_orig = cv2.imread(self.gt_path, -1) / params_de.OUTPUT_SCALE_NYU_DEPTH_V2
        #self.gt_mask_orig = postprocess_depth_mask(self.gt_mask_orig, is_gt = True)
        #self.valid_mask_orig = get_valid_mask(self.gt_mask_orig, self.args)
        #self.gt_mask_orig, self.gt_mask, self.gt_mask_display, self.gt_mask_vis, self.gt_mask_padded = self.get_mask_tuple(self.gt_mask_orig, is_img = False)
        #self.valid_mask = cv2.resize(self.valid_mask_orig.astype(np.uint8), (self.gt_mask.shape[1], self.gt_mask.shape[0]), interpolation = cv2.INTER_NEAREST)
        #if self.save_visualization:
        #    self.save_gt_result()

        # Set up left canvas
        self.window.image_display_rgb = ImageTk.PhotoImage(Image.fromarray(self.display_img_rgb))
        self.window.canvas_rgb = tk.Canvas(self.window.panel_image_display_rgb, 
            width=self.display_img_rgb.shape[1], height=self.display_img_rgb.shape[0])
        self.window.canvas_img_rgb = self.window.canvas_rgb.create_image(0,0,image=self.window.image_display_rgb)
        self.window.canvas_rgb.config(scrollregion=self.window.canvas_rgb.bbox(tk.ALL))
        self.window.canvas_rgb.pack()
        
        # Set up right canvas
        self.window.image_display_freq = ImageTk.PhotoImage(Image.fromarray(self.display_img_freq))
        self.window.canvas_freq = tk.Canvas(self.window.panel_image_display_freq, 
            width=self.display_img_freq.shape[1], height=self.display_img_freq.shape[0])
        self.window.canvas_img_freq = self.window.canvas_freq.create_image(0,0,image=self.window.image_display_freq)
        self.window.canvas_freq.config(scrollregion=self.window.canvas_freq.bbox(tk.ALL))
        self.window.canvas_freq.pack()
        
        # Set up prediction canvas
        self.window.image_display_pred = ImageTk.PhotoImage(Image.fromarray(self.display_img_pred))
        self.window.canvas_pred = tk.Canvas(self.window.panel_image_display_pred, 
            width=self.display_img_pred.shape[1], height=self.display_img_pred.shape[0])
        self.window.canvas_img_pred = self.window.canvas_pred.create_image(0,0,image=self.window.image_display_pred)
        self.window.canvas_pred.config(scrollregion=self.window.canvas_pred.bbox(tk.ALL))
        self.window.canvas_pred.pack()
        
    def init_option_panel(self):
        if self.window.panel_option_ver is not None:
            self.window.panel_option_ver.pack_forget()
            del self.window.panel_option_ver
        # Vertical panel
        self.window.panel_option_ver = tk.PanedWindow(self.window.panel_top_hor, orient = tk.VERTICAL)
        self.window.panel_option_ver.pack(side=tk.RIGHT, fill=tk.BOTH)
        # Button panel layers
        self.init_button_layer0()
        self.init_button_layer1()
        self.init_button_layer10()
        
    def init_button_layer0(self):
        self.brush_r = 1
        self.window.panel_option_layer0 = tk.PanedWindow(self.window.panel_option_ver, orient = tk.HORIZONTAL)
        self.window.panel_option_layer0.pack(side = tk.TOP, fill=tk.X)
        self.shift_hor_scale = tk.Scale(self.window.panel_option_layer0, label='shift horizontal', from_=1, to=SHIFT_HOR_MAX, 
            orient=tk.HORIZONTAL, showvalue=int(SHIFT_HOR_MAX/2), command=self.shift_horizontal)
        self.shift_hor_scale.set(int(SHIFT_HOR_MAX/2))
        self.shift_hor_scale.pack(side = 'top', fill=tk.X)
        
    def init_button_layer1(self):
        self.brush_r = 1
        self.window.panel_option_layer1 = tk.PanedWindow(self.window.panel_option_ver, orient = tk.HORIZONTAL)
        self.window.panel_option_layer1.pack(side = tk.TOP, fill=tk.X)
        self.shift_ver_scale = tk.Scale(self.window.panel_option_layer1, label='shift vertical', from_=1, to=SHIFT_VER_MAX, 
            orient=tk.HORIZONTAL, showvalue=int(SHIFT_VER_MAX/2), command=self.shift_vertical)
        self.shift_ver_scale.set(int(SHIFT_VER_MAX/2))
        self.shift_ver_scale.pack(side = 'top', fill=tk.X)
        
    def init_button_layer10(self):
        self.window.panel_option_layer10 = tk.PanedWindow(self.window.panel_option_ver, orient = tk.HORIZONTAL)
        self.window.panel_option_layer10.pack(side = tk.BOTTOM)
        self.button_prev = tk.Button(self.window.panel_option_layer10, text = 'prev', bg = COLOR_WHITE, 
            height = BUTTON_H, width = BUTTON_W, command = self.button_prev_callback)
        self.button_prev.pack(side = tk.LEFT)
        self.button_next = tk.Button(self.window.panel_option_layer10, text = 'next', bg = COLOR_WHITE, 
            height = BUTTON_H, width = BUTTON_W, command = self.button_next_callback)
        self.button_next.pack(side = tk.LEFT)
        
    def button_prev_callback(self):
        self.img_i -= 1
        if self.img_i < 0:
            self.img_i = 0
        self.init_panels()
        
    def button_next_callback(self):
        self.img_i += 1
        if self.img_i >= len(self.img_path_list):
            sys.exit()
        self.init_panels()

    def shift_horizontal(self, val):
        self.shift_hor = int(int(val) - SHIFT_HOR_MAX/2)
        self.set_augmentaton((self.shift_ver, self.shift_hor))
        self.model_2d_pred()
        self.set_display_images()
        self.update_image()
        
    def shift_vertical(self, val):
        self.shift_ver = int(int(val) - SHIFT_VER_MAX/2)
        self.set_augmentaton((self.shift_ver, self.shift_hor))
        self.model_2d_pred()
        self.set_display_images()
        self.update_image()

    def load_data_paths(self):
        self.img_path_list, self.energy_path_list, self.kpts_2d_glob_path_list = read_ego2hands_files()
        self.bg_list = read_bg_data()

    def prepare_hand_image(self):
        right_img_init = cv2.imread(self.img_path_list[self.img_i], cv2.IMREAD_UNCHANGED)
        right_img_init = right_img_init.astype(np.float32)
        self.right_img_init = cv2.resize(right_img_init, (IMG_W, IMG_H))
        self.right_seg_init = right_img_init[:,:,-1] > 128
        right_energy_init = cv2.imread(self.energy_path_list[self.img_i], 0)
        self.right_energy_init = cv2.resize(right_energy_init, (IMG_W, IMG_H)).astype(np.float32)/255.0
        self.right_kpts_2d_glob_init = np.load(self.kpts_2d_glob_path_list[self.img_i])

        #self.bg_img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        self.bg_img = None
        while(self.bg_img is None):
            bg_i = random.randint(0, len(self.bg_list) - 1)
            self.bg_img = cv2.imread(self.bg_list[bg_i]).astype(np.float32)
            if self.bg_img.shape[0] < CROP_SIZE or self.bg_img.shape[1] < CROP_SIZE:
                self.bg_img = None
                continue
        
        #img_top_hand, img_bot_hand = merge_hands_no_bg(self.right_img_init, None)
        #img_top_cropped, img_bot_cropped, seg_cropped, kpts_2d_cropped = crop_hand_for_2d(img_top_hand, img_bot_hand, right_energy, right_seg.astype(np.uint8)*255, right_kpts_2d_glob, img_size=CROP_SIZE)
        
    def set_augmentaton(self, shift_augment):
        #right_img, right_seg, right_energy, right_kpts_2d_glob = seg_augmentation_w_kpts(self.right_img_init, self.right_seg_init, self.right_energy_init, self.right_kpts_2d_glob_init, shift_augment)
        self.finish_composition(self.right_img_init, self.right_energy_init, self.right_seg_init, self.right_kpts_2d_glob_init, shift_augment)
        
    def finish_composition(self, hand_img, hand_energy, hand_seg, hand_kpts_2d, shift_augment):
        img_top_hand, img_bot_hand = merge_hands_no_bg(hand_img, None)
        img_top_cropped, img_bot_cropped, seg_cropped, kpts_2d_cropped = crop_hand_for_2d(img_top_hand, img_bot_hand, hand_energy, hand_seg.astype(np.uint8)*255, hand_kpts_2d, shift_augment, img_size=CROP_SIZE)
        
        img_cropped = merge_hands_cropped(img_top_cropped, img_bot_cropped, self.bg_img, shift_augment)
        # Color domain
        self.img_cropped_gray = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2GRAY)
        self.img_cropped_seg = seg_cropped.astype(np.uint8)*255
        self.img_cropped_edge = cv2.Canny(self.img_cropped_gray.astype(np.uint8), 25, 100).astype(np.float32)
        # Frequency domain
        self.img_cropped_gray_freq = img2freq(self.img_cropped_gray)
        self.img_cropped_seg_freq = img2freq(self.img_cropped_seg)
        self.img_cropped_edge_freq = img2freq(self.img_cropped_edge)
        self.img_input_2d = np.stack((self.img_cropped_gray_freq, self.img_cropped_edge_freq, self.img_cropped_seg_freq), 0)
        
    def set_display_images(self):
        if self.display_mode == DISPLAY_MODE_IMG:
            self.display_img_rgb = self.img_cropped_gray
            self.display_img_freq = self.img_cropped_gray_freq
            self.display_img_pred = self.img_pred
        elif self.display_mode == DISPLAY_MODE_SEG:
            self.display_img_rgb = self.img_cropped_seg
            self.display_img_freq = self.img_cropped_seg_freq
            self.display_img_pred = self.img_pred
        elif self.display_mode == DISPLAY_MODE_EDGE:
            self.display_img_rgb = self.img_cropped_edge
            self.display_img_freq = self.img_cropped_edge_freq
            self.display_img_pred = self.img_pred
            
    def model_2d_pred(self):
        """
        img_input_tensor = normalize_tensor(torch.from_numpy(self.img_input_2d), 128.0, 256.0).cuda()
        img_input_tensor = img_input_tensor.unsqueeze_(0)
        _, _, _, _, _, heatmaps_stage_final = self.model_2d(img_input_tensor)
        heatmaps_output_np = heatmaps_stage_final.cpu().data.numpy().transpose(0,2,3,1)[0]
        heatmaps_output_combined = np.max(heatmaps_output_np[:, :, 1:], axis=2)*255.0
        heatmaps_output_combined = cv2.resize(heatmaps_output_combined, (CROP_SIZE, CROP_SIZE))
        self.img_pred = heatmaps_output_combined
        """
        self.img_pred = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype = np.uint8)
        
            
    # Update functions
    def update_image(self):
        self.window.image_display_rgb = ImageTk.PhotoImage(Image.fromarray(self.display_img_rgb))
        self.window.canvas_rgb.itemconfig(self.window.canvas_img_rgb, image=self.window.image_display_rgb)
        
        self.window.image_display_r = ImageTk.PhotoImage(Image.fromarray(self.display_img_freq))
        self.window.canvas_freq.itemconfig(self.window.canvas_img_freq, image=self.window.image_display_freq)
        
            
if __name__ == '__main__':
    hand_app = interactive_hand_app()
    hand_app.run_app()