

class interactive_hand_app:
    def __init__(self):
        # prepare UI
        self.init_window()

    def init_window(self):
        # Create tkinter UI
        self.window = tk.Tk()
        self.window.title('Interactive Hand App')
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
        self.window.panel_image_display = None
        self.window.panel_freq_display = None
        self.window.panel_option_ver = None
        self.img_i = 0
        self.load_data_paths()
        self.init_image_panel()
        self.init_cursor()
        self.init_zoomin_panel()
        self.init_option_panel()
        self.update_image()

    def init_image_panel(self):
        # Clear existing panel if exists
        if self.window.panel_image_display is not None:
            self.window.panel_image_display.pack_forget()
            del self.window.panel_image_display
        self.window.panel_image_display = tk.PanedWindow(self.window.panel_top_hor)
        self.window.panel_image_display.pack(side=tk.LEFT)
        
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

        # Set up Canvas
        self.window.image_display = ImageTk.PhotoImage(Image.fromarray(self.img_display))
        self.window.canvas = tk.Canvas(self.window.panel_image_display, 
            width=self.img_display.shape[1], height=self.img_display.shape[0])
        self.window.canvas_img = self.window.canvas.create_image(0,0,image=self.window.image_display)
        self.window.canvas.config(scrollregion=self.window.canvas.bbox(tk.ALL))
        self.window.canvas.bind('<Button-1>', self.mouse_left_click_callback)
        self.window.canvas.bind('<Button-3>', self.mouse_right_click_callback)
        self.window.canvas.pack()

    def init_cursor(self):
        self.window.canvas.bind('<Motion>', self.update_cursor_motion)
        self.update_cursor_coord(self.img_display.shape[0]//2, self.img_display.shape[1]//2)

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

        self.bg_img = np.zeros((1000, 2000, 3), dtype=np.uint8)

        
