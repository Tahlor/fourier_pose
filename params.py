MODEL_NAME_CPM = 'CPM'

MODEL_SAVE_DIR_PATH = 'models_saved'

VALID_HAND_SEG_TH = 5000
VALID_ENERGY_TH_RATIO = 0.5
HAND_2D_CROP_SIZE = 224
HAND_2D_CROP_STRIDE = 8
IMG_H = 288
IMG_W = 512

colors = [
[0, 255, 0],		#0
[0, 223, 0],		#1
[0, 191, 0],		#2
[0, 159, 0],		#3

[159, 255, 0],		#4
[159, 223, 0],		#5
[159, 191, 0],		#6
[159, 159, 0],		#7

[255, 0, 0], 		#8
[223, 0, 0],		#9
[191, 0, 0],	 	#10
[159, 0, 0], 		#11

[255, 0, 255],	 	#12
[255, 0, 223],		#13
[255, 0, 191],	 	#14
[255, 0, 159],		#15

[0, 0, 255], 		#16
[0, 0, 223], 		#17
[0, 0, 191], 		#18
[0, 0, 159],		#19
]
 
limbSeq_ganhand = [
[0, 1],		#Thumb1
[1, 2],		#Thumb2
[2, 3],		#Thumb3
[3, 4],		#Thumb4

[0, 5],		#index1
[5, 6],		#index2
[6, 7],		#index3
[7, 8],		#index4

[0, 9],		#middle1
[9, 10],	#middle2
[10 ,11],	#middle3
[11, 12],	#middle4

[0, 13],	#ring1
[13, 14],	#ring2
[14, 15],	#ring3
[15, 16],	#ring4

[0, 17],	#pinky1
[17, 18],	#pinky2
[18, 19],	#pinky3
[19, 20]	#pinky4
]