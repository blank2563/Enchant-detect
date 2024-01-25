import cv2 as cv
import numpy as np
from imutils.object_detection import non_max_suppression as nms
import os
import pandas as pd

def dupe(ench):
    return all(i[2] == ench[0][2] for i in ench)

cwd = os.path.dirname(os.path.abspath(__file__))
test_list = os.listdir('test-img')
enc_list = os.listdir('enchant')

threshold = 0.8

TopSig = []
TopLock = []
BotSig = []
BotLock = []
Top1 = []
Top2 = []
Top3 = []
Top4 = []
Top5 = []
Bot1 = []
Bot2 = []
Bot3 = []
Bot4 = []
Bot5 = []

df = pd.DataFrame()


for img_name in test_list:
    img_dir = os.path.join(cwd, 'test-img', img_name)
    
    img_large = cv.imread(img_dir, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img_large, (0,0), fx=0.5, fy=0.5) 
    
    img_large_RGB = cv.imread(img_dir)
    img_RGB = cv.resize(img_large_RGB, (0,0), fx=0.5, fy=0.5)

    H, W = img.shape[:2]

    top_lock = 'Unlocked'
    bot_lock = 'Unlocked'
    
    top_enc = []
    bot_enc = []
    top_sig = 'Empty'
    bot_sig = 'Empty'
    
    for enc_name in enc_list:
        template_dir = os.path.join(cwd, 'enchant', enc_name)
        template_large = cv.imread(template_dir, cv.IMREAD_GRAYSCALE)
        template_large_RGB = cv.imread(template_dir)
        template = cv.resize(template_large, (0,0), fx=0.5, fy=0.5) 
        h, w = template.shape[:2]

        res = cv.matchTemplate(img,template,cv.TM_CCOEFF_NORMED)

        x_res, y_res = np.where( res >= threshold)
        boxes = []
        for (x, y) in zip(x_res, y_res): 
            boxes.append((x, y, x+w, y+h))

        res_box = nms(np.array(boxes))
        for y1, x1, y2, x2 in res_box:
            cv.rectangle(img_RGB, (x1, y1), (x2, y2), (0, 255, 0), 3) 
            if x1 < 0.1*W:
                if y1 < 0.35*H:
                    top_sig = enc_name.removesuffix('.png')
                else:
                    bot_sig = enc_name.removesuffix('.png')
            elif y1 < 0.35*H:
                top_enc.append((x1, y1, enc_name.removesuffix('.png')))
            else:
                bot_enc.append((x1, y1, enc_name.removesuffix('.png')))
            
    top_sorted = sorted(top_enc, key = lambda x : x[0])
    
    while len(top_sorted) < 5:
        top_sorted.append((0,0,''))
    if dupe(top_sorted):
        top_lock = 'Locked'
    bot_sorted = sorted(bot_enc, key = lambda x : x[0])
    if dupe(bot_sorted):
        bot_lock = 'Locked'
    while len(bot_sorted) < 5:
        bot_sorted.append((0,0,''))
        
    TopSig.append(top_sig)
    TopLock.append(top_lock)
    BotSig.append(bot_sig)
    BotLock.append(bot_lock)
    Top1.append(top_sorted[0][2])
    Top2.append(top_sorted[1][2])
    Top3.append(top_sorted[2][2])
    Top4.append(top_sorted[3][2])
    Top5.append(top_sorted[4][2])
    Bot1.append(bot_sorted[0][2])
    Bot2.append(bot_sorted[1][2])
    Bot3.append(bot_sorted[2][2])
    Bot4.append(bot_sorted[3][2])
    Bot5.append(bot_sorted[4][2])

    #cv.imshow('match', img_RGB)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

df['TopSig'] = TopSig
df['TopLock'] = TopLock
df['BotSig'] = BotSig
df['BotLock'] = BotLock
df['Top1'] = Top1
df['Top2'] = Top2
df['Top3'] = Top3
df['Top4'] = Top4
df['Top5'] = Top5
df['Bot1'] = Bot1
df['Bot2'] = Bot2
df['Bot3'] = Bot3
df['Bot4'] = Bot4
df['Bot5'] = Bot5

df.to_excel(os.path.join(cwd, 'data.xlsx'), index=False)