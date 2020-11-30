# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import traceback
import matplotlib as mpl
mpl.use("tkagg")
from background_aware_correlation_filter import BackgroundAwareCorrelationFilter as BACF
from utils.arg_parse import parse_args #引入参数
from image_process.feature import get_pyhog#引入取特征过程
from utils.get_sequence import get_sequence_info, load_image#引入读取图片序列过程
from utils.report import LogManger#用来做展示

#提取黄色掩膜
def separate_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    
    lower_hsv = np.array([11, 50, 60])  # 提取颜色的低值
    high_hsv = np.array([34, 255, 255])
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)  # 下面详细介绍
    
    
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    mask = cv2.dilate(mask, kernel3)
    
    mask=~mask
    return mask/255
def sceolor(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离

    lower_hsv = np.array([11, 130, 100])  # 提取颜色的低值
    high_hsv = np.array([34, 255, 255])
    mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)  # 下面详细介绍

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask=cv2.erode(mask,kernel)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    mask = cv2.dilate(mask, kernel3)
    mask[0:770,:]=0

    mask=~mask
    return mask/255
def MovementFind(mImageOri, mImageNew):#核心的滤波处理流程

    mImgGray=cv2.cvtColor(mImageOri,cv2.COLOR_BGR2GRAY)

    vpPointsOri=cv2.goodFeaturesToTrack(mImgGray,200, 0.01, 10)
    vpPointsOri=cv2.cornerSubPix(mImgGray,vpPointsOri,(10,10),(-1,-1),criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 20, 0.03))
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    vpPointsNew,vpStatus,err = cv2.calcOpticalFlowPyrLK(mImageOri, mImageNew, vpPointsOri,None,**lk_params)#光流跟踪
    OpticalFlowChecheck(vpPointsOri, vpPointsNew, vpStatus); #剔除无跟踪点
    # a=(vpPointsNew[0]*255).astype(np.int)
    # b=(vpPointsNew[1]*255).astype(np.int)
    # good_new = vpPointsNew[vpStatus==1]
    #
    # for i, new in enumerate(good_new):
    #     a, b = new.ravel()
    #
    #     mImageNew = cv2.circle(mImageNew, (a, b), 5, (0, 0, 255), 1)
    # cv2.imshow('frame', mImageNew)
    vpBackPoints=[[],[]]
    vpBackPoints[0] = vpPointsOri
    vpBackPoints[1] = vpPointsNew
    cv2.findFundamentalMat(vpBackPoints[0], vpBackPoints[1],cv2.FM_RANSAC)#计算本体矩阵，剔除误匹配
    mTrans,mask = cv2.findHomography(vpBackPoints[0], vpBackPoints[1],cv2.RANSAC,3)
    mImgGray1=cv2.warpPerspective(mImgGray,mTrans,(mImgGray.shape[1],mImgGray.shape[0]))#应用仿射变换，将原图像转换到新图像空间
    mImgGrayNew = cv2.cvtColor(mImageNew, cv2.COLOR_BGR2GRAY)
    mImageShow=cv2.absdiff(mImgGray1, mImgGrayNew)
    return mImageShow

def OpticalFlowChecheck(vpOriPoints, vpNewPoints, vpStatus):
    for i in range(len(vpStatus)):
        if (vpStatus[i]!=0):
            vpFinePoints=[[],[]]
            vpFinePoints[0].append(vpOriPoints[i])
            vpFinePoints[1].append(vpNewPoints[i])
    vpOriPoints = vpFinePoints[0]
    vpNewPoints = vpFinePoints[1]
    return 0
# def smallTargetFilte(mImg,thresh):#小区域归并
#     movContours, hierarchy = cv2.findContours(mImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for i in range(len(movContours)):
#         mImg=cv2.drawContours(mImg, movContours,i,)
#     return mShow
def drawContours(mImg, differ):  # 画框
    movContours, hierarchy = cv2.findContours(differ, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    # copyto 的位置
    frame_contours = mImg.copy()
    k = 0
    if len(movContours) == 0:
        return 0, mImg, (0, 0, 0, 0)
    for i in range(0, len(movContours)):
        x, y, w, h = cv2.boundingRect(movContours[i])
        x = max(0, x - 13)
        y = max(0, y - 13)
        w = min(mImg.shape[1], w + 25)
        h = min(mImg.shape[0], h + 25)
        if heart(x, y, w, h):
            k += 1
            roi = (x, y, w, h)
    if k != 1:
        ret = 0
        return ret, mImg, 0
    else:
        ret = 1
        return ret, mImg, roi



def heart(x,y,w,h):
    #中心点
    x_h=x+w/2
    y_h=y+h/2
    if min(abs(x_h),abs(1920-x_h))<30:
        return False
    elif abs(y_h)<30:
	return False
    elif abs(1080-y_h)<50:
        return False
    elif w/h>3 or w/h<0.33:
        return False
    elif w*h>1920*11:
        return False
    else:return True
def heartg(x,y,w,h):
    #中心点
    x_h=x+w/2
    y_h=y+h/2
    if min(abs(x_h),abs(1920-x_h))<30:
        return False
    elif abs(y_h)<10:
	return False
    elif abs(1080-y_h)<50:
        return False
    elif w/h>3 or w/h<0.33:
        return False
    elif w*h>1920*11:
        return False
    else:return True
#主程序开始
cap = cv2.VideoCapture('/home/hedwig/Desktop/bacf_python-master/name2.mp4')
parser = parse_args()
params = parser.parse_args()
for param in dir(params):
    if not param.startswith('_'):
        print('{0} : {1}'.format(param, getattr(params, param)))
flag=0
while(1):
    while(1):
        ret, frame = cap.read()  # 读取图片
        cv2.waitKey(1)
        mImgPre = frame

        ret, mImg = cap.read()  # 读取图片
	timer = cv2.getTickCount()

        mShow = MovementFind(mImgPre, mImg)  # 单应矩阵处理过的图像
        mImgPre_s = cv2.resize(mImgPre, (480, 270))
        mImg_s = cv2.resize(mImg, (480, 270))
        mask_yw1 = separate_color(mImgPre_s)
        mask_yw2 = separate_color(mImg_s)
	mask=mask_yw1+mask_yw2
    	mask = cv2.resize(mask, (1920, 1080))
        mShow = mask * mShow    
        mShow = mShow.astype('uint8')
        size = frame.shape
        height = size[0]
        width = size[1]
        mShow_median = cv2.medianBlur(mShow, 5)  # 此时传入的应该是灰度图
        mShow_median = mShow_median
        _, mShow = cv2.threshold(mShow_median, 60, 255, cv2.THRESH_BINARY)
        # _, mShow = cv2.threshold(mShow, 60, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mShow1 = cv2.dilate(mShow, kernel15)
     	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        ret1, frame, bbox = drawContours(mImg, mShow1)
        if ret1 == 1:
            break
        if ret1==0 and flag==1:
            cv2.putText(mImg, 'bacf' + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
            cv2.putText( mImg, "No target has been detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
	    cv2.putText(mImg, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            cv2.imshow('Tracking', mImg)

    #ok = tracker.init(frame, bbox)#需要进行引用的函数，用bbox进行跟踪的第一帧，return状态
    rect_pos=bbox

    bacf = BACF(get_pyhog, admm_lambda=params.admm_lambda,
                cell_selection_thresh=params.cell_selection_thresh,
                dim_feature=params.dim_feature,
                filter_max_area=params.filter_max_area,
                feature_ratio=params.feature_ratio,
                interpolate_response=params.interpolate_response,
                learning_rate=params.learning_rate,
                search_area_scale=params.search_area_scale,
                reg_window_power=params.reg_window_power,
                n_scales=params.n_scales,
                newton_iterations=params.newton_iterations,
                output_sigma_factor=params.output_sigma_factor,
                refinement_iterations=params.refinement_iterations,
                reg_lambda=params.reg_lambda,
                reg_window_edge=params.reg_window_edge,
                reg_window_min=params.reg_window_min,
                scale_step=params.scale_step,
                search_area_shape=params.search_area_shape,
                save_without_showing=params.save_without_showing,
                debug=params.debug,
                visualization=params.visualization,
                is_redetection=params.is_redetection,
                redetection_search_area_scale=params.redetection_search_area_scale,
                is_entire_redection=params.is_entire_redection,
                psr_threshold=params.psr_threshold)

    patch = bacf.init(frame, rect_pos)
    flag=0
    while True:
        ret, frame = cap.read()
        timer = cv2.getTickCount()

        # Update tracker
        #ok, bbox = tracker.update(frame)#需要进行引用的函数，通过读取后续图片更新bbox，和状态
        patch, response = bacf.track(frame)
        bacf.train(frame)
        _, rect_pos, _, _ = bacf.get_state()
        bbox=rect_pos
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        mask=sceolor(frame)
        # Draw bounding box
        if heart(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])):
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            flag=1
            break

            # Display tracker type on frame
        cv2.putText(frame, 'bacf' + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display result
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
