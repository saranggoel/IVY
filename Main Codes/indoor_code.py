import copy
import math
# import sys
# import ogl_viewer.tracking_viewer as gl
import pyzed.sl as sl
# import copy
import time
# import keyboard
import networkx as nx
import matplotlib.pyplot as plt
import requests
#import pyttsx3  # for text to speech
import os
import csv
from datetime import datetime
from threading import Thread,Lock

import jetson.inference
import jetson.utils
import Jetson.GPIO as GPIO

import numpy as np
import statistics
import socket
import json
from shutil import copyfile

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
#gpio pins
left=13
center=33
right=29

GPIO.setup(left, GPIO.OUT) #left
GPIO.setup(center, GPIO.OUT) #center
GPIO.setup(right, GPIO.OUT) #right

#initialize
GPIO.output(left, GPIO.LOW)
GPIO.output(center, GPIO.LOW)
GPIO.output(right, GPIO.LOW)

def vibration(l,c,r): # 0 for off and 1 for on
    if (l == 1):
        l_state = GPIO.HIGH
    else:
        l_state = GPIO.LOW

    if (c == 1):
        c_state = GPIO.HIGH
    else:
        c_state = GPIO.LOW

    if ( r== 1):
        r_state = GPIO.HIGH
    else:
        r_state = GPIO.LOW

    GPIO.output(left, l_state)
    GPIO.output(center, c_state)
    GPIO.output(right, r_state)


def thread_position():
    global flag_map,flag_dest,text_list,key_press_dest_index,time2,points,edges
    while True:
        # if((time.time()-time2)%1<0.05): print(flag_map, trans_true, trans2, thres)

        if (key_press.startswith('d') == True):
            key_press_dest_index = key_press.replace('d', "")
            try:
                destination()  # set destination path
            except:
                print('destination input bad-retry')
                text_list = 'destination input bad-retry'
                update_list_audio(text_list)

        if (flag_dest == 1 and flag_obj == 0): dest_mode()  # in destination mode
        #mapping
        if (key_press == 'start map' and flag_map == 0): start_map()  # start mapping; saving first map point (generally, also the origin)
        if (flag_map == 1 and (abs(trans_true - trans2) >= thres)): map2()  # mapping second point onwards
        if (key_press == 'pause map' and flag_map == 1): pause_map()
        if (key_press == 'continue map' and flag_map == 2): continue_map()
        if (key_press == 'stop map' and (flag_map == 1 or flag_map == 2)): stop_map()  # stop mapping


        if (flag_dest == 1 and math_dist(dest_end2, (x_true, y_true)) <= 0.75 * thres): reached_destination()  # reached destination printout




        if (time.time()-time2>5 and flag_map == 0 and abs(trans_true - trans2) >= thres and flag_dest == 0 and flag_cal==0): print_update2()  # print updates
        if (key_press == 'end destination' and flag_dest == 1):
            flag_dest = 0
            print('ending destination - select new destination')
            text_list = 'ending destination - select new destination'
            update_list_audio(text_list)
        if (key_press.startswith('save destination') == True and flag_map == 1): save_destination()
        if (key_press.startswith('save location') == True): save_room()
        if (key_press.startswith('where am i') == True): where_am_i()
        if (key_press.startswith('select home') == True): select_home()


def thread_detection():
    global object_detected_list_last, object_detected_list, object_name_list, \
    lock,new_data,flag_dest,time1,time2,flag_obj, \
    zed,image_mat,depth_mat,image_np_global,depth_np_global,img,detections
    print("entered thread_detection")
    # net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.2)
    # net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.2)
    # camera = jetson.utils.videoSource("csi://0")  # '/dev/video0' for V4L2
    # display = jetson.utils.videoOutput("display://0")  # 'my_video.mp4' for file
    count_none = 0
    time31 = 0
    time32 = 0
    while True:
        if (flag_dest == 1 and new_data == False):
            zed.retrieve_image(image_mat, sl.VIEW.LEFT, resolution=image_size)
            zed.retrieve_measure(depth_mat, sl.MEASURE.XYZRGBA, resolution=image_size)
            image_np_global = load_image_into_numpy_array(image_mat)
            depth_np_global = load_depth_into_numpy_array(depth_mat)
            img = jetson.utils.cudaFromNumpy(image_np_global)
            detections = net.Detect(img)

            # display.Render(img)
            # display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
            new_data = True

        if(flag_dest==1 and new_data==True):
            time21 = round(time.time() - time2, 2)

            x_true_tf=copy.deepcopy(x_true)
            y_true_tf = copy.deepcopy(y_true)
            dest_next_tf = copy.deepcopy(dest_next)
            yaw_z_tf = copy.deepcopy(yaw_z)
            # time22 = round(time.time() - time2, 2)
            # time23 = round(time.time() - time2, 2)

            if (len(detections) != 0):
                # try:
                object_detected_list = []
                distance_list=[]
                count_invalid=0
                for i in range(len(detections)):
                    xmin = int(detections[i].Left) / width
                    xmax = int(detections[i].Right) / width
                    ymin = int(detections[i].Top) / height
                    ymax = int(detections[i].Bottom) / height

                    if(xmax-xmin<=0.1 or ymax-ymin<=0.13): continue

                    x_center = int(xmin * width + (xmax - xmin) * width * 0.5)
                    y_center = int(ymin * height + (ymax - ymin) * height * 0.5)
                    x_vect = []
                    y_vect = []
                    z_vect = []

                    min_y_r = y_center - 1
                    min_x_r = x_center - 1
                    max_y_r = y_center + 1
                    max_x_r = x_center + 1

                    if min_y_r < 0: min_y_r = 0
                    if min_x_r < 0: min_x_r = 0
                    if max_y_r > height: max_y_r = height
                    if max_x_r > width: max_x_r = width

                    for j_ in range(min_y_r, max_y_r):
                        for i_ in range(min_x_r, max_x_r):
                            z = depth_np_global[j_, i_, 2]
                            if not np.isnan(z) and not np.isinf(z):
                                x_vect.append(depth_np_global[j_, i_, 0])
                                y_vect.append(depth_np_global[j_, i_, 1])
                                z_vect.append(z)

                    if len(x_vect) > 0:
                        x = round(statistics.median(x_vect),2)  # relative position (left ornright) of center of bounding box wrt camera line of sight
                        y = round(statistics.median(y_vect), 2)  # distance of object from the camera
                        z = round(statistics.median(z_vect), 2)  # elevation of object center above or below

                        z_ver = 'level'
                        if (z > 0):
                            z_ver = 'above'
                        elif (z < 0):
                            z_ver = 'below'

                        distance = round(math.sqrt(x * x + y * y),1)

                        # print('xyz', i, x, y, z, distance,time1)

                        x_rel_tf = round(x * math.cos(yaw_z_tf) - y * math.sin(yaw_z_tf), 2)  # correction wrt to line of path
                        y_rel_tf = round(x * math.sin(yaw_z_tf) + y * math.cos(yaw_z_tf), 2)
                        x_obj_tf = round(x_true_tf + x_rel_tf, 2)  # absolute coordinates
                        y_obj_tf = round(y_true_tf + y_rel_tf, 2)
                        pd_obj_tf = -999

                        pd_obj_tf, pd_side_tf,pd_location_tf = dist_perp((x_true_tf, y_true_tf), dest_next_tf, (x_obj_tf, y_obj_tf))

                        if (distance <= 5):

                            object_name = object_name_list[detections[i].ClassID]
                            # print(object_name,pd_obj_tf, pd_side_tf, pd_location_tf, time1)

                            bbw = xmax - xmin
                            bbh = ymax - ymin
                            ow = round(bbw * 2 * math.tan(math.pi / 4) * y,
                                       1)  # zed has 90 degree horizontal coverage, 45 left, 45 right
                            oh = round(bbh * 2 * math.tan(math.pi / 6) * y,
                                       1)  # zed has 60 degree vertical coverage, 30 above, 30 below (or 15 above, 45 below)

                            pw = 1  # path width = 2pw, pw in feet to each side of line of path
                            # object in line of sight?? +/- path_width in ft of line of sight
                            oip = 'na'  # oip=object in path
                            oside = 'na'

                            # in absolute coordinate system:
                            if(pd_location_tf=='included' and ow>=0.75): #objects bigger than 9" wide
                                if (pd_side_tf=='left' and pd_obj_tf-ow/2 >pw):
                                    pass
                                    # oip = object_name + str(int(y_obj_tf * 5) / 5) + ' feet away not in path'
                                    # oside = 'on left'
                                elif (pd_side_tf=='right' and pd_obj_tf-ow/2 >pw):
                                    pass
                                    # oip = object_name + str(int(y_obj_tf * 5) / 5) + ' feet away not in path'
                                    # oside = 'on right'
                                elif (pd_side_tf=='left' and pd_obj_tf-ow/2 <pw-0.25): #0.25' threshold to ignore moving
                                    # oip = object_name + str(distance) + ' feet away:'
                                    # oside = 'move right by '+str(round(pw-(pd_obj_tf-ow/2),1))+' feet'
                                    oip = object_name, str(distance)
                                    oside = 'move right',str(round(pw - (pd_obj_tf - ow / 2), 1))
                                elif (pd_side_tf=='right' and pd_obj_tf-ow/2 <pw-0.25):
                                    # oip = object_name + str(distance) + ' feet away:'
                                    # oside = 'move left by '+str(round(pw-(pd_obj_tf-ow/2),1))+' feet'
                                    oip = object_name,str(distance)
                                    oside = 'move left',str(round(pw-(pd_obj_tf-ow/2),1))
                                elif (pd_side_tf=='center'):
                                    # oip = object_name + str(distance) + ' feet away:'
                                    # oside = 'move left or right by '+str(round(pw-(pd_obj_tf-ow/2),1))+' feet'
                                    oip = object_name,str(distance)
                                    oside = 'move either '+str(round(pw-(pd_obj_tf-ow/2),1))

                                if(oip!='na'):
                                    temp1 = oip, oside
                                    # temp1 = str(tempx)
                                    text_list=str(temp1)
                                    # update_list_audio(text_list)
                                    object_detected_list.append(text_list)
                                    distance_list.append(distance)

            time23 = round(time.time() - time2, 2)
            # print('thread',time23-time21,object_detected_list)

            if (len(object_detected_list)!=0):
                min_value = min(distance_list)
                min_index = distance_list.index(min_value)
                text_list = object_detected_list[min_index]
                print(time1, count_none, flag_obj, round(time23 - time21, 2), text_list)
                time31=time.time()-time2
                if(time31-time32>=1):
                    update_list_audio(text_list)
                    time32=time31
                count_none = 0
                flag_obj=1
            else:
                text_list=''
                count_none+=1
                if(count_none>=2): flag_obj=0

            # print(time1, count_none, flag_obj, round(time23 - time21, 2), text_list)
            object_detected_list_last = object_detected_list
            new_data = False

def coco_classes():
    COCO_CLASSES_LIST = [
        'unlabeled',
        'person',
        'object', #bicycle',
        'object', #car',
        'object', #motorcycle',
        'object', #airplane',
        'object', #bus',
        'object', #train',
        'object', #truck',
        'object', #boat',
        'object', #traffic light',
        'object', #fire hydrant',
        'object', #street sign',
        'object', #stop sign',
        'object', #parking meter',
        'bench',
        'bird',
        'cat',
        'dog',
        'object', #horse',
        'object', #sheep',
        'object', #cow',
        'object', #elephant',
        'object', #bear',
        'object', #zebra',
        'object', #giraffe',
        'hat',
        'backpack',
        'umbrella',
        'shoe',
        'eye glasses',
        'handbag',
        'tie',
        'suitcase',
        'frisbee',
        'skis',
        'snowboard',
        'sports ball',
        'kite',
        'baseball bat',
        'baseball glove',
        'skateboard',
        'surfboard',
        'tennis racket',
        'bottle',
        'plate',
        'wine glass',
        'cup',
        'fork',
        'knife',
        'spoon',
        'bowl',
        'banana',
        'apple',
        'sandwich',
        'orange',
        'broccoli',
        'carrot',
        'hot dog',
        'pizza',
        'donut',
        'cake',
        'chair',
        'couch',
        'potted plant',
        'bed',
        'mirror',
        'dining table',
        'window',
        'desk',
        'toilet',
        'door',
        'tv',
        'laptop',
        'mouse',
        'remote',
        'keyboard',
        'cell phone',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator',
        'blender',
        'book',
        'clock',
        'vase',
        'scissors',
        'teddy bear',
        'hair drier',
        'toothbrush',
    ]
    return COCO_CLASSES_LIST

def load_image_into_numpy_array(image):
    ar = image.get_data()
    ar = ar[:, :, 0:3]
    (im_height, im_width, channels) = image.get_data().shape
    return np.array(ar).reshape((im_height, im_width, 3)).astype(np.uint8)

def load_depth_into_numpy_array(depth):
    ar = depth.get_data()
    ar = ar[:, :, 0:4]
    (im_height, im_width, channels) = depth.get_data().shape
    return np.array(ar).reshape((im_height, im_width, channels)).astype(np.float32)

def math_dist(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    dist = math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return dist

def audio_thread():
    global flag, list_last, list_audio, stop, list_last_2,stop,key_press,time1,list_last2,list_last2_2,list_audio2
    list_last_2 = ""
    list_last = ""
    list_last2_2 = ""
    list_last2 = ""
    print("entered thread")
    text_list="entered thread"
    update_list_audio2(text_list)
    count=0
    count2=0
    while stop == True:


        if (flag == 0):
            flag = 1

            #important audio under post "a" [need to buffer in flutter]
            if(len(list_audio)>0):
                list_last = list_audio[len(list_audio) - 1]
            if (list_last != list_last_2):
                count=0
                # print('not equal',list_last,list_last_2)
                pos = {'a': str(list_last)}
                response = requests.post("http://saranggoel.pythonanywhere.com/", json=pos)
            else:
                if(count==2):
                    temp = ''
                    pos = {'a': str(temp)}
                    response = requests.post("http://saranggoel.pythonanywhere.com/", json=pos)

                count = count + 1
            #         print('equal=0', count, list_last, list_last_2,temp)
            #     if (count == 10 and flag_cal!=1):
            #         temp = int(x_true),int(y_true),int(yaw_zd)
            #         pos = {'a': str(temp)}
            #         response = requests.post("http://saranggoel.pythonanywhere.com/", json=pos)
            #         print('equal=10', count, list_last, list_last_2,temp)
            #         count=-5
            #
            list_last_2 = list_last


            # not so important audion under post "c" [no buffering]
            if(len(list_audio2)>0):
                list_last2 = list_audio2[len(list_audio2) - 1]
            if (list_last2 != list_last2_2):
                count2=0
                # print('not equal2',list_last2,list_last2_2)
                pos3 = {'c': str(list_last2)}
                response2 = requests.post("http://saranggoel.pythonanywhere.com/", json=pos3)
            else:

                if(count2==0 and flag_cal!=1):
                    temp2 = ''
                    pos3 = {'c': str(temp2)}
                    response3 = requests.post("http://saranggoel.pythonanywhere.com/", json=pos3)
                    # print('equal2=0', count2, list_last2, list_last2_2,temp2)
                if (count2 == 10 and flag_cal!=1):
                    temp2 = int(x_true),int(y_true),int(yaw_zd)
                    pos3 = {'c': str(temp2)}
                    response3 = requests.post("http://saranggoel.pythonanywhere.com/", json=pos3)
                    # print('equal2=10', count2, list_last2, list_last2_2,temp2)
                    count2=-5
                count2=count2+1
            list_last2_2 = list_last2

            response2 = requests.get("http://saranggoel.pythonanywhere.com/")
            dict_response2 = json.loads(response2.text)
            # request_keys2 = list(dict_response2.keys())
            # print(request_keys2)

            key_press = dict_response2.get("b")
            key_press=key_press.lower()
            if (key_press != ""):
                print(key_press)
                text_list = 'key pressed is ' + str(key_press)
                update_list_audio2(text_list)
                pos2 = {'b': ""}
                requests.post("http://saranggoel.pythonanywhere.com/", json=pos2)

            flag = 0

def get_beta(dest_next, dest_curr):
    dx = dest_next[0] - dest_curr[0]
    dy = dest_next[1] - dest_curr[1]
    if (0 <= dy <= 0.01 and dx > 0):
        angle = 90.0
    elif (-0.01 <= dy <= 0 and dx < 0):
        angle = -89.9
    else:
        angle = round(180 / math.pi * math.atan((dest_next[0] - dest_curr[0]) / (dest_next[1] - dest_curr[1])), 1)
    if (dx >= 0 and dy >= 0): beta = angle  # Q1
    if (dx < 0 and dy >= 0): beta = angle  # Q2
    if (dx < 0 and dy < 0): beta = angle - 180.0  # Q3
    if (dx >= 0 and dy < 0): beta = 180.0 + angle  # Q4

    return beta

def dist_perp(dest_curr, dest_next, curr_pos):
    x1 = dest_curr[0]
    y1 = dest_curr[1]
    x2 = dest_next[0]
    y2 = dest_next[1]
    x3 = curr_pos[0]
    y3 = curr_pos[1]
    side = 'na'
    location='na'

    try:
        beta12=get_beta(dest_next,dest_curr)
        beta13=get_beta(curr_pos,dest_curr)

        if(beta13*beta12>=0):
            if (beta13 < beta12):
                side = 'left'
            elif (beta13 > beta12):
                side = 'right'
            else:
                side = 'center'

        if (beta13 * beta12 < 0):
            if(abs(beta13)<=90 and abs(beta12)<=90):
                if (beta13 < beta12):
                    side = 'left'
                if (beta13 > beta12):
                    side = 'right'
            if (abs(beta13) > 90 and abs(beta12) > 90):
                if (beta13 < beta12):
                    side = 'right'
                if (beta13 > beta12):
                    side = 'left'

    except:
        pass

    if (abs(x2 - x1) >= 0.01):
        m = (y2 - y1) / (x2 - x1)
        a = m
        b = -1
        c = y1 - m * x1
        d = round(abs(a * x3 + b * y3 + c) / math.sqrt(a * a + b * b), 2)
    else:
        d = round(abs(x3 - x1), 2)
    # +-3" deadband for center - to prevent switching between left and right
    if(d<=0.25): side = 'center'

    try:  # to get coordinate of intersection (x4,y4) of perpendicular from point (x3,y3) on line segment (x1,y1):(x2,y2)
        t = ((x3 - x1) * (x2 - x1) + (y3 - y1) * (y2 - y1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
        x4 = x1 + t * (x2 - x1)
        y4 = y1 + t * (y2 - y1)
        perp_inter = (x4, y4)
    except:
        pass

    # if(x3-x1<=x2-x1 and y3-y1<=y2-y1):
    #     location='included'
    # else:
    #     location='excluded'


    if(math_dist((x1,y1),(x4,y4)) + math_dist((x2,y2),(x4,y4)) - math_dist((x1,y1),(x2,y2))<=0.2):
        location='included'
    else:
        location='excluded'

    return d, side,location

def add_edge_to_graph(G, e1, e2):
    G.add_edge(e1, e2)

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians

def getroom(x, y):


    for j in list_rooms_coor:  # j is each quad...
        n = list_rooms_coor.index(j)  # n is 0,1,..
        if (j[0] <= x < j[2] and j[1] <= y < j[3]):
            room_name = list_rooms_names[n]
            break
        else:
            room_name = 'hallway'

            # print(n, j, list_rooms_coor[n], list_rooms_names[n])  # list_rooms_coor[n] is each quad,
    return room_name

def set_default():
    global thread, thread_server,list_audio, flag, list_last, text_list, stop, trans2, trans_true, theta, points, edges, time2, flag_map, flag_cal, thres, thres_print, \
        thres_angle, thres_angle_print, flag, flag_dest, flag_temp,flag_server,flag_object,key_press,temp1_last,object_name_list,object_detected_list,object_detected_list_last,x_true,y_true, \
        temp_true,new_data,time_update_last,text1_last,time_print_last,text_print_last,time2,flag_obj,time1,time_update_last2, \
        text1_last2,list_audio2,list_last_2,list_last2,list_last2_2,home

    stop = True
    flag_obj = 0
    time_update_last=0
    time_update_last2 = 0
    time2=0
    time1=0
    text1_last=''
    text1_last2 = ''
    time_print_last=0
    text_print_last=''
    new_data=False
    object_name_list=coco_classes()
    home='home1'
    list_audio = ['select home as']
    list_audio2 = ['test 2']
    flag = 0
    list_last = 0
    list_last_2 = 0
    list_last2 = 0
    list_last2_2 = 0
    text_list = ''
    temp_true=''

    trans2 = -2.0
    trans_true = 0.0
    theta = 0.0
    points = []
    edges = []
    flag_map = 0
    flag_cal = 0
    thres = 0.5 #1
    thres_print = 3.0
    thres_angle = 5.0
    thres_angle_print = 20
    flag_dest = 0
    flag_temp = 0
    flag_server=0
    flag_object=0
    key_press=''
    temp1_last=''
    object_detected_list=[]
    object_detected_list_last=[]
    x_true=0
    y_true=0

def house_map():
    global points, edges,list_rooms_names,list_rooms_coor,list_dest_names,list_dest_coor,list_dest_yawzd
    points=[]
    edges=[]

    # Enable this "working" code once home data is final

    list_rooms_names = []
    list_rooms_coor = []
    list_dest_names = []
    list_dest_coor = []
    list_dest_yawzd = []

    filePath1 = '/Users/rakes/Documents/sf2021-2022/' + home + '/points_' + home + '.csv'
    filePath2 = '/Users/rakes/Documents/sf2021-2022/' + home + '/edges_' + home + '.csv'
    filePath3 = '/Users/rakes/Documents/sf2021-2022/' + home + '/destinations_' + home + '.csv'
    filePath4 = '/Users/rakes/Documents/sf2021-2022/' + home + '/rooms_' + home + '.csv'

    with open(filePath1) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            temp = (float(row[0]), float(row[1]))
            points.append((temp))

    with open(filePath2) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            temp = (int(row[0]), int(row[1]))
            edges.append((temp))

    with open(filePath4) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            temp = row[0].lower()
            if (list_rooms_names.count(temp) == 0):
                list_rooms_names.append(temp)

    for room in list_rooms_names:
        with open(filePath4) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                if (row[0].lower() == room.lower() and row[1].lower().startswith('bottom') == True):
                    bottom_x = float(row[2])
                    bottom_y = float(row[3])
                if (row[0].lower() == room.lower() and row[1].lower().startswith('top') == True):
                    top_x = float(row[2])
                    top_y = float(row[3])
            temp = [bottom_x, bottom_y, top_x, top_y]
            list_rooms_coor.append(temp)

    for room in list_rooms_names:
        with open(filePath3) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                temp = row[1].lower()
                if (temp == room.lower()):
                    list_dest_names.append(row[0].lower())
                    temp2 = (float(row[2]), float(row[3]))
                    list_dest_coor.append(temp2)
                    list_dest_yawzd.append(float(row[4]))

    # end of "working" code

    # map with area saved 090421 and cleaned, has extra unused points that were not useful
    # when cleaning map, only add or remove edges (new pairing of indexes). never remove undesired points.
    # points = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12),
    #           (0, 13), (0, 14), (0, 15), (0, 16), (0, 17), (0, 18), (0, 19), (0, 20), (0, 21), (0, 22), (0, 23),
    #           (0, 24), (0, 25), (0, 26), (0, 27), (0, 28), (0, 29), (0, 30), (0, 31), (0, 32), (0, 33), (0, 34),
    #           (0, 35), (0, 36), (0, 37), (0, 38), (0, 39), (0, 40), (0, 41), (0, 42), (0, 43), (0, 44), (0, 45),
    #           (0, 46), (0, 47), (0, 48), (0, 49), (0, 50), (0, 51), (0, 52), (0, 53), (0, 54), (0, 55), (1, 8), (2, 8),
    #           (3, 8), (4, 8), (5, 8), (6, 9), (7, 10), (8, 11), (6, 7), (7, 6), (8, 5), (9, 4), (-1, 17), (-2, 17),
    #           (-3, 17), (-4, 17), (-5, 17), (-6, 17), (-7, 17), (-7, 16), (-7, 15), (-1, 23), (-2, 23), (-3, 23),
    #           (-4, 23), (-5, 23), (-6, 23), (-7, 23), (-8, 23), (-9, 23), (-10, 23), (-11, 23), (-12, 23), (-13, 23),
    #           (-14, 23), (-15, 23), (-16, 23), (-17, 23), (-18, 23), (-19, 23), (-19, 22), (-19, 21), (-19, 20),
    #           (-19, 19), (-19, 18), (-8, 24), (-8, 25), (-8, 26), (-8, 27), (-8, 28), (-8, 29), (-8, 30), (-9, 30),
    #           (-10, 30), (-11, 30), (-12, 30), (1, 31), (2, 31), (3, 31), (4, 31), (5, 31), (6, 31), (6, 32), (6, 33),
    #           (6, 34), (5, 34), (4, 34), (3, 34), (2, 34), (1, 34), (-1, 35), (-2, 35), (-3, 35), (-4, 35), (-5, 35),
    #           (-6, 35), (-6, 36), (-6, 37), (-6, 38), (-6, 39), (-6, 40), (-6, 41), (-6, 42), (-6, 43), (-6, 44),
    #           (-6, 45), (-7, 45), (-8, 45), (-9, 45), (-9, 46), (-9, 47), (-9, 48), (-9, 49), (-9, 50), (-10, 45),
    #           (-11, 45), (-12, 45), (-13, 45), (-14, 45), (-15, 45), (-16, 45), (-16, 44), (-16, 43), (-16, 42),
    #           (-16, 41), (-16, 40), (-16, 39), (-16, 38), (-17, 42), (-18, 42), (-19, 42), (-20, 42), (-19, 41),
    #           (-18, 40), (-17, 39)]
    # edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 56), (8, 9), (9, 10), (10, 11),
    #          (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 68), (17, 18), (18, 19), (19, 20),
    #          (20, 21), (21, 22), (22, 23), (23, 77), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
    #          (29, 30), (30, 31), (31, 112), (31, 32), (32, 33), (33, 34), (34, 125), (34, 35), (35, 126), (35, 36),
    #          (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46),
    #          (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (56, 57),
    #          (57, 58), (58, 59), (59, 60), (60, 64), (60, 61), (61, 62), (62, 63), (64, 65), (65, 66), (66, 67),
    #          (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (77, 78), (78, 79),
    #          (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 101), (84, 85), (85, 86), (86, 87), (87, 88),
    #          (88, 89), (89, 90), (90, 91), (91, 92), (92, 93), (93, 94), (94, 95), (95, 96), (96, 97), (97, 98),
    #          (98, 99), (99, 100), (101, 102), (102, 103), (103, 104), (104, 105), (105, 106), (106, 107), (107, 108),
    #          (108, 109), (109, 110), (110, 111), (112, 113), (113, 114), (114, 115), (115, 116), (116, 117), (117, 118),
    #          (118, 119), (119, 120), (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (126, 127), (127, 128),
    #          (128, 129), (129, 130), (130, 131), (131, 132), (132, 133), (133, 134), (134, 135), (135, 136), (136, 137),
    #          (137, 138), (138, 139), (139, 140), (140, 141), (141, 142), (142, 143), (143, 144), (144, 150), (144, 145),
    #          (145, 146), (146, 147), (147, 148), (148, 149), (150, 151), (151, 152), (152, 153), (153, 154), (154, 155),
    #          (155, 156), (156, 157), (157, 158), (158, 159), (159, 164), (159, 160), (160, 161), (161, 162), (162, 163),
    #          (163, 170), (164, 165), (165, 166), (166, 167), (167, 168), (168, 169), (169, 170)]
    #
    # list_dest_names = ['front_door', 'front_sarang_table', 'front_glass_table', 'powder_sink', 'kitchen_stove',
    #                    'kitchen_island', 'washing_machine', 'family_area', 'back_door', 'bedroom_weights',
    #                    'bedroon_bed', 'bathroom_sink1', 'bathroom_sink2']
    # list_dest_coor = [(0, 0), (8, 11), (9, 4), (-7, 17), (-12, 30), (-8, 23), (-19, 23), (6, 31), (0, 53), (-9, 50),
    #              (-9, 43), (-20, 42), (-16, 38)]  # revise these destination coordinates
    #
    # # defining rooms as [x1,y1,x2,y2] where (x1,y1) bottom left corner, (x2,y2) top-right corner
    #
    # # these will be replaced with lists in the def house section
    # front = [2, 0, 14, 15]  # substract 5 from all Ys
    # family = [2, 21, 14, 42]
    # study = [2, 42, 14, 57]
    # bedroom = [-17, 42, -2, 57]
    # bathroom = [-24, 34, -10, 45]
    # kitchen = [-24, 21, -2, 34]
    # powder = [-10, 15, -2, 18]
    # hallway_front = [-2, 0, 2, 15]
    # hallway_powder = [-2, 15, 2, 18]
    # hallway_family = [-2, 21, 2, 42]
    # hallway_study = [-2, 42, 2, 57]
    # hallway_bedroom = [-10, 34, -2, 42]
    #
    # # comment list_rooms_names and list_rooms_coor when home data is available
    # list_rooms_names = ['front', 'family', 'study', 'bedroom', 'bathroom', 'kitchen', 'powder', 'hallway_front',
    #                     'hallway_powder',
    #                     'hallway_family', 'hallway_study', 'hallway_bedroom']  # names of each room
    # list_rooms_coor = [front, family, study, bedroom, bathroom, kitchen, powder, hallway_front, hallway_powder,
    #                   hallway_family, hallway_study, hallway_bedroom]


def current_situation():
    global tx, ty, tz, x_true, y_true, trans_true, ox, oy, oz, ow, room_name, roll_x, pitch_y, yaw_z, roll_xd, pitch_yd, yaw_zd_temp, yaw_zd
    tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
    ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
    tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
    x_true = round(tx * math.cos(theta) - ty * math.sin(theta), 2)
    y_true = round(tx * math.sin(theta) + ty * math.cos(theta), 2)
    trans_true = round(math.sqrt(x_true * x_true + y_true * y_true + tz * tz), 2)
    # Display orientation quaternion
    ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3)
    oy = round(zed_pose.get_orientation(py_orientation).get()[1], 3)
    oz = round(zed_pose.get_orientation(py_orientation).get()[2], 3)
    ow = round(zed_pose.get_orientation(py_orientation).get()[3], 3)
    room_name = 'waiting'
    # Change quaternion to Angles in degrees
    roll_x, pitch_y, yaw_z = euler_from_quaternion(ox, oy, oz, ow)
    roll_xd = round(roll_x * (180.0 / math.pi), 1)
    pitch_yd = round(pitch_y * (180.0 / math.pi), 1)
    yaw_zd_temp = -round(yaw_z * (180.0 / math.pi), 1)
    yaw_zd = round(yaw_zd_temp - round(180.0 / math.pi * theta), 2)

def start_calib():
    global x_cal_start, y_cal_start, flag_cal
    x_cal_start = tx
    y_cal_start = ty

    temp=('starting calibration at:', (x_cal_start, y_cal_start))
    print(temp)
    text_list =str(temp)
    update_list_audio(text_list)
    flag_cal = 1
    time.sleep(0.3)

def finish_calib():
    global trans_cal, x_meas, y_trav, theta,flag_cal
    # time.sleep(1)
    print('completing calibration...')
    trans_cal = round(math.sqrt(math.pow(tx - x_cal_start, 2) + math.pow(ty - y_cal_start, 2)), 2)  # *3.28084
    x_meas = tx  # float(input("enter x_meas:"))
    y_trav = trans_cal  # float(input("enter y_trav:"))
    theta = math.asin(x_meas / y_trav)
    print('completed calibration', x_meas, y_trav, theta)
    print('return to origin begin mapping')
    text_list='completed calibration, return to origin'
    update_list_audio(text_list)
    flag_cal=2
    time.sleep(0.3)

def map2():
    global x_curr, y_curr, temp_last, trans2, temp2, points, edges, room_name
    x_curr = x_true
    y_curr = y_true
    temp1 = (x_curr, y_curr)
    # check if this point is a duplicate, ie within "thres" of a point in points list
    flag = 0

    for k in points:
        k_index = points.index(k)
        dist = math_dist(k, temp1)
        if (dist < 1): #thres
            flag = 0
            # print('point', temp1, 'is repeat of index:', k_index)
            temp_last = k
            trans2 = round(math.sqrt(k[0] * k[0] + k[1] * k[1] + tz * tz), 2)
            # join to this point k_index
            break
        else:
            flag = 1
    if (flag == 1):
        points.append(temp1)
        print('added to map points at index:', len(points) - 1, temp1)
        temp2 = (points.index(temp_last), points.index(temp1))
        edges.append(temp2)
        print('added to map edges at index:', len(edges) - 1, temp2)
        print(temp2)
        temp_last = temp1
        # printing
        tempx = ('mapped point', len(points) - 1)
        text_list = str(tempx)
        update_list_audio(text_list)
        room_name = getroom(x_true, y_true)
        temp_print=room_name, round(x_true, 2), round(y_true, 2), round(trans_true, 2),\
              "timestamp: {3}, tx: {0}, ty:  {1}, tz:  {2}".format(tx, ty, tz,\
                                                                   round(time1, 2)),\
              "Orientation: ox: {0}, oy:  {1}, oz: {2}, ow: {3}".format(ox, oy, oz, ow)
        print_update(str(temp_print))
        trans2 = trans_true
        with open('/usr/local/zed/points_'+home+'.csv', 'a+',
                  newline='') as patient_output:
            patient_writer = csv.writer(patient_output, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            patient_writer.writerow(temp1)
        with open('/usr/local/zed/edges_'+home+'.csv', 'a+',
                  newline='') as patient_output:
            patient_writer = csv.writer(patient_output, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            patient_writer.writerow(temp2)

def save_destination():
    global key_press,text_list
    # save destination as island in kitchen
    start = 'as '
    end = ' in'
    s = key_press
    dest = s[s.find(start) + len(start):s.rfind(end)]  # island

    start = 'in '
    end = ''
    s = key_press
    room = s[s.find(start) + len(start):s.rfind(end)]  # kitchen

    with open('/usr/local/zed/destinations_'+home+'.csv', 'a+',
              newline='') as patient_output:
        patient_writer = csv.writer(patient_output, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        patient_writer.writerow([dest, room,x_curr, y_curr,yaw_zd])
    temp=('saved destination as',dest,'in',room,x_curr, y_curr,yaw_zd)
    print(str(temp))
    text_list = str(temp)
    update_list_audio(text_list)

    key_press=''

def save_room():
    global key_press,text_list
    # save destination as island in kitchen
    s = key_press
    start = 'as '
    end=''
    if(s.find('bottom')!=-1):
        end = ' bottom'
    if (s.find('top') != -1):
        end = ' top'
    room = s[s.find(start) + len(start):s.rfind(end)]
    start2=room+' '
    end2=''
    location=s[s.find(start2) + len(room):s.rfind(end2)]

    with open('/usr/local/zed/rooms_'+home+'.csv', 'a+',
              newline='') as patient_output:
        patient_writer = csv.writer(patient_output, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
        patient_writer.writerow([room,location,x_true, y_true])
    temp=('saved as', room,location,x_true, y_true)
    print(str(temp))
    text_list = str(temp)
    update_list_audio(text_list)

    key_press=''


def where_am_i():
    temp_getroom=getroom(x_true, y_true)
    temp = ('you are at', x_true, y_true, yaw_zd,'in',temp_getroom) #change to get_room
    print(str(temp))
    text_list = str(temp)
    update_list_audio(text_list)

def select_home():
    global home,key_press
    s = key_press
    start = 'as '
    end = ''
    home = s[s.find(start) + len(start):s.rfind(end)]
    temp = 'selected home as '+home
    print(temp)
    text_list = str(temp)
    update_list_audio(text_list)
    key_press=''

def destination():
    global dest_start2, list_dest_names, list_dest_coor, dest_start, dest_start2, dest_index, dest_end, G, G_path, points, edges, pos, fig, ax, \
        shortest_path, points_path, edges_path, temp_path, pos_path, fig2, ax2, dest_curr, dest_curr_init, dest_next, points_path2, \
        beta, dest_next2, beta2, flag_dest, dist_dest_next_last, yaw_zd_last, dest_end2

    dest_start2 = (-999, -999)

    dest_start = (x_true, y_true)
    # check if destination start point is near the map
    for k in points:
        k_index = points.index(k)
        dist = math_dist(k, dest_start)
        if (dist <= 2 * thres):
            # flag = 0
            print('destination starting point', dest_start, 'is near map point', k, 'at index', k_index)
            # temp=('destination starting point', dest_start, 'is near map point', k, 'at index', k_index)
            # text_list=str(temp)
            # update_list_audio(text_list)
            dest_start2 = k
            break

    if (dest_start2 == (-999, -999)):
        print('destination starting point', dest_start, 'is NOT near map point')
        # temp=('destination starting point', dest_start, 'is NOT near map point')
        # text_list=str(temp)
        # update_list_audio(text_list)

    else:
        # select destination end from the list
        print('destinations available are listed below:')
        for i in list_dest_coor:
            print(list_dest_coor.index(i), ' - ', list_dest_names[list_dest_coor.index(i)])

        # dest_index_text = input('enter number for selected destination: ')
        # while dest_index_text.isdigit() == False:
        #     print('bad input, re-enter')
        #     dest_index_text = input('enter number for selected destination: ')
        dest_index_text=key_press_dest_index

        dest_index = int(dest_index_text)
        dest_end = list_dest_coor[dest_index]

        # verify that destination end is near the map
        for k in points:
            k_index = points.index(k)
            dist = math_dist(k, dest_end)
            if (dist <= thres):
                # flag = 0
                print('destination end point', dest_end, 'is near map point', k, 'at index', k_index)
                # temp=('destination end point', dest_end, 'is near map point', k, 'at index', k_index)
                # text_list=str(temp)
                # update_list_audio(text_list)
                dest_end2 = k
                break

        # display gragh of map and shortest path from destination start to end
        G = nx.Graph()
        G_path = nx.Graph()

        # map from positional_tracking2 code

        for i in range(len(edges)):
            add_edge_to_graph(G, points[edges[i][0]], points[edges[i][1]])

        # you want your own layout
        pos = {point: point for point in points}

        # add axis
        fig, ax = plt.subplots()
        nx.draw(G, pos=pos, node_color='k', ax=ax)
        nx.draw(G, pos=pos, node_size=60, ax=ax)  # draw nodes and edges
        # nx.draw_networkx_labels(G, pos=pos,font_size=8)  # draw node labels/names
        plt.axis("on")
        ax.set_xlim(-30, 30)
        ax.set_ylim(0, 60)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        # plt.show()
        # #shortest path from dest_start2 to dest_end2

        try:
            shortest_path = nx.bidirectional_shortest_path(G, source=dest_start2, target=dest_end2)
            # print(shortest_path)

            points_path = shortest_path
            edges_path = []

            # create edges for points on shortest_path
            for i in points_path:
                # print(points_path.index(i), round(i[0], 2), round(i[1], 2))
                temp_path = (points_path.index(i), points_path.index(i) + 1)
                if (points_path.index(i) >= len(points_path) - 1):
                    break
                else:
                    edges_path.append(temp_path)

            # print(edges_path)

            for i in range(len(edges_path)):
                add_edge_to_graph(G_path, points_path[edges_path[i][0]], points_path[edges_path[i][1]])

            # you want your own layout
            pos_path = {point: point for point in points_path}

            # add axis
            fig2, ax2 = plt.subplots()
            nx.draw(G_path, pos=pos_path, node_color='k', ax=ax2)
            nx.draw(G_path, pos=pos_path, node_size=60, ax=ax2)  # draw nodes and edges
            # nx.draw_networkx_labels(G, pos=pos,font_size=8)  # draw node labels/names
            plt.axis("on")
            ax2.set_xlim(-30, 30)
            ax2.set_ylim(0, 60)
            ax2.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            # plt.show()
            dist_dest_next_last = 1000
            yaw_zd_last = yaw_zd - 30
            dest_curr = (x_true, y_true)
            dest_curr_init = (x_true, y_true)
            dest_next = points_path[points_path.index(dest_start2)]
            points_path2 = points_path[points_path.index(dest_next):(len(points_path) - 1)]
            for i in points_path2:
                if (math_dist(dest_next, (x_true, y_true)) <= 0.75 * thres or math_dist(dest_end2,
                                                                                        (x_true, y_true)) < math_dist(
                    dest_end2, dest_next)):
                    dest_next = points_path2[points_path2.index(i) + 1]
                    continue
                beta = get_beta(dest_next, dest_curr)

                try:
                    dest_next2 = points_path2[points_path2.index(i) + 1]
                    beta2 = get_beta(dest_next2, dest_next)

                    if (abs(beta2 - beta) <= thres_angle):
                        dest_next = dest_next2
                        pass
                    else:
                        break
                except:
                    pass
            temp_print='from', dest_curr_init, ' to dest_next', dest_next
            print_update(str(temp_print))
            # temp=('from', dest_curr_init, ' to dest_next', dest_next)
            # text_list=str(temp)
            # update_list_audio(text_list)
            flag_dest = 1
        except:
            pass
    time.sleep(0.3)

def dest_mode():
    global flag_server,thread_server,dist_dest_next, dest_curr, beta, pd, side, dist_dest_next_last, yaw_zd_last, \
        list_audio, dest_curr_init, points_path3, dest_next, dest_next2, beta2, flag_dest, time1_last, list_last2, time3
    # if(0<time1%5<0.01):
    #     print('(x_true, y_true), dest_next, dist_dest_next, dist_dest_next_last, beta, yaw_zd,yaw_zd_last)')
    #     print((x_true,y_true),dest_next,dist_dest_next,dist_dest_next_last,beta,yaw_zd,yaw_zd_last)

    dist_dest_next = math_dist(dest_next, (x_true, y_true))

    if (dist_dest_next >= thres):

        dest_curr = (x_true, y_true)
        beta = get_beta(dest_next, dest_curr)

        pd, side, location = dist_perp(dest_curr_init, dest_next, dest_curr)

        if (dist_dest_next_last - dist_dest_next >= min(thres_print, max(0.25 * thres_print, 0.25 * dist_dest_next))):
            time3 = time1
            if (abs(beta - yaw_zd) <= thres_angle_print and pd <= 1.0):
                temp_print='move forward', (x_true, y_true), dest_next, round(dist_dest_next, 2),round(dist_dest_next_last, 2), round(beta, 1), yaw_zd, yaw_zd_last, pd
                print_update(str(temp_print))
                dist_dest_next_last = dist_dest_next

                # text_list = ('move forward for ' + str(int(dist_dest_next)) + ' feet')
                text_list = ('move ' + str(int(dist_dest_next)))
                update_list_audio(text_list)
                vibration(0,1,0)

            if (abs(yaw_zd - yaw_zd_last) >= thres_angle_print or pd > 1.0):
                if (pd > 1.0 and side == 'left'):
                    temp_print=temp_true+': turn right by 45 degrees and move forward 1 ft'
                    print_update(str(temp_print))
                    text_list = ('right 45')
                    update_list_audio(text_list)
                    vibration(0, 1, 1)
                    dist_dest_next_last = dist_dest_next
                    yaw_zd_last = yaw_zd
                elif (pd > 1.0 and side == 'right'):
                    temp_print=temp_true+':turn left by 45 degrees and move forward 1 ft'
                    print_update(str(temp_print))
                    text_list = ('left 45')
                    update_list_audio(text_list)
                    vibration(1, 1, 0)
                    dist_dest_next_last = dist_dest_next
                    yaw_zd_last = yaw_zd
                elif (beta - yaw_zd > thres_angle_print or (pd > 1.0 and side == 'left')):
                    if (abs(beta - yaw_zd) <= 180):
                        temp_print=temp_true+':turn right by ', round(abs(beta - yaw_zd), 1), ' degrees',\
                              (x_true, y_true), dest_next, round(dist_dest_next, 2),\
                              round(dist_dest_next_last, 2), round(beta, 1), yaw_zd, yaw_zd_last, pd, side
                        print_update(str(temp_print))
                        text_list = ('right ' + str(int(abs(beta - yaw_zd))) + ' degrees')
                        update_list_audio(text_list)
                        vibration(0, 1, 1)
                    else:
                        temp_print=temp_true+':turn left by ', 360 - round(abs(beta - yaw_zd), 1), ' degrees',\
                              (x_true, y_true), dest_next, round(dist_dest_next, 2),\
                              round(dist_dest_next_last, 2), round(beta, 1), yaw_zd, yaw_zd_last, pd, side
                        print_update(str(temp_print))
                        text_list = ('left ' + str(360 - int(abs(beta - yaw_zd))) + ' degrees')
                        update_list_audio(text_list)
                        vibration(1, 1, 0)

                    # print('turn right by ', round(abs(beta - yaw_zd),1), 'degrees',(x_true,y_true),dest_next,round(dist_dest_next,2),round(dist_dest_next_last,2),round(beta,1),yaw_zd,yaw_zd_last)
                    yaw_zd_last = yaw_zd

                elif (beta - yaw_zd < -thres_angle_print or (pd > 1.0 and side == 'right')):
                    if (abs(beta - yaw_zd) <= 180):
                        temp_print=temp_true+':turn2 left by ', round(abs(beta - yaw_zd), 1), 'degrees', (x_true, y_true), dest_next,\
                              round(dist_dest_next, 2), round(dist_dest_next_last, 2), round(beta, 1), yaw_zd,\
                              yaw_zd_last, pd, side
                        print_update(str(temp_print))
                        text_list = ('left ' + str(int(abs(beta - yaw_zd))) + ' degrees')
                        update_list_audio(text_list)
                        vibration(1, 1, 0)

                    else:
                        temp_print=temp_true+':turn2 right by ', 360 - round(abs(beta - yaw_zd), 1), ' degrees',\
                              (x_true, y_true), dest_next, round(dist_dest_next, 2),\
                              round(dist_dest_next_last, 2), round(beta, 1), yaw_zd, yaw_zd_last, pd, side
                        print_update(str(temp_print))
                        text_list = ('right ' + str(360 - int(abs(beta - yaw_zd))) + ' degrees')
                        update_list_audio(text_list)
                        vibration(0, 1, 1)

                    yaw_zd_last = yaw_zd

        elif (time1 - time1_last >= 5 and time1 - time3 >= 3):
            temp_print=temp_true+':move forward at '+str(round(time1,1))+" s"
            print_update(str(temp_print))
            text_list = ('move')# at '+str(round(time1,1))+" s")
            update_list_audio2(text_list)
            vibration(0, 1, 0)
            time1_last = time1
            list_last2 = ""
            # list_audio = []



        # put the code here for checking if the object detected are along the path
        # if(flag_server==0):
        #     thread_server = Thread(target=object_server())
        #     thread_server.start()

    else:
        dist_dest_next_last = 1000
        yaw_zd_last = yaw_zd - 30
        dest_curr = (x_true, y_true)
        dest_curr_init = (x_true, y_true)

        points_path3 = points_path[points_path.index(dest_next):(len(points_path) - 1)]
        # print(points_path3)

        if (len(points_path3) <= 1): reached_destination()

        for i in points_path3:
            if (math_dist(dest_next, (x_true, y_true)) <= 0.75 * thres or math_dist(dest_end2,
                                                                                    (x_true, y_true)) < math_dist(
                dest_end2, dest_next)):
                dest_curr = dest_next
                try:
                    dest_next = points_path3[points_path3.index(i) + 1]
                except:
                    reached_destination()
                    break
                continue
            beta = get_beta(dest_next, dest_curr)

            try:
                dest_next2 = points_path3[points_path3.index(i) + 1]
                beta2 = get_beta(dest_next2, dest_next)

                if (abs(beta2 - beta) <= thres_angle):
                    dest_next = dest_next2
                    pass
                else:
                    break
            except:
                pass

        temp_print='from2', dest_curr, ' to dest_next', dest_next
        print_update(str(temp_print))
        # text_list='from2 '+str(dest_curr)+ ' to dest_next '+ str(dest_next)
        # update_list_audio(text_list)

def reached_destination():
    global text_list, list_audio, flag_dest,text_list

    print('reached destination', (x_true, y_true,yaw_zd))
    text_list = 'reached destination at '+str(list_dest_names[dest_index])
    update_list_audio(text_list)
    flag_dest = 0

def update_list_audio(text1):
    global list_audio,x_true,y_true,temp_true,time_update_last,text1_last,time2
    time_update=time.time()-time2
    if(text1!=text1_last or time_update-time_update_last>=3):
        # print(text1,text1_last,time_update,time_update_last)
        # temp_true = str(round(x_true, 1)) + " " + str(round(y_true, 1))
        temp_true=''
        text2 = temp_true + " " + text1
        if (len(list_audio) > 20):
            list_audio.remove(list_audio[0])
            list_audio.append(text2)
        else:
            list_audio.append(text2)
        time_update_last = time.time() - time2
        text1_last=text1

def update_list_audio2(text1):
    global list_audio2,x_true,y_true,temp_true2,time_update_last2,text1_last2,time2
    time_update2=time.time()-time2
    if(text1!=text1_last2 or time_update2-time_update_last2>=3):
        # print(text1,text1_last,time_update,time_update_last)
        # temp_true = str(round(x_true, 1)) + " " + str(round(y_true, 1))
        temp_true2=''
        text2 = temp_true2 + " " + text1
        if (len(list_audio2) > 20):
            list_audio2.remove(list_audio2[0])
            list_audio2.append(text2)
        else:
            list_audio2.append(text2)
        time_update_last2 = time.time() - time2
        text1_last2=text1

def print_update(text1):
    global time_print_last,text_print_last,time2
    time_print=time.time()-time2
    if(text1!=text_print_last or time_print-time_print_last>=3):
        print(text1)
        text_print_last=text1
        time_print_last=time.time()-time2

def summarize():
    # print('points=', points)
    # for m in points:
    #     print(m[0], m[1])
    #
    # print('edges=', edges)
    # for e in edges:
    #     print(e[0], e[1])

    # '+str(dt_string)+'
    # zed.disable_positional_tracking("./goel-house_"+str(dt_string)+".area") #save area

    # create map for valid points(nodes) and edges
    G = nx.Graph()

    # map from positional_tracking2 code

    for i in range(len(edges)):
        add_edge_to_graph(G, points[edges[i][0]], points[edges[i][1]])

    # you want your own layout
    pos = {point: point for point in points}

    # add axis
    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, node_color='k', ax=ax)
    nx.draw(G, pos=pos, node_size=60, ax=ax)  # draw nodes and edges
    # nx.draw_networkx_labels(G, pos=pos,font_size=8)  # draw node labels/names
    plt.axis("on")
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 60)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()

def start_map():
    global flag_map, x_curr, y_curr, temp, points, edges,temp_last,text_list
    if(flag_map!=0):
        print('map already started-use continue map')
        text_list = ('map already started-use continue map')
        update_list_audio(text_list)
    else:
        flag_map = 1
        points = []
        edges = []
        # print(flag_map)
        print('start mapping')
        text_list='start mapping'
        update_list_audio(text_list)
        x_curr = x_true
        y_curr = y_true
        temp = (x_curr, y_curr)
        points.append(temp)
        temp_last = temp
        print('added to map poq ints at index:', len(points) - 1, temp)
        text_list='added to map poq ints at index: '+ str(len(points) - 1)+" "+ str(temp)
        update_list_audio(text_list)

        filePath = '/usr/local/zed/points_'+home+'.csv'
        filePath_backup = '/usr/local/zed/points_'+home+'backup.csv'

        # As file at filePath is deleted now, so we should check if file exists or not not before deleting them
        # if os.path.exists(filePath):
        #     copyfile(filePath, filePath_backup)
        #     os.remove(filePath)

        filePath2 = '/usr/local/zed/edges_'+home+'.csv'
        filePath2_backup = '/usr/local/zed/edges_'+home+'_backup.csv'

        # As file at filePath is deleted now, so we should check if file exists or not not before deleting them
        # if os.path.exists(filePath2):
        #     copyfile(filePath2, filePath2_backup)
        #     os.remove(filePath2)

        filePath3 = '/usr/local/zed/destinations_'+home+'.csv'
        filePath3_backup = '/usr/local/zed/destinations_'+home+'_backup.csv'

        # As file at filePath is deleted now, so we should check if file exists or not not before deleting them
        # if os.path.exists(filePath3):
        #     copyfile(filePath3, filePath3_backup)
        #     os.remove(filePath3)

        filePath4 = '/usr/local/zed/rooms_' + home + '.csv'

        with open(filePath, 'a+',
                  newline='') as patient_output:
            patient_writer = csv.writer(patient_output, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            patient_writer.writerow(str(dt_string))
            patient_writer.writerow(temp)

        with open(filePath2, 'a+',
                  newline='') as patient_output:
            patient_writer = csv.writer(patient_output, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            patient_writer.writerow(str(dt_string))

        with open(filePath3, 'a+',
                  newline='') as patient_output:
            patient_writer = csv.writer(patient_output, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            patient_writer.writerow(str(dt_string))

        with open(filePath4, 'a+',
                  newline='') as patient_output:
            patient_writer = csv.writer(patient_output, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            patient_writer.writerow(str(dt_string))

    time.sleep(0.3)  # needed to avoid multiple triggers of key press


def continue_map():
    global flag_map, x_curr, y_curr, temp, points, temp_last, text_list
    flag_map = 1

    # print(flag_map)
    print('continue mapping')
    text_list = 'continue mapping'
    update_list_audio(text_list)
    map2()

    time.sleep(0.3)  # needed to avoid multiple triggers of key press

def pause_map():
    global flag_map,text_list
    flag_map=2
    print('mapping paused')
    text_list='mapping paused'
    update_list_audio(text_list)
    time.sleep(0.3)  # needed to avoid multiple triggers of key press

def stop_map():
    global flag_map,text_list
    flag_map = 0
    print('mapping stopped')
    text_list='mapping stopped'
    update_list_audio(text_list)
    time.sleep(0.3)  # needed to avoid multiple triggers of key press

def print_update2():
    global room_name, trans2
    room_name = getroom(x_true, y_true)
    print(room_name, round(x_true, 2), round(y_true, 2), round(trans_true, 2),
          "timestamp: {3}, tx: {0}, ty:  {1}, tz:  {2}".format(tx, ty, tz, round(time1, 2)),
          "Orientation: ox: {0}, oy:  {1}, oz: {2}, ow: {3}".format(ox, oy, oz, ow), yaw_zd, yaw_zd_temp)
    temp=('you are in the ',room_name,' at ',(round(x_true, 2), round(y_true, 2)))
    text_list=str(temp)
    # update_list_audio(text_list)
    trans2 = trans_true

if __name__ == "__main__":

    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD720 video mode (default fps: 60)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP  # Use a right-handed Y-up coordinate system
    init_params.coordinate_units = sl.UNIT.FOOT  # Set units in meters
    init_params.camera_fps = 60
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    zed = sl.Camera()
    status = zed.open(init_params)
    while status != sl.ERROR_CODE.SUCCESS:
        status = zed.open(init_params)
        print(repr(status))
        time.sleep(1)


    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    runtime_parameters = sl.RuntimeParameters()
    py_translation = sl.Translation()
    pose_data = sl.Transform()

    # Enable positional tracking with default parameters
    tracking_parameters = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_parameters)

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    # runtime_parameters = sl.RuntimeParameters()
    width = 800
    height = 600
    image_size = sl.Resolution(width, height)
    image_np_global = np.zeros([width, height, 3], dtype=np.uint8)
    depth_np_global = np.zeros([width, height, 4], dtype=np.float)

    # net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.2)
    net = jetson.inference.detectNet("ssd-inception-v2", threshold=0.5)
    camera = jetson.utils.videoSource("csi://0")  # '/dev/video0' for V4L2
    display = jetson.utils.videoOutput("display://0")  # 'my_video.mp4' for file
    set_default()  # set all defaults

    thread = Thread(target=audio_thread)
    thread.start()

    thread2 = Thread(target=thread_detection)
    thread2.start()

    time2 = time.time()

    thread3 = Thread(target=thread_position)
    thread3.start()

    print('camera started')
    text_list = 'camera started'
    update_list_audio2(text_list)


    house_map()  # load house map
    time1_last = time.time() - time2
    time3 = time.time() - time2
    time13 = time.time() - time2
    time43 = time.time() - time2
    research_distance_box = 30

    # while status == sl.ERROR_CODE.SUCCESS:
    while True:
        time1 = round(time.time() - time2,2)
        zed_pose = sl.Pose()
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            tracking_state = zed.get_position(zed_pose)
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%m%d%Y-%H%M%S")
            time11 = round(time.time() - time2, 2)

            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                # Get the pose of the camera relative to the world frame
                # state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.FRAME_WORLD)
                # Display translation and timestamp
                py_translation = sl.Translation()
                py_orientation = sl.Orientation()
                current_situation()
                time11 = round(time.time() - time2, 2)

                if ((key_press=='start calibration') and flag_cal == 0):  start_calib()  # start calibration
                if(flag_cal==1):
                    temp=x_true,y_true,yaw_zd
                    # print(str(temp))
                    text_list9=str(temp)
                    time42=time.time()-time2
                    if(time42-time43>=2):
                        update_list_audio(text_list9)
                        time43=time42
                if ((key_press=='stop calibration') and flag_cal == 1): finish_calib()  # finish calibration
                if (key_press=='quit'): break  # stop the code
                if (key_press=='escape'):  # summarize and stop the code
                    summarize()
                    break
                pose_data = zed_pose.pose_data(sl.Transform())
                time13 = round(time.time() - time2, 2)
                # print('loop time',round(time13-time1,2))
                # moved to thread_detection

            else:
                if (time1 % 1 == 0):
                    print('tracking_state not OK ', time1)
                    text_list='tracking_state not OK '+str(round(time1,1))
                    update_list_audio2(text_list)

        else:
            if (time1 % 1 == 0):
                print('zed.grab failed ', time1)
                text_list = 'zed grab failed ' + str(round(time1, 1))
                update_list_audio2(text_list)

    # zed.disable_object_detection()
    zed.disable_positional_tracking()
    # zed.disable_streaming()
    zed.close()
    stop = False
    # thread_server.join()
    thread.join()
    thread2.join()
    thread3.join()
