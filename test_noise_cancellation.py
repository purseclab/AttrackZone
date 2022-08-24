# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import argparse, cv2, torch, json
import numpy as np
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists

from net import SiamRPNotb
from run_defense import SiamRPN_init, SiamRPN_track
from utils import rect_2_cxy_wh, cxy_wh_2_rect

import glob
from datetime import datetime


parser = argparse.ArgumentParser(description='PyTorch SiamRPN OTB Test')
parser.add_argument('--dataset', dest='dataset', default='OTB2015', help='datasets')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
#    xB = min(boxA[2], boxB[2])
#    yB = min(boxA[3], boxB[3])
    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
#    print("RETURNING IOU: " + str(iou))
    # return the intersection over union value
    return iou

# function to get the output layer names 
# in the architecture
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, classes, COLORS, class_id, confidence, x, y, x_plus_w, y_plus_h, index=None):
    label = str(classes[class_id])
    if index is not None:
        label += " (" + str(index) + ")"
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_bbox(image):
    config_path = '' #Path to yolov3 config
    weights_path = '' #Path to yolov3 weights
    names_path = '' #path to yolov3 names
    bboxes_output = []
    current_class = None
    current_confidence = None

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # read class names from text file
    classes = None
    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # read pre-trained model and config file
    net = cv2.dnn.readNet(weights_path, config_path)

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    # go through the detections remaining
    # after nms and draw bounding box
    numBoxes = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_bounding_box(image, classes, COLORS, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), i)
        numBoxes += 1
        # display output image    
    if(numBoxes > 0):
        cv2.imshow("object detection", image)

        # wait until any key is pressed
        cv2.waitKey()
        # release resources
        cv2.destroyAllWindows()
        bb_select = input("Which class should we select?")
        if(bb_select != "" and bb_select.isdigit()):
            return boxes[int(bb_select)]
        elif bb_select == "automatic":
            return -1
        else:
            return -2
    else:
        return -2
    


def track_video(model, videos):
    for video in videos:
        dpath = '' #fill in as needed
        out_path = join(dpath, video.split('\\')[-1], datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        image_save = 0
        toc, regions = 0, []
        f = 0
    #    image_files, gt = video['image_files'], video['gt']
        cap = cv2.VideoCapture(video)
        if not (cap.isOpened()):
            print("Error reading video " + video)
            continue
        cached_frames = []
        while(cap.isOpened()):
            if len(cached_frames) < 1:
                ret, im = cap.read()
                if not ret:
                    break
            else:
                im = cached_frames.pop(0)
#            im = cv2.imread(image_file)  # TODO: batch load
            tic = cv2.getTickCount()
            if f == 0:  # init
                bbox = get_bbox(im)
                if bbox == -1:
                    bbox = [0, 0, im.shape[1], im.shape[0]]
                elif bbox == -2:
                    cached_frames.append(im)
                    bbox_2 = -2
                    while bbox_2 == -2:
                        ret, im = cap.read()
                        if not ret:
                            bbox = [0, 0, im.shape[1], im.shape[0]]
                            im = cached_frames.pop(0)
                            break
                        bbox_2 = get_bbox(im)
                        if bbox_2 == -1:
                            bbox = [0, 0, im.shape[1], im.shape[0]]
                            cached_frames.append(im)
                            im = cached_frames.pop(0)
                            break
                        elif bbox_2 != -2:
                            bbox = bbox_2
                            cached_frames.append(im)
                            im = cached_frames.pop(0)
                            break
                        cached_frames.append(im)
                
                            
                target_pos, target_sz = rect_2_cxy_wh(bbox)
                state = SiamRPN_init(im, target_pos, target_sz, model) # init tracker
                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                regions.append(np.array(bbox))
                att_per = 0  # adversarial perturbation in attack
                def_per = 0  # adversarial perturbation in defense
            elif f > 0:  # tracking
                if f % 30 == 1:  # clean the perturbation from last frame
                    att_per = 0
                    def_per = 0
                    state, att_per, def_per = SiamRPN_track(state, im, f, regions[f-1], att_per, def_per, image_save, iter=10)  # gt_track
                    location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])
                    regions.append(location)
                else:
                    state, att_per, def_per = SiamRPN_track(state, im, f, regions[f-1], att_per, def_per, image_save, iter=5)  # gt_track
                    location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])
                    regions.append(location)
            toc += cv2.getTickCount() - tic

            if args.visualization and f >= 0:  # visualization
                if f == 0: cv2.destroyAllWindows()
                if len(location) == 8:
                    cv2.polylines(im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 2)
                else:
                    location = [int(l) for l in location]  #
                    cv2.rectangle(im, (location[0], location[1]),
                                  (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 2)
                cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                #Save to the out path
                if not isdir(out_path): 
                    makedirs(out_path)
                if not cv2.imwrite(join(out_path, '%d.jpg' % f), im):
                    print("COULD NOT WRITE ADDED")
                    print(out_path)
                    return -1
                cv2.imshow('Noise Cancellation Demo', im)
                cv2.waitKey(1)
            f += 1
        toc /= cv2.getTickFrequency()

        # save result
        video_path = join('test', args.dataset, 'DaSiamRPN_defense')
        if not isdir(video_path): makedirs(video_path)
        result_path = join(video_path, '{:s}.txt'.format(video['name']))
        with open(result_path, "w") as fin:
            for x in regions:
                fin.write(','.join([str(i) for i in x])+'\n')

        print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
            v_id, video['name'], toc, f / toc))
    return f / toc


def load_dataset(dataset):
    base_path = join(realpath(dirname(__file__)), 'data', dataset)
    if not exists(base_path):
        print("Please download OTB dataset into `data` folder!")
        exit()
    json_path = join(realpath(dirname(__file__)), 'data', dataset + '.json')
    info = json.load(open(json_path, 'r'))
    for v in info.keys():
        path_name = info[v]['name']
        info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
        info[v]['gt'] = np.array(info[v]['gt_rect'])-[1,1,0,0]  # our tracker is 0-index
        info[v]['name'] = v
    return info


def main():
    global args, v_id
    args = parser.parse_args()

    net = SiamRPNotb()
    net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNOTB.model')))
    #net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNVOT.model')))
    net.eval().cuda()
    vids_path = '.\\trials_morales\\attack'
#    vids_path = '.\\trials_morales\\benign'

    vid_list = glob.glob(vids_path + '\\*')
#    dataset = load_dataset(args.dataset)
    fps_list = []
    fps_list.append(track_video(net, vid_list))
    print('Mean Running Speed {:.1f}fps'.format(np.mean(np.array(fps_list))))


if __name__ == '__main__':
    main()
