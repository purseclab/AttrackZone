# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import argparse, cv2, torch, json
import numpy as np
import os
from os import makedirs
from os.path import realpath, dirname, join, isdir, exists

from net import SiamRPNotb, SiamRPNBIG, SiamRPNvot
from run_attack import SiamRPN_init, SiamRPN_track
from utils import rect_2_cxy_wh, cxy_wh_2_rect
from datetime import datetime
import sys

import pixellib
from pixellib.semantic import semantic_segmentation
import segment

parser = argparse.ArgumentParser(description='PyTorch SiamRPN OTB Test')
parser.add_argument('--dataset', dest='dataset', default='OTB2015', help='datasets')
parser.add_argument('-v', '--visualization', dest='visualization', action='store_true',
                    help='whether visualize result')

realworldattack = False
realtimeattack = False
output_bboxes_on_added = False
output_noise = False


def track_video(model, video, dataset, net2=None):
    image_save = 0
    toc, regions = 0, []
    final_pos = None
    image_files, gt = video['image_files'], video['gt']
    segment_image = None
    az_utils = []
    max_perturbation = -1
    

    #print(video)
    if 'attack_mask' in video:
        attack_masks = video['attack_mask']
    else:
        attack_masks = None
        segment_image = semantic_segmentation()
        segment_image.load_ade20k_model("deeplabv3_xception65_ade20k.h5")

        
    out_path = join('out_data\\BIG_TO_OTB', dataset, datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    for f, image_file in enumerate(image_files):
        if f >= len(gt):
            break
        im = cv2.imread(image_file)  # TODO: batch load
#        print("IMAGE SIZE")
#        print(im.shape)
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            ## CODE FOR RETARGETING
            #Select move in or out based on where the thingamobab is
            if(target_pos[0] + target_sz[0]/2 > im.shape[0]/2):
                dxywh = [200, 0, 0, 0] # Attempt to move to the left by 200 px; remember to flip the sign for all nums!
            else:
                dxywh = [-200, 0, 0, 0]
            final_pos = [target_pos[0] + dxywh[0], target_pos[1] + dxywh[1], target_sz[0] + dxywh[2], target_sz[1] + dxywh[3]]
#            target_pos, target_sz = rect_2_cxy_wh(gt[f])
            state = SiamRPN_init(im, target_pos, target_sz, model, model_eval=net2) # init tracker
            location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            regions.append(gt[f])
            att_per = 0  # adversarial perturbation in attack
            def_per = 0  # adversarial perturbation in defense
        elif f > 0:  # tracking
            if f % 30 == 1:  # clean the perturbation from last frame
                att_per = 0
                def_per = 0
                state, att_per, def_per = SiamRPN_track(state, im, f, regions[f-1], att_per, def_per, image_save, iter=10, attack_mask=None, final_pos = final_pos, im_bounds = [im.shape[1], im.shape[0]], use_alt_model = True)  # gt_track
                location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])
                regions.append(location)
            else:
                state, att_per, def_per = SiamRPN_track(state, im, f, regions[f-1], att_per, def_per, image_save, iter=5, attack_mask=None, final_pos = final_pos, im_bounds = [im.shape[1], im.shape[0]], use_alt_model = True)  # gt_track
                location = cxy_wh_2_rect(state['target_pos']+1, state['target_sz'])
                regions.append(location)
        toc += cv2.getTickCount() - tic
        if type(att_per) != type(0):
            im_old = im # To restore later
            #Transformation jutsu
            transf = att_per[0].cpu().detach().numpy()
            transf = np.reshape(transf, (transf.shape[1], transf.shape[2], transf.shape[0]))
            # REMOVE 0 VALS
            transf = np.where(transf<0, 0, transf)
            if attack_masks != None:
                attack_mask = attack_masks[f]
                if(attack_mask != None):
                    # Upscale
                    # Do some funky stuff
                    mask3d = np.dstack((attack_mask, attack_mask, attack_mask))
                    rdim = (len(attack_mask[0]), len(attack_mask))
                    #print(rdim)
                    transf = cv2.resize(transf, rdim, interpolation=cv2.INTER_CUBIC)

                    transf = np.multiply(transf, mask3d)
            else:
                mask_i, az_util = segment.segmentation_attack_mask(segment_image, image_file)
                az_utils.append(az_util)
                mask3d = np.stack((mask_i,)*3, axis=-1)
                rdim = (len(mask_i[0]), len(mask_i))
                transf = cv2.resize(transf, rdim, interpolation=cv2.INTER_CUBIC)
                transf = np.multiply(transf, mask3d)
                    #
            max_noise_idx = np.unravel_index(transf.argmax(), transf.shape)
            if transf[max_noise_idx[0]][max_noise_idx[1]][max_noise_idx[2]] > max_perturbation:
                max_perturbation = transf[max_noise_idx[0]][max_noise_idx[1]][max_noise_idx[2]]
#            # Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
#            transf = (255*(transf - np.min(transf))/np.ptp(transf))       
#            print(transf[correct_idx[0]][correct_idx[1]][correct_idx[2]])
#            correct_idx = np.unravel_index(transf.argmin(), transf.shape)
#            print(transf[correct_idx[0]][correct_idx[1]][correct_idx[2]])

            # Test negative image
#            sub_arr = np.full(transf.shape, 255)
#            transf = sub_arr - transf
            #im = np.resize(transf, im.shape)
            rdim = (im.shape[1], im.shape[0])
#            rdim = (1920, 1080)
            im = cv2.resize(transf, rdim, interpolation=cv2.INTER_CUBIC)
            if realworldattack:
                #export a different version
                transf *= 20
                tm = cv2.resize(transf, rdim, interpolation=cv2.INTER_CUBIC)
                transf /= 20
            else:
                tm = im
            # Save both to directory (TODO: REENABLE)
            if output_noise:
                current_noise_dir = os.path.join(out_path, 'noise')
                if not isdir(current_noise_dir): 
                    makedirs(current_noise_dir)
                if not cv2.imwrite(os.path.join(current_noise_dir, '%d.jpg' % f), tm):
                    print(current_noise_dir)
                    return
                
            added_im = im_old + tm

            if output_bboxes_on_added and f >= 0:  # visualization
                # Uncomment below to get bboxes
                if len(gt[f]) == 8:
                    cv2.polylines(added_im, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
                else:
                    cv2.rectangle(added_im, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 2)
                if len(location) == 8:
                    cv2.polylines(added_im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 2)
                else:
                    location = [int(l) for l in location]  #
                    cv2.rectangle(added_im, (location[0], location[1]),
                                  (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 2)
                cv2.putText(added_im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            current_added_dir = os.path.join(out_path, 'added')
            if not isdir(current_added_dir): 
                makedirs(current_added_dir)
            if not cv2.imwrite(os.path.join(current_added_dir, '%d.jpg' % f), added_im):
                print("COULD NOT WRITE ADDED")
                print(current_added_dir)
                return
            
            if realtimeattack:
                im = tm
            else:
                im = im_old
            # Get the max value
#            correct_idx = np.unravel_index(im.argmax(), im.shape)
#            print(im[correct_idx[0]][correct_idx[1]][correct_idx[2]])
            #print(im)
        if args.visualization and f >= 0:  # visualization
            if f == 0: cv2.destroyAllWindows()
            # Uncomment below to get bboxes
            if len(gt[f]) == 8:
                cv2.polylines(im, [np.array(gt[f], np.int).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            else:
                cv2.rectangle(im, (gt[f, 0], gt[f, 1]), (gt[f, 0] + gt[f, 2], gt[f, 1] + gt[f, 3]), (0, 255, 0), 2)
            if len(location) == 8:
                cv2.polylines(im, [location.reshape((-1, 1, 2))], True, (0, 255, 255), 2)
            else:
                location = [int(l) for l in location]  #
                cv2.rectangle(im, (location[0], location[1]),
                              (location[0] + location[2], location[1] + location[3]), (0, 255, 255), 2)
            cv2.putText(im, str(f), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow(video['name'], im)
            cv2.waitKey(1)
        if type(att_per) != type(0):
            im = im_old
    toc /= cv2.getTickFrequency()
    print("UTILS")
    print(az_utils)
    print("MAX PERTURBATION")
    print(max_perturbation)
    stats_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_path)
    if not isdir(stats_dir): 
        makedirs(stats_dir)
    if(len(az_utils) > 0):
        np.savetxt(os.path.join(stats_dir, "utilization.txt"), np.array(az_utils))
    kuutoro = open(os.path.join(stats_dir, "stats.txt"), "w")
    kuutoro.write("Max perturbation: " + str(max_perturbation) + "\n")
    kuutoro.close()
    # save result
#    video_path = join('test', dataset, 'DaSiamRPN_attack')
#    if not isdir(video_path): makedirs(video_path)
#    result_path = join(video_path, '{:s}.txt'.format(video['name']))
#    with open(result_path, "w") as fin:
#        for x in regions:
#            fin.write(','.join([str(i) for i in x])+'\n')
    if toc == 0:
        toc = 0.00001
    print('({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
        v_id, video['name'], toc, f / toc))
    return f / toc


def load_dataset(dataset):
    base_path = join(realpath(dirname(__file__)), 'data', dataset)
#    print("Path: " + str(base_path))
#    if not exists(base_path):
#        print("Please download OTB dataset into `data` folder!")
#        exit()
    json_path = join(realpath(dirname(__file__)), 'data', dataset + '.json')
    info = json.load(open(json_path, 'r'))
    for v in info.keys():
        path_name = info[v]['name']
        info[v]['image_files'] = [join(base_path, path_name, 'img', im_f) for im_f in info[v]['image_files']]
        #info[v]['gt'] = np.array(info[v]['gt_rect'])-[1,1,0,0]  # our tracker is 0-index
        info[v]['gt'] = np.array(info[v]['gt'])
        info[v]['name'] = v
    return info


def main():
    global args, v_id
    args = parser.parse_args()
    dataset_names = [] # Fill with generated datasets or videos
    for dset in dataset_names:
        try:
            print(dset)
            net = SiamRPNBIG()
            net2 = SiamRPNotb()
#            net = SiamRPNotb()
#            net = SiamRPNvot()
            net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNBIG.model')))
            net2.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'SiamRPNOTB.model')))
            net.eval().cuda()
            net2.eval().cuda()

            dataset = load_dataset(dset)
            #print(dataset)
            fps_list = []
            for v_id, video in enumerate(dataset.keys()):
                if v_id > -1:
                    fps_list.append(track_video(net, dataset[video], dset, net2=net2))
            print('Mean Running Speed {:.1f}fps'.format(np.mean(np.array(fps_list))))
            fps_list.clear()
            del net
            del dataset
            del net2
        except Exception as e:
            print("!!!! FAILED TO RUN SET !!!!")
            print(e)
            fps_list.clear()
            del net
            del net2
            del dataset
            continue



if __name__ == '__main__':
    main()
