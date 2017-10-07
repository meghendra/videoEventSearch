import sys
import args

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch
from pdb import set_trace as st
import os
import cv2
import torch
import numpy as np
from torch.multiprocessing import Pool
from pdb import set_trace as st
from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
import shutil


class VideoFeatureExtractor:

    def __init__(self,model_path,infos_path):

        # Input arguments and options
        parser = argparse.ArgumentParser()
        # Input paths
        parser.add_argument('--model', type=str, default='',
                        help='path to model to evaluate')
        parser.add_argument('--cnn_model', type=str,  default='resnet101',
                        help='resnet101, resnet152')
        parser.add_argument('--infos_path', type=str, default='',
                        help='path to infos to evaluate')
        # Basic options
        parser.add_argument('--batch_size', type=int, default=0,
                        help='if > 0 then overrule, otherwise load from checkpoint.')
        parser.add_argument('--num_images', type=int, default=-1,
                        help='how many images to use when periodically evaluating the loss? (-1 = all)')
        parser.add_argument('--language_eval', type=int, default=0,
                        help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
        parser.add_argument('--dump_images', type=int, default=1,
                        help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
        parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
        parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')

        # Sampling options
        parser.add_argument('--sample_max', type=int, default=1,
                        help='1 = sample argmax words. 0 = sample from distributions.')
        parser.add_argument('--beam_size', type=int, default=2,
                        help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
        parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
        # For evaluation on a folder of images:
        parser.add_argument('--image_folder', type=str, default='', 
                        help='If this is nonempty then will predict on the images in this folder path')
        parser.add_argument('--image_root', type=str, default='', 
                        help='In case the image paths have to be preprended with a root path to an image folder')
        # For evaluation on MSCOCO images from some split:
        parser.add_argument('--input_fc_dir', type=str, default='',
                        help='path to the h5file containing the preprocessed dataset')
        parser.add_argument('--input_att_dir', type=str, default='',
                        help='path to the h5file containing the preprocessed dataset')
        parser.add_argument('--input_label_h5', type=str, default='',
                        help='path to the h5file containing the preprocessed dataset')
        parser.add_argument('--input_json', type=str, default='', 
                        help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
        parser.add_argument('--split', type=str, default='test', 
                        help='if running on MSCOCO images, which split to use: val|test|train')
        parser.add_argument('--coco_json', type=str, default='', 
                        help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
        # misc
        parser.add_argument('--id', type=str, default='', 
                        help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')

        opt = parser.parse_args()

        self.opt = opt
        self.opt.model=model_path
        self.opt.infos_path=infos_path

        with open(opt.infos_path) as f:
            infos = cPickle.load(f)

        # override and collect parameters
        if len(opt.input_fc_dir) == 0:
            opt.input_fc_dir = infos['opt'].input_fc_dir
            opt.input_att_dir = infos['opt'].input_att_dir
            opt.input_label_h5 = infos['opt'].input_label_h5
        if len(opt.input_json) == 0:
            opt.input_json = infos['opt'].input_json
        if opt.batch_size == 0:
            opt.batch_size = infos['opt'].batch_size
        if len(opt.id) == 0:
            opt.id = infos['opt'].id
        ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval"]
        for k in vars(infos['opt']).keys():
            if k not in ignore:
                if k in vars(opt):
                    assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
                else:
                    vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

        vocab = infos['vocab'] # ix -> word mapping
        self.infos = infos
        # Setup the model
        model = models.setup(opt)
        self.model=model
        model.load_state_dict(torch.load(opt.model))
        model.cuda()
        model.eval()



    def extractFrames(self,video_path):
        # Create a temp dir for the video
        directory= os.path.join(args.tmp_dir, os.path.basename(video_path).split(".")[0] ) 
        print directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.opt.image_folder= directory
        import subprocess
        ffmpeg = "ffmpeg -i "+video_path+" -vf fps="+str(args.fps_rate)+" "+directory+"/img%d.jpg"
        p = subprocess.Popen(ffmpeg, stdout=subprocess.PIPE, shell=True)
        (output, err) = p.communicate()  
        p_status = p.wait()
        files = os.listdir(directory)
        timestamps = {}
        for f in files:
            frame_number = int(f.split(".jpg")[0].split("img")[1])
            if frame_number >3:
                frame_number = frame_number - args.shift
            time_stamp = int(frame_number  * 1/args.fps_rate )
            timestamps[time_stamp]=[]
        self.opt.num_images = len(timestamps.keys())+1
        captions = self.evaluate_model(self.opt)
        self.opt.video_path=None
        objects= self.extractObjects(directory)
        documents=[]
        for index,i in enumerate(captions):
            document = (video_path, int( int(i['image_id']) * 1/args.fps_rate) , i['caption']+" : "+  objects[index])
            documents.append(document)
        print documents
        self.extractObjects(directory)
        shutil.rmtree(directory)
        return documents

    
    def extractCaptions(self,video_path):
        pass


    def extractVideoText(self,video_path):
        pass


    def extractObjects(self,video_path):
        import os
        import cv2
        import torch
        import numpy as np
        from torch.multiprocessing import Pool

        from darknet import Darknet19
        import utils.yolo as yolo_utils
        import utils.network as net_utils
        from utils.timer import Timer
        import cfgs.config as cfg


        def preprocess(fname):
            # return fname
            image = cv2.imread(fname)
            im_data = np.expand_dims(yolo_utils.preprocess_test((image, None, cfg.inp_size))[0], 0)
            return image, im_data


        # hyper-parameters
        # npz_fname = 'models/yolo-voc.weights.npz'
        # h5_fname = 'models/yolo-voc.weights.h5'
        trained_model = cfg.trained_model
        # trained_model = os.path.join(cfg.train_output_dir, 'darknet19_voc07trainval_exp3_158.h5')
        thresh = 0.5
        im_path = video_path
        # ---

        net = Darknet19()
        net_utils.load_net(trained_model, net)
        # net.load_from_npz(npz_fname)
        # net_utils.save_net(h5_fname, net)
        net.cuda()
        net.eval()
        print('load model succ...')

        t_det = Timer()
        t_total = Timer()
        # im_fnames = ['person.jpg']
        im_fnames = sorted([fname for fname in sorted(os.listdir(im_path)) if os.path.splitext(fname)[-1] == '.jpg'])
        im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
        objectDetect=[]
        for i, (image) in enumerate( im_fnames):
            t_total.tic()
            im_data = preprocess(image)
            image=im_data[0]
            im_data=im_data[1]
            im_data = net_utils.np_to_variable(im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
            t_det.tic()
            bbox_pred, iou_pred, prob_pred = net(im_data)
            det_time = t_det.toc()
            # to numpy
            bbox_pred = bbox_pred.data.cpu().numpy()
            iou_pred = iou_pred.data.cpu().numpy()
            prob_pred = prob_pred.data.cpu().numpy()

            # print bbox_pred.shape, iou_pred.shape, prob_pred.shape
            bboxes, scores, cls_inds = yolo_utils.postprocess(bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)
            objectDetect.append( ','.join(set( [cfg.label_names[i] for i in cls_inds] ) ) )
        return objectDetect


    def getFeatures():
        pass

    def evaluate_model(self,opt):
        model= self.model
        crit = utils.LanguageModelCriterion()
        # Create the Data Loader instance
        if len(opt.image_folder) == 0:
          loader = DataLoader(opt)
        else:
          loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                                    'coco_json': opt.coco_json,
                                    'batch_size': opt.batch_size,
                                    'cnn_model': opt.cnn_model})
        # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
        # So make sure to use the vocab in infos file.
        loader.ix_to_word = self.infos['vocab']
        
        # Set sample options
        loss, split_predictions, lang_stats = eval_utils.eval_split(model, crit, loader, 
            vars(opt))
        
        return split_predictions


vf = VideoFeatureExtractor('/home/sanket/repositories/video_extract/model-best.pth', '/home/sanket/repositories/video_extract/infos.pkl' )
video_dir="/home/sanket/repositories/video_extract/custom"
documents=[]


for i in os.listdir(video_dir):
    tf = open('captions.txt','a')
    tf.write( str(vf.extractFrames(os.path.join(video_dir,i)) )+"\n" )
    tf.close()
