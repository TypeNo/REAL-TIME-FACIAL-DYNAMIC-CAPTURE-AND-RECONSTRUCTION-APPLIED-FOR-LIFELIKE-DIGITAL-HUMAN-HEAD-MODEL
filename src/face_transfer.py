# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np
from time import time
import argparse
import torch
import imageio
from skimage.transform import rescale

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
import obj2glb
import mybridge

def main(args, transfer_codedict = None):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    if(transfer_codedict == None):
    # load test image
        testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
        i = 0
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]

    # initialize DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device)

    with torch.no_grad():
        # Encode identity
        if(transfer_codedict == None):
            id_codedict = deca.encode(images)
        else:
            id_codedict = transfer_codedict

        # Prepare animation with varying expression
        exp_dim = deca_cfg.model.n_exp
        num_frames = exp_dim+1  # total number of expression animations
        pad_width = len(str(num_frames - 1))  # e.g., 100 -> 3 digits
        visdict_list = []

        # Create frame directory
        frame_dir = os.path.join(savefolder, 'expression_frames')
        os.makedirs(frame_dir, exist_ok=True)
        model_dir = os.path.join(savefolder, 'Predefined_Model')
        
        gif_path = os.path.join(frame_dir, 'expression_sweep.gif')
        writer = imageio.get_writer(gif_path, mode='I')

        for j in range(num_frames):
            frame_name = f'expression_{j:0{pad_width}d}'
            if j == 0:
                # Frame 0: original
                codedict = id_codedict
            else:
                # Frames 1–50: zero pose, expression j-1 activated
                codedict = id_codedict.copy()
                codedict['pose'][:, 3:] = torch.zeros_like(codedict['pose'][:, 3:])  # neutral neck pose
                codedict['exp'] = torch.zeros_like(codedict['exp'])  # zero expression
                codedict['exp'][:, j - 1] = 2.0  # activate expression j-1e
            opdict, visdict = deca.decode(codedict)
            visdict_list.append(visdict)
            
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, frame_name +'.obj')
            deca.save_obj(model_path, opdict)
    
            # Visualize and save frame
            image = deca.visualize({'transferred_shape': visdict['shape_detail_images']})
            image_rescaled = rescale(image, 0.6, channel_axis=-1)
            frame_bgr = (image_rescaled[:, :, [2, 1, 0]] * 255).astype(np.uint8)

            # Save frame image
            frame_path = os.path.join(frame_dir, frame_name +'.png')
            cv2.imwrite(frame_path, image)

            # Add to GIF
            writer.append_data(frame_bgr)

        writer.close()
        print(f'-- Expression animation saved to {gif_path}')
        print(f'-- Individual frames saved to {frame_dir}')

    frame_dir = os.path.join(savefolder, 'expression_models')
    output_glb = os.path.join(savefolder, 'animation', 'manual_animation.glb')

    obj2glb.main(model_dir, frame_dir, output_glb, 30.0)
    
    mybridge.update_expression_path(output_glb)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Expression Sweep Demo')
    parser.add_argument('-i', '--image_path', default='output/video6140796694809285689/inputs/video6140796694809285689_frame0000_inputs.jpg', type=str)
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--rasterizer_type', default='pytorch3d', type=str)
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'])
    parser.add_argument('--detector', default='fan', type=str)
    parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'])
    main(parser.parse_args())