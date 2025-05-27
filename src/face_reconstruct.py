import os, sys

sys.path.append("E:/Project/DECA3/DECA/build/Release")
import mybridge
#print(mybridge.__file__)
#progress = mybridge.create_progress()
#mybridge.set_global_progress(progress)

import cv2
import numpy as np
import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.getcwd())
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.tensor_cropper import transform_points
import ctypes
import os
from ctypes import cdll
import obj2glb
# Load the DLL (ensure path is correct)
dll_path = os.path.abspath("build/bin/Release/progress_shared.dll")
progress = ctypes.CDLL(dll_path)
dll = ctypes.WinDLL("progress_shared.dll")

# Call update_progress
progress.update_progress.argtypes = [ctypes.c_int, ctypes.c_int]
#progress.update_progress(7, 15)
#print("[Python] update_progress(7, 15) called")

progress_lib = cdll.LoadLibrary("build/bin/Release/progress_shared.dll")
progress_lib.update_progress.argtypes = [ctypes.c_int, ctypes.c_int]
progress_lib.update_progress.restype = None

progress_lib.get_current_progress.restype = ctypes.c_int
progress_lib.get_total_progress.restype = ctypes.c_int

# Wait for C++ app to check
time.sleep(5)

path_to_check = 'E:\\Project\\DECA3\\DECA\\build\\bin\\Release'
is_in_path = path_to_check in sys.path

print(f'Is {path_to_check} in the path: {is_in_path}')

progress_status = {"total": 0, "done": 0}

def set_progress(total, done):
    global progress_status
    progress_status["total"] = total
    progress_status["done"] = done

def get_progress():
    return progress_status

def main(args):
    print("face_reconstruct main() called with args:", args)
    print("sys.argv:", sys.argv)
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector, sample_step=args.sample_step)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.extract_tex = args.extractTex
    deca = DECA(config = deca_cfg, device=device)
    
    for i in tqdm(range(len(testdata))):
        set_progress(len(testdata), i)
        try:
            print(f"Updating progress: {i} / {len(testdata)}")
            #progress = mybridge.create_progress()
            #progress.update(i, len(testdata))
            mybridge.update_progress(i, len(testdata))
            progress_lib.update_progress(i, len(testdata))
            dll.update_progress(i, len(testdata))
        except Exception as e:
            print(f"Error updating progress: {e}")
        print(f"Python Progress: {progress_status['done']} / {progress_status['total']}")
        inputname = testdata[i]['imageinputname']
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]   
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor
            if args.render_orig:
                tform = testdata[i]['tform'][None, ...]
                tform = torch.inverse(tform).transpose(1,2).to(device)
                original_image = testdata[i]['original_image'][None, ...].to(device)
                _, orig_visdict = deca.decode(codedict, render_orig=True, original_image=original_image, tform=tform)    
                orig_visdict['inputs'] = original_image 

            #if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
                #os.makedirs(os.path.join(savefolder, name), exist_ok=True)
            # -- save results
            if args.saveDepth:
                depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1, 3, 1, 1)
                visdict['depth_images'] = depth_image

                # Define the output path
                depth_output_dir = os.path.join(savefolder, inputname, 'depth_images')
                os.makedirs(depth_output_dir, exist_ok=True)  # Create the folder if it doesn't exist

                # Save the depth image
                cv2.imwrite(os.path.join(depth_output_dir, f'{name}_depth.jpg'), util.tensor2image(depth_image[0]))
            if args.saveKpt:
                # Create output directories
                kpt2d_dir = os.path.join(savefolder, inputname, 'Kpt2d')
                kpt3d_dir = os.path.join(savefolder, inputname, 'Kpt3d')
                os.makedirs(kpt2d_dir, exist_ok=True)
                os.makedirs(kpt3d_dir, exist_ok=True)

                # Save the 2D and 3D keypoints
                np.savetxt(os.path.join(kpt2d_dir, f'{name}_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
                np.savetxt(os.path.join(kpt3d_dir, f'{name}_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())

            if args.saveObj:
                # Create the 'model' directory
                model_dir = os.path.join(savefolder, inputname, 'Model')
                os.makedirs(model_dir, exist_ok=True)

                # Save the .obj file inside the 'model' folder
                deca.save_obj(os.path.join(model_dir, f'{name}.obj'), opdict)
            if args.saveMat:
                # Create the 'matlab_file' directory
                mat_dir = os.path.join(savefolder, inputname, 'Matlab_File')
                os.makedirs(mat_dir, exist_ok=True)
                opdict = util.dict_tensor2npy(opdict)
                savemat(os.path.join(mat_dir, name + '.mat'), opdict)
            if args.saveVis:
                vis_dir = os.path.join(savefolder, inputname, 'Visualization')
                cv2.imwrite(os.path.join(vis_dir, name + '_vis.jpg'), deca.visualize(visdict))
                if args.render_orig:
                    cv2.imwrite(os.path.join(vis_dir, name + '_vis_original_size.jpg'), deca.visualize(orig_visdict))
            if args.saveImages:
                for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                    if vis_name not in visdict.keys():
                        continue

                    # Save normal render
                    vis_folder = os.path.join(savefolder, inputname, vis_name)
                    os.makedirs(vis_folder, exist_ok=True)

                    image = util.tensor2image(visdict[vis_name][0])
                    cv2.imwrite(os.path.join(vis_folder, f'{name}_{vis_name}.jpg'), image)

                    # Save original render in separate folder if enabled
                    if args.render_orig:
                        orig_folder = os.path.join(savefolder, inputname, "original", vis_name)
                        os.makedirs(orig_folder, exist_ok=True)

                        orig_image = util.tensor2image(orig_visdict[vis_name][0])
                        cv2.imwrite(os.path.join(orig_folder, f'{name}_{vis_name}.jpg'), orig_image)

        set_progress(len(testdata), i+1)
        print(f"Python Progress: {progress_status['done']} / {progress_status['total']}")
        #progress.update(i+1, len(testdata)) 
        mybridge.update_progress(i+1, len(testdata))
        #progress_lib.update_progress(i+1, len(testdata))
        #dll.update_progress(i+1, len(testdata))

    print(f'-- please check the results in {savefolder}')

    frame_dir = os.path.join(savefolder, inputname, 'frames_model')
    output_glb = os.path.join(savefolder, inputname, 'animation', 'dynamic_animation.glb')
    
    obj2glb.main(model_dir, frame_dir, output_glb)
        


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

        parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                            help='path to the test data, can be image folder, image path, image list, video')
        parser.add_argument('-s', '--savefolder', default='output', type=str,
                            help='path to the output directory, where results(obj, txt files) will be stored.')
        parser.add_argument('--device', default='cuda', type=str,
                            help='set device, cpu for using cpu' )
        # process test images
        parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to crop input image, set false only when the test image are well cropped' )
        parser.add_argument('--sample_step', default=10, type=int,
                            help='sample images from video data for every step' )
        parser.add_argument('--detector', default='fan', type=str,
                            help='detector for cropping face, check decalib/detectors.py for details' )
        # rendering option
        parser.add_argument('--rasterizer_type', default='pytorch3d', type=str,
                            help='rasterizer type: pytorch3d or standard' )
        parser.add_argument('--render_orig', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to render results in original image size, currently only works when rasterizer_type=standard')
        # save
        parser.add_argument('--useTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to use FLAME texture model to generate uv texture map, \
                                set it to True only if you downloaded texture model' )
        parser.add_argument('--extractTex', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode' )
        parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save visualization of output' )
        parser.add_argument('--saveKpt', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save 2D and 3D keypoints' )
        parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save depth image' )
        parser.add_argument('--saveObj', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                                Note that saving objs could be slow' )
        parser.add_argument('--saveMat', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save outputs as .mat' )
        parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'],
                            help='whether to save visualization output as seperate images' )
        main(parser.parse_args())

    except Exception as e:
        import traceback
        print("⚠️ Exception caught at top level:")
        traceback.print_exc()
        raise