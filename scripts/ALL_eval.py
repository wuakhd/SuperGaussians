
import os
from argparse import ArgumentParser

mipnerf360_outdoor_scenes =  ["bicycle", "flowers", "garden", "stump", "treehill"] #
mipnerf360_indoor_scenes =["room", "counter", "kitchen","bonsai"] #
nerf_scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
tanks_and_temples_scenes = ["truck", "train"]


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", action="store_true")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
# all_scenes.extend(mipnerf360_indoor_scenes)
# all_scenes.extend(tanks_and_temples_scenes)
# all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", action="store_true")
    parser.add_argument('--nerf', "-nerf", action="store_true")
    parser.add_argument('--dtu', "-dtu", action="store_true")
    parser.add_argument("--tanksandtemples", "-tat", action="store_true")
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval"
    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/tat/" + scene + common_args)
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/m360/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/m360/" + scene + common_args)

    common_args = " --eval --quiet"
    for scene in nerf_scenes:
        source = args.nerf + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/nerf/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/nerf/" + scene + common_args)
    common_args = " -r 2"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/DTU/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/DTU/" + scene + common_args)

    
if not args.skip_rendering:
    common_args = " --quiet --eval --skip_train --skip_mesh"
    # common_args = " --quiet --eval --skip_train --render_path"


    for scene in tanks_and_temples_scenes:
        source = args.tanksandtemples + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/tat/" + scene + common_args)
        os.system("CUDA_VISIBLE_DEVICES=5 python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/tat/" + scene + common_args)
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("CUDA_VISIBLE_DEVICES=5 python render.py --iteration 30000 -s " + source + " -i images_4 -m " + args.output_path + "/m360/" + scene + common_args)
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene
        os.system("CUDA_VISIBLE_DEVICES=5 python render.py --iteration 30000 -s " + source + " -i images_2 -m " + args.output_path + "/m360/" + scene + common_args)

    common_args = " --eval --quiet"
    for scene in nerf_scenes:
        source = args.nerf + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/nerf/" + scene + common_args)
        os.system("CUDA_VISIBLE_DEVICES=5 python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/nerf/" + scene + common_args)
    common_args = " -r 2"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/DTU/" + scene + common_args)
        os.system("CUDA_VISIBLE_DEVICES=5 python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/DTU/" + scene + common_args)