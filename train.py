# Copyright (C) 2025, TSGaussian
# TSGaussian research group, https://github.com/leon2000-ai/TSGaussian
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_cls_3d, loss_depth_smoothness, patch_norm_mse_loss, patch_norm_mse_loss_global, pearson_depth_loss_0, local_pearson_loss
from gaussian_renderer import render, network_gui, render_for_depth, render_for_opa
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json

###
from torchvision import transforms
import open3d as o3d
import cv2
import random
import numpy as np
import lpips
lpips_model = lpips.LPIPS(net='alex')
lpips_model = lpips_model.to('cuda')

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 2024
seed_everything(seed)

def check_out(iteration, image, gt_image, depth, depth_mono, gaussians, opt, mask):
    if iteration % 500==0:
        toPIL = transforms.ToPILImage()
        mask = mask.float()*10
        mask_pic = toPIL(mask)
        mask_pic.save(f'./output/render_mask_{iteration}.jpg')
        pic = toPIL(image)
        pic.save(f'./output/render_{iteration}.jpg')
        print("Save render")
        pic1 = toPIL(gt_image)
        pic1.save(f'./output/gt_{iteration}.jpg')
        print("Save gt")
        tensor_render_depth = depth.to('cpu').detach()
        tensor_min_render_depth = tensor_render_depth.min()
        tensor_max_render_depth = tensor_render_depth.max()
        normalized_tensor_render_depth = (tensor_render_depth - tensor_min_render_depth) / (tensor_max_render_depth - tensor_min_render_depth)
        scaled_tensor_render_depth = normalized_tensor_render_depth * 255
        array_render_depth = scaled_tensor_render_depth.numpy()
        # array = (array * 255).astype(np.uint8)
        if len(array_render_depth.shape) == 3 and array_render_depth.shape[0] == 1:
            array_render_depth = array_render_depth.squeeze(0)

        cv2.imwrite(f'./output/depth_{iteration}.jpg', array_render_depth)
        print("Save depth")
        tensor_mono_depth = depth_mono.to('cpu').detach()
        tensor_min_mono_depth = tensor_mono_depth.min()
        tensor_max_mono_depth = tensor_mono_depth.max()
        normalized_tensor_mono_depth = (tensor_mono_depth - tensor_min_mono_depth) / (tensor_max_mono_depth - tensor_min_mono_depth)
        scaled_tensor_mono_depth = normalized_tensor_mono_depth * 255
        array_mono_depth = scaled_tensor_mono_depth.numpy()
        # array = (array * 255).astype(np.uint8)
        if len(array_mono_depth.shape) == 3 and array_mono_depth.shape[0] == 1:
            array_mono_depth = array_mono_depth.squeeze(0) 

        cv2.imwrite(f'./output/depth_mono_{iteration}.jpg', array_mono_depth)
        print("Save monodepth")

        point_cloud_tensor = gaussians._xyz.detach()
        point_cloud_np = point_cloud_tensor.cpu().numpy()

        # 指定保存的文件路径
        file_path = f'./output/pointcloud_{iteration}.txt'

        with open(file_path, 'w') as file:
            for point in point_cloud_np:
                file.write(f'{point[0]} {point[1]} {point[2]}\n')

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, use_wandb, value_to_mask):
    first_iter = 0
    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    classifier.cuda()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    patch_range = (5, 17)

    for iteration in range(first_iter, opt.iterations + 1):
        if iteration == opt.iterations:
            print((gaussians._objects_dc).shape)        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        gt_obj = viewpoint_cam.objects.cuda().long()
        indices_to_zero_out = ~(torch.isin(gt_obj,  torch.tensor(value_to_mask).cuda()))
        gt_obj[indices_to_zero_out] = 0
        mask_1f = gt_obj == 0
        mask = mask_1f.unsqueeze(0).expand(3, -1, -1)


        gt_image = viewpoint_cam.original_image.cuda()
        gt_image[mask] = 0

        # -------------------------------------------------- DEPTH --------------------------------------------
        if iteration > opt.hard_depth_start:
            render_pkg_for_depth = render_for_depth(viewpoint_cam, gaussians, pipe, background) #
            depth = render_pkg_for_depth["depth"]
            loss_hard = 0
            # Depth loss
            depth_mono = 255 - viewpoint_cam.depth_mono
            
            #
            depth_mono[mask_1f.unsqueeze(0)] = 0
            #
            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance) ### 局部计算损失
            loss_hard += 0.1 * loss_l2_dpt ### 

            
            if iteration > 3000:
                loss_hard += 0.1 * loss_depth_smoothness(depth[None, ...], depth_mono[None, ...]) #

            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance) ### 全局计算损失
            loss_hard += 0.4 * loss_global
            
            loss_hard.backward()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
           
        ###
        # -------------------------------------------------- pnt --------------------------------------------
        if iteration > opt.soft_depth_start :
            render_pkg_for_opa = render_for_opa(viewpoint_cam, gaussians, pipe, background)
            depth, alpha = render_pkg_for_opa["depth"], render_pkg["alpha"]

            # Depth loss
            loss_pnt = 0
            depth_mono = 255.0 - viewpoint_cam.depth_mono
            depth_mono[mask_1f.unsqueeze(0)] = 0

            loss_l2_dpt = patch_norm_mse_loss(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_pnt += 0.1 * loss_l2_dpt

            if iteration > 3000:
                loss_pnt += 0.1 * loss_depth_smoothness(depth[None, ...], depth_mono[None, ...])

            loss_global = patch_norm_mse_loss_global(depth[None,...], depth_mono[None,...], randint(patch_range[0], patch_range[1]), opt.error_tolerance)
            loss_pnt += 0.4 * loss_global
            
            loss_pnt.backward()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
        
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, objects, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"], render_pkg["opacities"]
        Ll1 = l1_loss(image, gt_image) 
        logits = classifier(objects)
        loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes))
        loss_obj_3d = None
        if iteration % opt.reg3d_interval == 0:
            # regularize at certain intervals
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d_3dmask = torch.softmax(logits3d,dim=0)
            prob_obj3d = prob_obj3d_3dmask.squeeze().permute(1,0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj + loss_obj_3d
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj
        if iteration > opt.hard_depth_start:
            check_out(iteration, image, gt_image, depth, depth_mono, gaussians, opt, mask)
        loss.backward()
        iter_end.record()

        with torch.no_grad():

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb, value_to_mask)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                if iteration > opt.densify_from_iter:
                    if iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, classifier, iteration, value_to_mask)
                        if iteration >= 3000:
                            gaussians.semantic_mask_and_prune(classifier, value_to_mask)
                        # 3dmask prune

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                ### 优化器
                cls_optimizer.step()
                cls_optimizer.zero_grad()
                ### 

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb, value_to_mask):
    
    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "iter_time": elapsed, "iter": iteration})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    gt_obj = viewpoint.objects.cuda().long()

                    indices_to_zero_out = ~(torch.isin(gt_obj,  torch.tensor(value_to_mask).cuda()))
                    gt_obj[indices_to_zero_out] = 0
                    mask_1f = gt_obj == 0
                    mask = mask_1f.unsqueeze(0).expand(3, -1, -1)
                    gt_image[mask] = 0

                    if use_wandb:
                        if idx < 5:
                            wandb.log({config['name'] + "_view_{}/render".format(viewpoint.image_name): [wandb.Image(image)]})
                            if iteration == testing_iterations[0]:
                                wandb.log({config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name): [wandb.Image(gt_image)]})
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips_model(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])   
                ssim_test /= len(config['cameras'])   
                lpips_test /= len(config['cameras'])   
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test, config['name'] + "/loss_viewpoint - SSIM": ssim_test, config['name'] + "/loss_viewpoint - LPIPS": lpips_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6059)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2_000, 5_000, 7_000, 9_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000, 5_000, 7_000, 9_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

    parser.add_argument("--value_to_mask", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)
    
    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 200)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    ###

    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="exp")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)/sc  
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    value_to_mask = list(map(int, args.value_to_mask.split(",")))
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.use_wandb, value_to_mask)

    # All done
    print("\nTraining complete.")
