import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from CLIP import clip
from depth_anything.dpt import DepthAnything

from dataloaders.dataset_test import (
    SUNRGBDDataset, iBims, HyperSim, DIODE,
    change_to_sunrgbd,
    change_to_ibims,
    change_to_hypersim,
    change_to_diode_outdoor,
)
from dataloaders.dataset_simcol import SimCol, change_to_simcol

from eval_indomain import recover_image
from options import MonodepthOptions
from scalemap_depth import ScaleMapModel
from utils import compute_errors, setup_tee_logging, visualize_eval

EPS = 1e-8

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

options = MonodepthOptions()
args = options.parse()


def get_text(txt_path):
    """Load all-text file (one description per line) for zero-shot eval."""
    text_list = []
    with open(txt_path, 'r') as file:
        for line in file:
            text_list.append(line)
    return text_list


def eval(Scalemap_model, depth_model, CLIP_model, Image_f_model, dataloader_eval,
         txt_path=None, vis_save_path=None, post_process=False, dataset=None):
    eval_measures = torch.zeros(10).cuda()

    if dataset != 'simcol':
        text_list = get_text(txt_path)

    for step, eval_sample_batched in tqdm(enumerate(dataloader_eval), file=sys.__stderr__):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            depth_h, depth_w = gt_depth.shape[2], gt_depth.shape[3]

            if dataset != 'simcol':
                text = text_list[step][:-1]
            else:
                text = eval_sample_batched['text']

            text_tokens = clip.tokenize(text, truncate=True).to("cuda")

            image_h, image_w = image.shape[2], image.shape[3]
            if "da" in args.depth_model:
                if dataset == 'sunrgbd':
                    a, b = 518, 728
                elif dataset in ('ibims', 'diode_outdoor', 'hypersim', 'simcol'):
                    a, b = 479 - 45, 603 - 43
                else:
                    a, b = image_h, image_w
                image = torch.nn.functional.interpolate(
                    image, size=(a, b), mode="bilinear", align_corners=True,
                )

            text_features = CLIP_model.encode_text(text_tokens).unsqueeze(1)
            image_features = Image_f_model.get_intermediate_layers(
                image, n=1, return_class_token=False
            )[0]

            scale_pred, shift_pred, _, _ = Scalemap_model(
                image_features.float(), text_features.float(), int(a / 14), int(b / 14)
            )

            relative_depth = depth_model(image).unsqueeze(1)
            pred_depth = 1 / (scale_pred * relative_depth + shift_pred)

            if "da" in args.depth_model or args.depth_model == "midas":
                pred_depth = torch.nn.functional.interpolate(
                    pred_depth, size=(depth_h, depth_w), mode="bilinear", align_corners=True,
                )

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            relative_depth = relative_depth.cpu().numpy().squeeze()

            if vis_save_path is not None:
                image_vis = recover_image(image.squeeze())
                scale_pred_np = scale_pred.cpu().numpy().squeeze()
                shift_pred_np = shift_pred.cpu().numpy().squeeze()
                sample_path = eval_sample_batched['image_path'][0].replace('/', '_')

                if dataset == 'ibims':
                    s = 1
                elif dataset == 'diode_outdoor':
                    s = 5
                else:
                    s = 50
                if step % s == 0:
                    visualize_eval(image_vis, gt_depth, relative_depth, None, pred_depth,
                                   scale_pred_np, shift_pred_np,
                                   args.min_depth_eval, args.max_depth_eval, image_h, image_w,
                                   sample_path, vis_save_path)

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        if dataset == 'diode_outdoor':
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                      int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
        elif dataset == 'sunrgbd' or dataset == 'ibims':
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)
            eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
        else:
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])
        if not np.isnan(measures).any():
            eval_measures[:9] += torch.tensor(measures).cuda()
            eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)),
          ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.3f}'.format(eval_measures_cpu[8]))
    print()

    return eval_measures_cpu


def main(args):
    if args.log_file:
        from datetime import datetime
        log_path = args.log_file
        if log_path == 'auto':
            log_path = f"logs/eval_zero_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        setup_tee_logging(log_path)

    torch.cuda.empty_cache()
    cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    change_to_sunrgbd(args)
    dataloader_eval_sunrgbd = torch.utils.data.DataLoader(
        SUNRGBDDataset(args), batch_size=1, shuffle=False, num_workers=1) \
        if getattr(args, 'sunrgbd_root', None) else None

    change_to_ibims(args)
    dataloader_eval_ibims1 = torch.utils.data.DataLoader(
        iBims(args), batch_size=1, shuffle=False, num_workers=1) \
        if getattr(args, 'ibims_root', None) else None

    change_to_diode_outdoor(args)
    dataloader_eval_diode_outdoor = torch.utils.data.DataLoader(
        DIODE(args), batch_size=1, shuffle=False, num_workers=1) \
        if getattr(args, 'diode_root', None) else None

    change_to_hypersim(args)
    dataloader_eval_hypersim = torch.utils.data.DataLoader(
        HyperSim(args), batch_size=1, shuffle=False, num_workers=1) \
        if getattr(args, 'hypersim_root', None) else None

    change_to_simcol(args)
    dataloader_eval_simcol = torch.utils.data.DataLoader(
        SimCol(args), batch_size=1, shuffle=False, num_workers=1) \
        if getattr(args, 'simcol_root', None) else None

    if "da" in args.depth_model:
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        }
        encoder = 'vit' + args.depth_model[-1]
        depth_model = DepthAnything(model_configs[encoder])
        depth_model.load_state_dict(
            torch.load(f'./depth_anything/depth_anything_{encoder}14.pth')
        )
        depth_model.to("cuda").eval()
    else:
        raise ValueError(
            f"Unsupported depth_model: {args.depth_model}. "
            "This release supports DepthAnything (da_s/da_b/da_l)."
        )

    print(f"== Depth Model Initialized : {encoder}")
    Scalemap_model = ScaleMapModel(args.image_channels, args.out_channels, args.text_channels,
                                   args.features, args.cross_attn_nhead,
                                   args.coef_scale, args.coef_shift)
    Scalemap_model.eval()
    Scalemap_model.cuda()
    print("== ScaleMap Model Initialized")

    CLIP_model, _ = clip.load(args.text_encoder, device="cuda")
    Image_f_model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{args.img_encoder}14').to("cuda")
    CLIP_model.eval()
    Image_f_model.eval()

    print(f"== Text Model Initialized : {args.text_encoder}")
    print(f"== Image Model Initialized : {args.img_encoder}")

    if args.load_ckpt_path is not None:
        checkpoint = torch.load(args.load_ckpt_path)
        state_dict = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        Scalemap_model.load_state_dict(state_dict)
        print(f"== Loaded ScaleMap weights from {args.load_ckpt_path}")

    if args.visualize_results:
        vis_root = args.load_ckpt_path + '_vis'
        vis_save_paths = {
            'sunrgbd':       os.path.join(vis_root, 'sunrgbd'),
            'ibims':         os.path.join(vis_root, 'ibims'),
            'hypersim':      os.path.join(vis_root, 'hypersim'),
            'diode_outdoor': os.path.join(vis_root, 'diode_outdoor'),
            'simcol':        os.path.join(vis_root, 'simcol'),
        }
        for p in vis_save_paths.values():
            os.makedirs(p, exist_ok=True)
    else:
        vis_save_paths = {k: None for k in
                          ('sunrgbd', 'ibims', 'hypersim', 'diode_outdoor', 'simcol')}

    print("Evaluating")
    Scalemap_model.eval()

    with torch.no_grad():
        if dataloader_eval_sunrgbd is not None:
            change_to_sunrgbd(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_sunrgbd, args.txt_path_eval, vis_save_paths['sunrgbd'],
                 post_process=False, dataset=args.dataset)

        if dataloader_eval_ibims1 is not None:
            change_to_ibims(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_ibims1, args.txt_path_eval, vis_save_paths['ibims'],
                 post_process=False, dataset=args.dataset)

        if dataloader_eval_hypersim is not None:
            change_to_hypersim(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_hypersim, args.txt_path_eval, vis_save_paths['hypersim'],
                 post_process=False, dataset=args.dataset)

        if dataloader_eval_diode_outdoor is not None:
            change_to_diode_outdoor(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_diode_outdoor, args.txt_path_eval, vis_save_paths['diode_outdoor'],
                 post_process=False, dataset=args.dataset)

        if dataloader_eval_simcol is not None:
            change_to_simcol(args)
            eval(Scalemap_model, depth_model, CLIP_model, Image_f_model,
                 dataloader_eval_simcol, None, vis_save_paths['simcol'],
                 post_process=False, dataset=args.dataset)


if __name__ == '__main__':
    main(args)
