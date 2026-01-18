
import os
import sys
import time
import json
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import DACoNModel, DACoNTinyModel
from data import KritaDACoNSingleDataset, krita_dacon_single_pad_collate_fn
from utils import (
    move_data_to_device,
    load_config,
    format_time,
    colorize_target_image,
    get_folder_names,
    get_file_names,
    extract_segment,
    extract_color,
    make_krita_inference_data_list,
)


def check_seg_and_color(data_root, line_name, color_name):

    # all frames have corresponding lines - get their basenames from lines
    frame_names = get_file_names(os.path.join(data_root, line_name, "target"))
    ref_frame_names = get_file_names(os.path.join(data_root, line_name, "ref"))

    for frame_name in frame_names:
        line_image_path = os.path.join(data_root, line_name, "target", frame_name)
        seg_path = os.path.join(data_root, line_name, "seg", frame_name)
        os.makedirs(os.path.join(data_root, line_name, "seg"), exist_ok=True)

        if not(os.path.isfile(seg_path)):
            extract_segment(line_image_path, seg_path)

    for frame_name in ref_frame_names:
        color_image_path = os.path.join(data_root, color_name, "ref", frame_name)
        line_image_path = os.path.join(data_root, line_name, "ref", frame_name)
        seg_path = os.path.join(data_root, line_name, "seg", "ref", frame_name)
        color_json_path = os.path.join(data_root, line_name, "seg", f"{frame_name.split('.')[0]}.json")
        os.makedirs(os.path.join(data_root, line_name, "seg", "ref"), exist_ok=True)

        if not(os.path.isfile(seg_path)):
            extract_segment(line_image_path, seg_path)
        if not(os.path.isfile(color_json_path)):
            extract_color(color_image_path, seg_path, color_json_path)



def run_inference(line_name, color_name,
                  data_root='./tmp',
                  config='../configs/krita-inference.yaml',
                  model_path='../checkpoints/dacon_krita.pth',
                  ):

    config = load_config(config)

    version = config['version']
    num_workers_val = config['datasets']['val']['num_worker']

    batch_size = 1
    save_images = config['val']['save_images']
    save_json = config['val']['save_json']
    save_path = data_root


    device = torch.device("cuda" if torch.cuda.is_available() and config['num_gpu'] > 0 else "cpu")
    print(f"Using device: {device}")

    model = DACoNTinyModel(config['network'], version).to(device)

    print(f"Loading checkpoint {os.path.basename(model_path)}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)

    
    # --- Dataset and DataLoader setup ---
    print(f"Extracting Segment and Color")
    check_seg_and_color(data_root, line_name, color_name)

    print("\n--- Start Inference ---")
    inference_start_time = time.time()

    model.eval()
    with torch.no_grad():

        ref_data_list = make_krita_inference_data_list(data_root, line_name, color_name, is_ref = True)
        ref_dataset = KritaDACoNSingleDataset(ref_data_list, data_root, is_ref=True, mode = "infer")
        ref_dataloader = DataLoader(ref_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_val, collate_fn=krita_dacon_single_pad_collate_fn)

        all_seg_feats_ref = torch.empty(0, device=device)
        all_seg_colors_ref = torch.empty(0, device=device)

        for i, ref_data in enumerate(ref_dataloader):
            ref_data = move_data_to_device(ref_data, device)
            seg_colors_ref = ref_data['seg_colors']
            seg_feats_ref, _ = model._process_single(ref_data['line_image'], ref_data['seg_image'], ref_data["seg_num"])

            for b in range(batch_size):
                all_seg_feats_ref = torch.cat((all_seg_feats_ref, seg_feats_ref[b]), dim = 0)
                all_seg_colors_ref = torch.cat((all_seg_colors_ref, seg_colors_ref[b]), dim = 0)

            del ref_data, seg_feats_ref
            torch.cuda.empty_cache()

        inference_data_list = make_krita_inference_data_list(data_root, line_name, color_name, is_ref = False)
        inference_dataset = KritaDACoNSingleDataset(inference_data_list, data_root, is_ref=False, mode = "infer")
        inference_dataloader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_val, collate_fn=krita_dacon_single_pad_collate_fn)

        print(f"\n  Inference on {len(inference_dataset)} samples from ref on {color_name}")

        all_seg_feats_ref = all_seg_feats_ref.unsqueeze(0)
        all_seg_feats_ref = all_seg_feats_ref.repeat(batch_size, 1, 1)

        for i, data in enumerate(inference_dataloader):
            data = move_data_to_device(data, device)
            seg_feats_tgt, _ = model._process_single(data['line_image'], data['seg_image'], data["seg_num"])
            seg_sim_map = model.get_seg_cos_sim(all_seg_feats_ref.unsqueeze(1), seg_feats_tgt.unsqueeze(1))
            seg_sim_map = seg_sim_map.squeeze(1)

            frame_name = data["frame_name"]
            line_image_tgt = data["line_image"] 
            seg_image_tgt = data["seg_image"]
            color_name = data["color_name"]
            line_name = data["line_name"]

            for b in range(batch_size):

                seg_sim_map_batch = seg_sim_map[b]
                nearest_patch_indices = torch.argmax(seg_sim_map_batch, dim=-1)

                color_list_pred = all_seg_colors_ref[nearest_patch_indices]
                color_list_pred = color_list_pred * 255

                if save_images:
                    image_pred = colorize_target_image(color_list_pred, line_image_tgt[b], seg_image_tgt[b], colors_only=True)
                    folder_path = os.path.join(save_path, color_name[b], "pred")
                    os.makedirs(folder_path, exist_ok=True)
                    file_path  = os.path.join(folder_path, f"{frame_name[b]}.png")
                    image_pred = image_pred.permute(2, 0, 1)
                    save_image(image_pred, file_path)

                if save_json:
                    folder_path = os.path.join(save_path, color_name[b], "pred")
                    os.makedirs(folder_path, exist_ok=True)
                    json_file_path  = os.path.join(folder_path, f"{frame_name[b]}.json")
                    color_dict = {str(idx + 1): [int(value) for value in color.tolist()] for idx, color in enumerate(color_list_pred)}

                    with open(json_file_path, "w") as json_file:
                        json.dump(color_dict, json_file)

            print(f"  Sample {i+1}/{len(inference_dataset)}", end='\r')

        del data, seg_feats_tgt, seg_sim_map
        torch.cuda.empty_cache()

    del all_seg_feats_ref
    torch.cuda.empty_cache()

    print("\n--- Inference complete! ---")
    inference_finish_time = time.time()
    print(f"\nTotal time: {format_time(inference_finish_time - inference_start_time)}")

    # Log absolute peak memory usage
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU Memory Usage: {peak_memory_gb:.2f}GB")

