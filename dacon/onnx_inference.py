import psutil
import numpy as np
from utils_np import *
import matplotlib.pyplot as plt
import os
import onnxruntime as ort
import cv2

class KritaDACoNSingleDataset:
    
    def __init__(self, data_list, data_root, is_ref = True):
        self.data_list = data_list
        self.data_root = data_root
        self.seg_size = None
        self.is_ref = is_ref

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line_name = self.data_list[idx][0]
        color_name = self.data_list[idx][1]
        frame_name = self.data_list[idx][2]

        frame_name_ext = frame_name
        frame_name = os.path.splitext(frame_name)[0]
        
        if self.is_ref:
            color_image_path = os.path.join(self.data_root, color_name, "ref", frame_name_ext)
            line_image_path = os.path.join(self.data_root, line_name, "ref", frame_name_ext)
            seg_path = os.path.join(self.data_root, line_name, "seg", "ref", frame_name_ext)
            json_color_path = os.path.join(self.data_root, line_name, "seg", f"{frame_name}.json")
            color_image = get_image(color_image_path)
        else:
            line_image_path = os.path.join(self.data_root, line_name, "target", frame_name_ext)
            seg_path = os.path.join(self.data_root, line_name, "seg", frame_name_ext)
            json_color_path = os.path.join(self.data_root, line_name, "seg", f"{frame_name}.json")
            color_image = np.zeros((0,), dtype=np.float32)
            json_color_path = None
        
        line_image = get_image(line_image_path)
        
        seg_num, seg_sizes, seg_colors, seg_coords, seg_image = get_seg_info(seg_path, json_color_path, self.seg_size)
      
        H, W = seg_image.shape
        seg_colors = normalize_color(seg_colors)
        seg_coords = normalize_coordinate(seg_coords, (H, W))

        return {
            "line_name": line_name,
            "color_name": color_name,
            "frame_name": frame_name,
            "color_image": color_image,
            "line_image": line_image,
            "seg_num": seg_num,
            "seg_sizes": seg_sizes,
            "seg_colors": seg_colors,
            "seg_coords": seg_coords,
            "seg_image": seg_image,
        }

class DACoNONNXModel:
    def __init__(self, dacon_config="../configs/krita-inference.yaml"):

        self.sess_options = ort.SessionOptions()
        self.sess_options.intra_op_num_threads = psutil.cpu_count(logical=False) or 4
        self.sess_options.inter_op_num_threads = 1
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        dacon_config = load_config(dacon_config)
        network_config = dacon_config['network']
        
        dino_path = "dino.onnx"
        unet_path = "unet.onnx"
        mlp_path = "dino_mlp.onnx"
        
        self.dino_dim = network_config.get("dino_dim", 192)
        self.feats_dim = network_config["feats_dim"]
        
        self.unet_input_size = network_config["unet_input_size"]
        self.dino_input_size = network_config["dino_input_size"]
        self.segment_pool_size = network_config["segment_pool_size"]
        
        self.dino = ort.InferenceSession(
            dino_path,
            sess_options=self.sess_options,
            providers=['CPUExecutionProvider']
        )
        self.unet = ort.InferenceSession(
            unet_path,
            sess_options=self.sess_options,
            providers=['CPUExecutionProvider']
        )
        self.dino_mlp = ort.InferenceSession(
            mlp_path,
            sess_options=self.sess_options,
            providers=['CPUExecutionProvider']
        )
        
        print("✓ DACoN ONNX model initialized")

    def l2_normalize(self, x, axis=-1, eps=1e-6):
        norm = np.linalg.norm(x, ord=2, axis=axis, keepdims=True)
        return x / (norm + eps)
    
    def _prepare_images(self, images, target_size):
        images = images[:, :, 0:3, :, :]
        
        B, S, C, H, W = images.shape
        
        # images = images.reshape(B * S, C, H, W)
        
        H_out, W_out = target_size
        resized = np.zeros((B, S, C, H_out, W_out), dtype=np.float32)
        
        for b in range(B):
            for s in range(S):
                for c in range(C):
                    resized[b, s, c] = cv2.resize(
                        images[b, s, c],
                        (W_out, H_out),
                        interpolation=cv2.INTER_LINEAR
                    )
        
        return resized.astype(np.float32)

    def get_dino_feats_map(self, images):
        images = self._prepare_images(images, self.dino_input_size)
        patch_tokens = self.dino.run(None, {'input': images})[0]
        return patch_tokens
       
    def get_unet_feats_map(self, images):
        images = self._prepare_images(images, self.unet_input_size)
        unet_outputs = self.unet.run(None, {'input': images})[0]
        return unet_outputs

    def dino_dim_reduction(self, seg_dino_feats):
        B, S, L, C = seg_dino_feats.shape
        feats_flat = seg_dino_feats.reshape(B * S * L, C).astype(np.float32)
        reduced = self.dino_mlp.run(None, {'input': feats_flat})[0]
        return reduced.reshape(B, S, L, -1)
    
    def dino_unet_fusion(self, seg_dino_feats, seg_unet_feats):
        B, S, L, C = seg_dino_feats.shape
        
        dino_flat = seg_dino_feats.reshape(B * S * L, C)
        unet_flat = seg_unet_feats.reshape(B * S * L, C)
        
        seg_fusion_feats = self.l2_normalize(dino_flat) + self.l2_normalize(unet_flat)
        
        return seg_fusion_feats

    def get_segment_feats(self, feats_map, seg_image, seg_num):
        seg_feats_src = segment_pooling(
            feats_map, 
            seg_image, 
            seg_num, 
            self.segment_pool_size
        )
        return seg_feats_src

    def get_seg_cos_sim(self, seg_feats_src, seg_feats_tgt):
        # L2 normalize
        seg_feats_src = self.l2_normalize(seg_feats_src, axis=-1)
        seg_feats_tgt = self.l2_normalize(seg_feats_tgt, axis=-1)
        
        # Reshape source
        B, S_src, L_src, C = seg_feats_src.shape
        seg_feats_src = seg_feats_src.reshape(B, S_src * L_src, C)
        
        # Reshape target
        B, S_tgt, L_tgt, C = seg_feats_tgt.shape
        seg_feats_tgt = seg_feats_tgt.reshape(B, S_tgt * L_tgt, C)
        
        # Compute cosine similarity: tgt @ src.T
        # [B, S_tgt*L_tgt, C] @ [B, C, S_src*L_src] → [B, S_tgt*L_tgt, S_src*L_src]
        seg_cos_sim = np.matmul(
            seg_feats_tgt, 
            seg_feats_src.transpose(0, 2, 1)  # Swap last two dims
        )
        
        return seg_cos_sim
    
    def _process_single(self, line_image, seg_image, seg_num):
        # Add batch and sequence dimension: [C, H, W] → [1, 1, C, H, W]
        line_images = np.expand_dims(line_image, axis=0)
        line_images = np.expand_dims(line_images, axis=0)
        seg_images = np.expand_dims(seg_image, axis=0)
        seg_images = np.expand_dims(seg_images, axis=0)
        seg_nums = np.expand_dims(np.array([seg_num]), axis=0)
        
        seg_feats, raw_dino_feats = self._process_multi(
            line_images, 
            seg_images, 
            seg_nums
        )
        
        # Remove sequence dimension: [B, 1, L, C] → [B, L, C]
        return np.expand_dims(seg_feats, axis=0), np.squeeze(raw_dino_feats, axis=1)
    
    def _process_multi(self, line_images, seg_images, seg_nums):
        dino_feats_map = self.get_dino_feats_map(line_images)
        seg_dino_feats = self.get_segment_feats(dino_feats_map, seg_images, seg_nums)
        raw_dino_feats = seg_dino_feats.copy()
        
        unet_feats_map = self.get_unet_feats_map(line_images)
        seg_unet_feats = self.get_segment_feats(unet_feats_map, seg_images, seg_nums)
        
        seg_dino_feats_reduced = self.dino_dim_reduction(seg_dino_feats)
        seg_feats = self.dino_unet_fusion(seg_dino_feats_reduced, seg_unet_feats)
        
        return seg_feats, raw_dino_feats

    def forward(self, data):
        seg_feats_src, seg_dino_feats_src = self._process_single(
            data['line_images_src'], 
            data['seg_images_src'], 
            data['seg_nums_src']
        )
        
        seg_feats_tgt, seg_dino_feats_tgt = self._process_single(
            data['line_images_tgt'], 
            data['seg_images_tgt'], 
            data['seg_nums_tgt']
        )
        
        dino_seg_cos_sim = self.get_seg_cos_sim(
            seg_dino_feats_src, 
            seg_dino_feats_tgt
        )
        seg_cos_sim = self.get_seg_cos_sim(
            seg_feats_src, 
            seg_feats_tgt
        )
        
        return seg_cos_sim, dino_seg_cos_sim

def run_inference(line_name,
                  color_name,
                  data_root,
                  config_path):
    config = load_config(config_path)

    version = config['version']
    
    save_images = config['val']['save_images']
    save_path = data_root
    
    model = DACoNONNXModel(config_path)

    print(f"Extracting Segment and Color")
    check_seg_and_color(data_root, line_name, color_name)
    
    ref_data_list = make_krita_inference_data_list(data_root, line_name, color_name, is_ref = True)
    ref_dataset = KritaDACoNSingleDataset(ref_data_list, data_root, is_ref=True)
    
    all_seg_feats_ref = []
    all_seg_colors_ref = []

    print(f"Processing {len(ref_dataset)} reference images...")

    for i, ref_data in enumerate(ref_dataset):
        seg_colors_ref = ref_data['seg_colors']
        seg_feats_ref, _ = model._process_single(ref_data['line_image'], ref_data['seg_image'], ref_data["seg_num"])
    
        all_seg_feats_ref.append(seg_feats_ref)
        all_seg_colors_ref.append(seg_colors_ref)
    
    all_seg_feats_ref = np.concatenate(all_seg_feats_ref, axis=0)
    all_seg_feats_ref = np.expand_dims(all_seg_feats_ref, axis=0)
    all_seg_colors_ref = np.concatenate(all_seg_colors_ref, axis=0)
    all_seg_colors_ref = np.expand_dims(all_seg_colors_ref, axis=0)
    
    inference_data_list = make_krita_inference_data_list(data_root, line_name, color_name, is_ref = False)
    inference_dataset = KritaDACoNSingleDataset(inference_data_list, data_root, is_ref=False)

    for i, data in enumerate(inference_dataset):
        seg_feats_tgt, _ = model._process_single(data['line_image'], data['seg_image'], data["seg_num"])
        seg_sim_map = model.get_seg_cos_sim(
            all_seg_feats_ref,
            np.expand_dims(seg_feats_tgt, axis=0))
        
        # seg_sim_map = np.squeeze(seg_sim_map, axis=1)
    
        frame_name = data["frame_name"]
        line_image_tgt = data["line_image"] 
        seg_image_tgt = data["seg_image"]
        color_name = data["color_name"]
        line_name = data["line_name"]
    
        nearest_patch_indices = np.argmax(seg_sim_map[i], axis=-1)
    
        color_list_pred = all_seg_colors_ref[i][nearest_patch_indices]
        color_list_pred = color_list_pred * 255
        color_list_pred = color_list_pred[:, :3]
    
        image_pred = colorize_target_image(color_list_pred, line_image_tgt, seg_image_tgt, colors_only=True)
        folder_path = os.path.join(save_path, color_name, "pred")
        os.makedirs(folder_path, exist_ok=True)
        file_path  = os.path.join(folder_path, f"{frame_name}.png")
        pil_image = Image.fromarray((image_pred*255).astype('uint8'), mode="RGBA")
        pil_image.save(file_path)

from onnx_inference import run_inference
import psutil
import threading
import time

class MemoryTracker:
    def __init__(self):
        self.peak_memory = 0
        self.running = False
        self.process = psutil.Process()
        
    def start(self):
        self.running = True
        self.peak_memory = self.process.memory_info().rss / 1024 / 1024
        threading.Thread(target=self._track, daemon=True).start()
    
    def _track(self):
        while self.running:
            current = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current)
            time.sleep(0.01)  # Check every 10ms
    
    def stop(self):
        self.running = False
        return self.peak_memory
