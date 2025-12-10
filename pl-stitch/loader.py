# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import math
import numpy as np
import os
import lmdb
from pathlib import Path
import torch
import glob
from tqdm import tqdm
from typing import Any, Callable, Optional, List, Tuple, Dict
from PIL import Image
import utils
import cv2
import random, collections
import decord
import json
import collections
from collections import defaultdict
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as T



class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base, times):
        self.dataset  = base
        self.times = times
        self._len  = len(base)

    def __len__(self):
        return self._len * self.times

    def __getitem__(self, idx):
        return self.dataset[idx % self._len]


class Dataset_puzzle(VisionDataset):
    def __init__(
        self,
        lmdb_path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        size: int = 224,
    ) -> None:

        self.lmdb = lmdb_path
        self.transform=transform
        self.target_transform=target_transform
        self.size = (size, size)
        self.data = []
        self.targets = []
        self.num_samples = 0
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            print(f"total number is: {txn.stat()['entries']}")
            cursor = txn.cursor()
            for key in tqdm(cursor.iternext(keys=True, values=False), total=txn.stat()['entries']):
                self.data.append(key)
                self.num_samples += 1


    def load_image_from_lmdb(self, img_key):
        """
        Load an image from LMDB using the provided key.
        """
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(img_key)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        return img

    '''
    def get_time_sequence_frame(self, frame_name: str):
        """
        Pick a neighbor frame by trying offsets in random order.
        No membership checks; just LMDB gets. Falls back to current frame.
        """
        # Parse "video_timeindex"
        video_name, t_str = frame_name.rsplit('_', 1)
        t = int(t_str)
        vid_b = video_name.encode('utf-8')

        offsets = np.array([-3, -2, -1, 1, 2, 3], dtype=np.int32)

        with self.env.begin(write=False, buffers=True) as txn:
            # Try neighbors in random order
            for off in np.random.permutation(offsets):
                tt = t + int(off)
                nei_key = vid_b + b'_' + str(tt).encode('ascii')
                buf = txn.get(nei_key)
                if buf is not None:
                    arr = np.frombuffer(memoryview(buf), dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)   # keep your original decoding
                    return Image.fromarray(img)

            # Fallback to current frame
            cur_key = frame_name.encode('utf-8')
            buf0 = txn.get(cur_key)
            if buf0 is None:
                # Shouldn't happen if keys were indexed from LMDB,
                # but guard to avoid crashing.
                raise KeyError(f"LMDB missing key: {frame_name}")
            arr0 = np.frombuffer(memoryview(buf0), dtype=np.uint8)
            img0 = cv2.imdecode(arr0, cv2.IMREAD_COLOR)
            return Image.fromarray(img0)
    '''
    def get_time_sequence_frame(self, frame_name: str):
        video_name, t_str = frame_name.rsplit('_', 1)
        t = int(t_str)
        vid_b = video_name.encode('utf-8')

        past_offsets = np.array([-3, -2, -1])
        future_offsets = np.array([1, 2, 3])
    
        past_frame = None
        future_frame = None
    
        with self.env.begin(write=False, buffers=True) as txn:
            # --- Step 1: Search for a "past" frame ---
            for off in np.random.permutation(past_offsets):
                tt = t + int(off)
                if tt < 0: continue
                
                nei_key = vid_b + b'_' + str(tt).encode('ascii')
                buf = txn.get(nei_key)
                
                if buf is not None:
                    arr = np.frombuffer(memoryview(buf), dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    past_frame = Image.fromarray(img_bgr)
                    break 
            
            # --- Step 2: Search for a "future" frame ---
            for off in np.random.permutation(future_offsets):
                tt = t + int(off)
                nei_key = vid_b + b'_' + str(tt).encode('ascii')
                buf = txn.get(nei_key)
    
                if buf is not None:
                    arr = np.frombuffer(memoryview(buf), dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    future_frame = Image.fromarray(img_bgr)
                    break
    
            # --- Step 3: Load the current frame ONLY if a neighbor was not found ---
            if past_frame is None or future_frame is None:
                cur_key = frame_name.encode('utf-8')
                buf0 = txn.get(cur_key)
                if buf0 is None:
                    raise KeyError(f"LMDB missing the current frame key: {frame_name}")
                
                arr0 = np.frombuffer(memoryview(buf0), dtype=np.uint8)
                img0_bgr = cv2.imdecode(arr0, cv2.IMREAD_COLOR)
                fallback_img = Image.fromarray(img0_bgr)
    
                # Apply fallback only where needed
                if past_frame is None:
                    past_frame = fallback_img
                if future_frame is None:
                    future_frame = fallback_img
                    
        return past_frame, future_frame
    
    def __getitem__(self, index: int):
        img_key = self.data[index]
        frame_name = img_key.decode('utf-8')  # Assuming the keys are encoded as bytes
        original_img = self.load_image_from_lmdb(img_key)

        # Get time sequence frames
        time_sequence_frames = self.get_time_sequence_frame(frame_name)

        if self.transform is not None:
            imgs = self.transform(original_img, time_sequence_frames)

        return imgs

    def __len__(self) -> int:
        return self.num_samples



class Dataset(VisionDataset):
    def __init__(
        self,
        lmdb_path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        size: int = 224,
    ) -> None:

        self.lmdb = lmdb_path
        self.transform=transform
        self.target_transform=target_transform
        self.size = (size, size)
        self.data = []
        self.targets = []
        self.num_samples = 0
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            print(f"total number is: {txn.stat()['entries']}")
            cursor = txn.cursor()
            for key in tqdm(cursor.iternext(keys=True, values=False), total=txn.stat()['entries']):
                self.data.append(key)
                self.num_samples += 1


    def load_image_from_lmdb(self, img_key):
        """
        Load an image from LMDB using the provided key.
        """
        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(img_key)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img = Image.fromarray(img)
        return img
    
    def __getitem__(self, index: int):
        img_key = self.data[index]
        frame_name = img_key.decode('utf-8')  # Assuming the keys are encoded as bytes
        img = self.load_image_from_lmdb(img_key)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self) -> int:
        return self.num_samples




class Video_Dataset(VisionDataset):
    def __init__(
        self,
        video_path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        size: int = 224,
    ) -> None:

        self.videos = video_path
        self.transform=transform
        self.target_transform=target_transform
        self.size = (size, size)
        self.data = []
        self.targets = []
        self.num_samples = 0

        self.data = self._make_dataset(self.videos)
        


    
    def __getitem__(self, index: int):
        video, sec = self.data[index]
        decord_vr = decord.VideoReader(video, num_threads=1)
        fps = decord_vr.get_avg_fps()
        fps_int = int(round(fps))
        duration = len(decord_vr)

        selected_idx = int(sec) * fps_int + np.random.randint(0, fps_int-1)
        if selected_idx >= duration:
            selected_idx = duration - 1
        
        frame_nd = decord_vr[selected_idx].asnumpy()  # (H,W,3), RGB, uint8

        fetched_image = Image.fromarray(frame_nd).convert('RGB')


        if self.transform is not None:
            img = self.transform(fetched_image)
        else:
            img = fetched_image 

        return img

    
    def _make_dataset(self, directory):
        images = []
        exts = (".mp4", ".avi", ".mkv", ".mov")
        if os.path.isdir(directory):
            vids = os.listdir(directory)
            videos = [os.path.join(directory, vid) for vid in vids if vid.lower().endswith(exts)]
        for video in tqdm(videos, total=len(videos)):
            if "lemon" not in directory.lower():
                decord_vr = decord.VideoReader(video, num_threads=1)
                fps = decord_vr.get_avg_fps()
                fps_int = int(round(fps))
                duration = len(decord_vr)

                # last full second that still has a frame
                last_sec = (duration - 1) // fps_int
                for sec in range(last_sec + 1):
                    images.append((video, sec))
            else:
                for sec in range(10):
                    images.append((video, sec))
                    
        return images
    
    
    def __len__(self) -> int:
        return len(self.data)






class kinetics_singleFrame_Dataset(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        img_size: int = 224,
        seq_len: int = 8,
        transform: Optional[T.Compose] = None,
        min_step: int = 1,
        max_step: int = 20,
        cache_path: str = "kinetics_vid2frames_cache.json"
    ):
        super().__init__(lmdb_path, transform=None) 

        assert seq_len > 0, "seq_len must be positive"
        assert 1 <= min_step <= max_step, "Require 1 <= min_step <= max_step"
        self.seq_len = seq_len
        self.min_step = min_step
        self.max_step = max_step
        self.rng = random.Random() # Random number generator

        '''
        # 1) Group all frame keys by video ID
        self.vid2frames: Dict[str, List[int]] = collections.defaultdict(list)
        for key_bytes in getattr(self, "data", []):
            try:
                key_str = key_bytes.decode("utf-8")
                vid, idx_str = key_str.rsplit("_", 1)
                self.vid2frames[vid].append(int(idx_str))
            except (ValueError, IndexError):
                continue

        # 2) Filter and sort frames for each video
        self.videos: List[str] = []
        for vid, frames in self.vid2frames.items():
            # Ensure enough frames for both sequence and neighbor sampling
            if len(frames) >= self.seq_len: 
                self.vid2frames[vid] = sorted(list(set(frames)))
                self.videos.append(vid)
        '''
        # --- CACHING LOGIC STARTS HERE ---
        if cache_path and os.path.exists(cache_path):
            print(f"Loading video-to-frame mapping from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                self.vid2frames = json.load(f)
        else:
            print("Building video-to-frame mapping from scratch (this will take a while)...")
            # 1) Group all frame keys by video ID (the slow part)
            vid2frames_temp = collections.defaultdict(list)
            for key_bytes in tqdm(getattr(self, "data", []), desc="Scanning LMDB keys"):
                try:
                    key_str = key_bytes.decode("utf-8")
                    vid, idx_str = key_str.rsplit("_", 1)
                    vid2frames_temp[vid].append(int(idx_str))
                except (ValueError, IndexError):
                    continue

            # 2) Filter out videos that are too short and sort frame indices
            print("Filtering and sorting videos...")
            self.vid2frames = {}
            for vid, frames in tqdm(vid2frames_temp.items(), desc="Filtering videos"):
                sorted_frames = sorted(list(set(frames)))
                if len(sorted_frames) >= self.seq_len:
                    self.vid2frames[vid] = sorted_frames
            
            # 3) Save the processed dictionary to the cache file for next time
            if cache_path:
                print(f"Saving mapping to cache for future use: {cache_path}")
                with open(cache_path, 'w') as f:
                    json.dump(self.vid2frames, f)
        # --- CACHING LOGIC ENDS HERE ---

        self.videos: List[str] = list(self.vid2frames.keys())
        
        print(f"Initialized dataset with {len(self.videos)} videos.")

        self.transform = transform
        
    def set_seed(self, epoch: int):
        """Seeds the random number generator for reproducibility if needed."""
        self.rng.seed(epoch)

    def __len__(self) -> int:
        return len(self.videos)

    def _get_frame_from_idx(self, vid, frame_idx):
        """Helper to load and transform a single frame by its integer index."""
        key = f"{vid}_{frame_idx}".encode("utf-8")
        img = self.load_image_from_lmdb(key)
        return img # Return PIL image, transform will be applied later

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vid = self.videos[idx]
        idx_list = self.vid2frames[vid]
        n_frames = len(idx_list)


        # --- 2. Sample the Single Frame for iBOT ---
        # Calculate the start and end positions for the [0.3, 0.7] window
        start_pos = int(n_frames * 0.3)
        end_pos = int(n_frames * 0.7)
        
        # Ensure the range is valid for randint (start must be less than end)
        if start_pos >= end_pos:
            end_pos = start_pos + 1 
        
        # Randomly pick an index from the calculated window
        ibot_frame_pos = self.rng.randint(start_pos, end_pos)
        ibot_frame_idx = idx_list[ibot_frame_pos]
        ibot_frame_pil = self._get_frame_from_idx(vid, ibot_frame_idx)

        if self.transform is not None:
            imgs = self.transform(ibot_frame_pil)


        return imgs






class Temporal_RandStep_puzzle_kinetics_Dataset(Dataset):
    def __init__(
        self,
        lmdb_path: str,
        img_size: int = 224,
        seq_len: int = 8,
        transform: Optional[T.Compose] = None,
        min_step: int = 1,
        max_step: int = 20,
        cache_path: str = "kinetics_vid2frames_cache.json"
    ):
        super().__init__(lmdb_path, transform=None) 

        assert seq_len > 0, "seq_len must be positive"
        assert 1 <= min_step <= max_step, "Require 1 <= min_step <= max_step"
        self.seq_len = seq_len
        self.min_step = min_step
        self.max_step = max_step
        self.rng = random.Random() # Random number generator

        '''
        # 1) Group all frame keys by video ID
        self.vid2frames: Dict[str, List[int]] = collections.defaultdict(list)
        for key_bytes in getattr(self, "data", []):
            try:
                key_str = key_bytes.decode("utf-8")
                vid, idx_str = key_str.rsplit("_", 1)
                self.vid2frames[vid].append(int(idx_str))
            except (ValueError, IndexError):
                continue

        # 2) Filter and sort frames for each video
        self.videos: List[str] = []
        for vid, frames in self.vid2frames.items():
            # Ensure enough frames for both sequence and neighbor sampling
            if len(frames) >= self.seq_len: 
                self.vid2frames[vid] = sorted(list(set(frames)))
                self.videos.append(vid)
        '''
        # --- CACHING LOGIC STARTS HERE ---
        if cache_path and os.path.exists(cache_path):
            print(f"Loading video-to-frame mapping from cache: {cache_path}")
            with open(cache_path, 'r') as f:
                self.vid2frames = json.load(f)
        else:
            print("Building video-to-frame mapping from scratch (this will take a while)...")
            # 1) Group all frame keys by video ID (the slow part)
            vid2frames_temp = collections.defaultdict(list)
            for key_bytes in tqdm(getattr(self, "data", []), desc="Scanning LMDB keys"):
                try:
                    key_str = key_bytes.decode("utf-8")
                    vid, idx_str = key_str.rsplit("_", 1)
                    vid2frames_temp[vid].append(int(idx_str))
                except (ValueError, IndexError):
                    continue

            # 2) Filter out videos that are too short and sort frame indices
            print("Filtering and sorting videos...")
            self.vid2frames = {}
            for vid, frames in tqdm(vid2frames_temp.items(), desc="Filtering videos"):
                sorted_frames = sorted(list(set(frames)))
                if len(sorted_frames) >= self.seq_len:
                    self.vid2frames[vid] = sorted_frames
            
            # 3) Save the processed dictionary to the cache file for next time
            if cache_path:
                print(f"Saving mapping to cache for future use: {cache_path}")
                with open(cache_path, 'w') as f:
                    json.dump(self.vid2frames, f)
        # --- CACHING LOGIC ENDS HERE ---

        self.videos: List[str] = list(self.vid2frames.keys())
        
        print(f"Initialized dataset with {len(self.videos)} videos.")

        # 3) Define transformation pipelines for each output
        # You might want different augmentations for each branch
        self.transform_seq = T.Compose([
            T.Resize((img_size, img_size)),
            #T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            utils.GaussianBlur(0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        self.transform_puzzle = transform
        
    def set_seed(self, epoch: int):
        """Seeds the random number generator for reproducibility if needed."""
        self.rng.seed(epoch)

    def __len__(self) -> int:
        return len(self.videos)

    def _get_frame_from_idx(self, vid, frame_idx):
        """Helper to load and transform a single frame by its integer index."""
        key = f"{vid}_{frame_idx}".encode("utf-8")
        img = self.load_image_from_lmdb(key)
        return img # Return PIL image, transform will be applied later

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vid = self.videos[idx]
        idx_list = self.vid2frames[vid]
        n_frames = len(idx_list)

        # --- 1. Sample the Temporal Sequence ---
        max_possible_step = (n_frames - 1) // (self.seq_len - 1)
        step = self.rng.randint(self.min_step, min(self.max_step, max_possible_step))
        max_start_index = n_frames - (self.seq_len - 1) * step - 1
        start_index = self.rng.randint(0, max_start_index)
        
        sequence_frames_pil = []
        for i in range(self.seq_len):
            frame_original_index = idx_list[start_index + i * step]
            sequence_frames_pil.append(self._get_frame_from_idx(vid, frame_original_index))
        
        sequence_tensor = torch.stack([self.transform_seq(f) for f in sequence_frames_pil], 0)

        # --- 2. Sample the Single Frame for iBOT ---
        # Calculate the start and end positions for the [0.3, 0.7] window
        start_pos = int(n_frames * 0.3)
        end_pos = int(n_frames * 0.7)
        
        # Ensure the range is valid for randint (start must be less than end)
        if start_pos >= end_pos:
            end_pos = start_pos + 1 
        
        # Randomly pick an index from the calculated window
        ibot_frame_pos = self.rng.randint(start_pos, end_pos)
        ibot_frame_idx = idx_list[ibot_frame_pos]
        ibot_frame_pil = self._get_frame_from_idx(vid, ibot_frame_idx)

        # --- 3. Sample the Two Neighboring Frames for the Puzzle ---
        # Calculate past neighbor range
        past_min_offset = int(n_frames * 0.15)
        past_max_offset = int(n_frames * 0.25)
        past_offset = self.rng.randint(past_min_offset, past_max_offset)
        past_pos = max(0, ibot_frame_pos - past_offset) # Ensure index is not negative
        past_frame_idx = idx_list[past_pos]
        
        # Calculate future neighbor range
        future_min_offset = int(n_frames * 0.15)
        future_max_offset = int(n_frames * 0.25)
        future_offset = self.rng.randint(future_min_offset, future_max_offset)
        future_pos = min(n_frames - 1, ibot_frame_pos + future_offset) # Ensure index is within bounds
        future_frame_idx = idx_list[future_pos]
        
        # Load frames
        past_neighbor_pil = self._get_frame_from_idx(vid, past_frame_idx)
        future_neighbor_pil = self._get_frame_from_idx(vid, future_frame_idx)
        if self.transform_puzzle is not None:
            imgs, puzzles = self.transform_puzzle(ibot_frame_pil, [past_neighbor_pil, future_neighbor_pil])


        return imgs, puzzles, sequence_tensor




class Temporal_RandStep_dataset(Dataset):
    '''
    _RULES: List[Tuple[int, int, int]] = [
        (0,   60, 10),     # segments  8–15  → 100 samples/epoch
        (60,  120, 20),     # segments 16–31  → 200 samples/epoch
        (120,  10**9, 30),  # segments ≥32    → 300 samples/epoch
    ]
    '''
    _RULES: List[Tuple[int, int, int]] = [
        (0,   240, 60),     # segments  8–15  → 100 samples/epoch
        (240,  960, 200),     # segments 16–31  → 200 samples/epoch
        (960,  10**9, 300),  # segments ≥32    → 300 samples/epoch
    ]

    def __init__(
        self,
        lmdb_path: str,
        split_txt: Optional[str] = None,   # kept for API compatibility (unused)
        img_size: int = 224,
        seq_len: int = 8,
        transform: Optional[T.Compose] = None,
        min_step: int = 1,
        max_step: int = 20,
    ):
        # Parent must set self.data and provide load_image_from_lmdb
        super().__init__(lmdb_path, transform=None, size=img_size)

        assert seq_len > 0, "seq_len must be positive"
        assert 1 <= min_step <= max_step, "Require 1 <= min_step <= max_step"
        self.seq_len = seq_len
        self.min_step = min_step
        self.max_step = max_step

        # 1) Group inherited keys by vid -> sorted list of integer frame indices
        self.vid2frames: Dict[str, List[int]] = collections.defaultdict(list)

        def _decode(k):
            return k.decode("utf-8") if isinstance(k, (bytes, bytearray)) else str(k)

        for k in getattr(self, "data", []):
            s = _decode(k).strip()
            if not s:
                continue
            try:
                vid, idx_str = s.rsplit("_", 1)
                idx = int(idx_str)
            except Exception:
                # Skip malformed keys silently
                continue
            self.vid2frames[vid].append(idx)
        print(len(self.vid2frames))

        # sort indices per video and drop videos with fewer than seq_len indices
        for vid in list(self.vid2frames.keys()):
            arr = sorted(set(self.vid2frames[vid]))
            if len(arr) < self.seq_len:
                del self.vid2frames[vid]
            else:
                self.vid2frames[vid] = arr

        # 2) Aug pipeline (independent across frames; matches your original)
        self.frame_tx = transform or T.Compose([
            T.Resize((img_size, img_size)),
            #T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            utils.GaussianBlur(0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 3) Build samples for epoch 0
        self._build_samples(seed=0)

    def set_epoch(self, epoch: int):
        """Call once per (global) epoch to resample."""
        self._build_samples(seed=epoch)

    # ---- sampling helpers ----

    def _per_video_repeats(self, n_indices: int) -> int:
        # Proxy “segments” = how many seq_len-sized chunks exist
        segments = max(1, n_indices)
        return next(rep for lo, hi, rep in self._RULES if lo <= segments < hi)

    # ---- core sampler ----

    def _build_samples(self, seed: int):
        rng = random.Random(seed)
        samples: List[List[bytes]] = []

        for vid, idx_list in self.vid2frames.items():
            n = len(idx_list)
            if n < self.seq_len:
                continue

            repeats = self._per_video_repeats(n)

            # Minimal span required to place a clip with min_step
            min_total_span = self.min_step * (self.seq_len - 1)
            if n - 1 < min_total_span:
                # Shouldn't happen since n >= seq_len and min_step >= 1, but guard anyway
                continue

            for _ in range(repeats):
                # 1) Random START that fits with min_step
                max_start_for_min = n - 1 - min_total_span  # inclusive
                start_pos = rng.randint(0, max_start_for_min)

                # 2) Max feasible step for this start (respecting bounds and max_step)
                s_max_feasible = (n - 1 - start_pos) // (self.seq_len - 1)
                s_max = min(self.max_step, s_max_feasible)  # s_max_feasible >= min_step by construction

                # 3) FIXED step for this clip (varies across clips)
                step = rng.randint(self.min_step, s_max)

                # 4) Build positions (start_pos + i*step)
                positions = [start_pos + i * step for i in range(self.seq_len)]

                # 5) Map to frame indices / LMDB keys
                chosen_frame_indices = [idx_list[p] for p in positions]
                keys = [f"{vid}_{t}".encode("utf-8") for t in chosen_frame_indices]
                samples.append(keys)

        self.samples: List[List[bytes]] = samples

    # ---- Dataset API ----

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        keys = self.samples[idx]  # list of seq_len keys
        # --- ADD THIS LINE TO REVERSE THE ORDER ---
        #keys = keys[::-1]
        
        frames = [self.frame_tx(self.load_image_from_lmdb(k)) for k in keys]
        return torch.stack(frames, 0)  # [seq_len, 3, H, W]






class Temporal_dataset(Dataset):
    _RULES: List[Tuple[int, int, int]] = [
        (8,   16, 100),      # clips  8–15  → 100 
        (16,  32, 300),      # clips 16–31  → 200
        (32,  10**9, 400),   # clips ≥32    → 300
    ]

    def __init__(self,
                 lmdb_path: str,
                 split_txt: str,
                 img_size: int = 224,
                 seq_len : int = 8,
                 transform: Optional[T.Compose] = None):
        super().__init__(lmdb_path, transform=None, size=img_size)

        self.seq_len = seq_len

        # 1) parse split file → vid_id → list[(start,len)]
        self.vid2clips: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        with open(split_txt) as fh:
            for line in fh:
                vid_start, clip_len, *_ = line.strip().split()[:3]
                vid, start = vid_start.rsplit("_", 1)
                self.vid2clips[vid].append((int(start), int(clip_len)))

        # 2) store augment pipeline
        self.frame_tx = transform or T.Compose([
            T.Resize((img_size, img_size)),
            #T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            utils.GaussianBlur(0.1),
            T.ToTensor(),                                    # 0-1 float
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 3) build sample list for epoch-0
        self._build_samples(seed=0)

    def set_epoch(self, epoch: int):
        """Call once per epoch from the training loop for fresh sampling."""
        self._build_samples(seed=epoch)

    def _build_samples(self, seed: int):
        rng = random.Random(seed)
        self.samples: List[List[bytes]] = []

        for vid, clips in self.vid2clips.items():
            n = len(clips)
            if n < self.seq_len:                      # skip small-clip videos
                continue

            repeats = next(rep for lo, hi, rep in self._RULES if lo <= n < hi)
            for _ in range(repeats):
                chosen = rng.sample(clips, self.seq_len)          # always ≥ 8 now
                chosen = sorted(chosen, key=lambda x: x[0])  # ← ensure time-order
                keys = []
                for start, L in chosen:
                    frame = start + rng.randint(0, L - 1)
                    keys.append(f"{vid}_{frame}".encode())
                self.samples.append(keys)


    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        keys = self.samples[idx]                      # 8 frame-keys
        frames = [self.frame_tx(self.load_image_from_lmdb(k)) for k in keys]
        return torch.stack(frames, 0)                 # [8,3,H,W]




class Temporal_video_dataset(Video_Dataset):
    _RULES: List[Tuple[int, int, int]] = [
        (8,   16, 100),      # clips  8–15  → 10 
        (16,  32, 200),      # clips 16–31  → 20
        (32,  10**9, 300),   # clips ≥32    → 30
    ]

    def __init__(self,
                 video_path: str,
                 split_txt: str,
                 img_size: int = 224,
                 seq_len : int = 8,
                 transform: Optional[T.Compose] = None):
        super().__init__(video_path, transform=None, size=img_size)

        self.seq_len = seq_len

        # 1) parse split file → vid_id → list[(start,len)]
        self.vid2clips: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        with open(split_txt) as fh:
            for line in fh:
                vid, start, clip_len, fps = line.strip().split()[:4]
                self.vid2clips[vid].append((int(start), int(clip_len), int(fps)))

        # 2) store augment pipeline
        self.frame_tx = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            utils.GaussianBlur(0.1),
            T.ToTensor(),                                    # 0-1 float
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 3) build sample list for epoch-0
        self._build_samples(seed=0)

    def set_epoch(self, epoch: int):
        """Call once per epoch from the training loop for fresh sampling."""
        self._build_samples(seed=epoch)

    def _build_samples(self, seed: int):
        rng = random.Random(seed)
        self.samples: List[List[bytes]] = []

        for vid, clips in self.vid2clips.items():
            n = len(clips)
            if n < self.seq_len:                      # skip small-clip videos
                continue

            repeats = next(rep for lo, hi, rep in self._RULES if lo <= n < hi)
            for _ in range(repeats):
                chosen = rng.sample(clips, self.seq_len)          # always ≥ 8 now
                chosen = sorted(chosen, key=lambda x: x[0])  # ← ensure time-order
                keys = []
                for start, L, fps in chosen:
                    frame = start + rng.randint(0, L - 1)
                    keys.append((vid, frame*fps, fps))
                self.samples.append(keys)


    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        keys = self.samples[idx]     # 8 frame-keys
        frames = []
        for video, frame_idx, fps in keys:
            decord_vr = decord.VideoReader(os.path.join(self.videos, video), num_threads=1)
            selected_idx = frame_idx + np.random.randint(0, fps-1)
            
            frame_nd = decord_vr[selected_idx].asnumpy()  # (H,W,3), RGB, uint8
            fetched_image = Image.fromarray(frame_nd).convert('RGB')
            img = self.frame_tx(fetched_image)
            frames.append(img)

        return torch.stack(frames, 0)
        

    


class Temporal_dataset_Original(ImageFolder):
    """
    Build clips of length 10: [digit0, digit1, ..., digit9],
    where each digit i is sampled from folder data_path/str(i).

    Returns a tensor of shape [10, C, H, W].
    """

    def __init__(self,
                 data_path: str,
                 img_size: int = 224,
                 clips_per_epoch: Optional[int] = None,
                 mode: str = "random",        # "random" (with replacement) or "deterministic"
                 transform: Optional[T.Compose] = None):
        """
        Args:
            data_path: root folder that contains subfolders '0'...'9'.
            img_size:  output H=W after transforms.
            clips_per_epoch: number of clips to pre-sample per epoch (only used in 'random' mode).
                             If None in random mode, defaults to 10 * min_class_count.
            mode: 'random' -> sample with replacement each epoch (rebuild with set_epoch).
                  'deterministic' -> clip i uses the i-th image from each class (wraps if needed).
            transform: per-frame transform. If None, a reasonable default is used.
        """
        super().__init__(data_path)
        self.mode = mode.lower()
        assert self.mode in {"random", "deterministic"}

        # 1) index all images by digit class
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        self.cls2paths: Dict[int, List[str]] = {}
        for d in range(10):
            paths = []
            for e in exts:
                paths.extend(glob.glob(os.path.join(data_path, str(d), e)))
            paths = sorted(paths)
            if len(paths) == 0:
                raise RuntimeError(f"No images found for class '{d}' in {os.path.join(data_path, str(d))}")
            self.cls2paths[d] = paths

        self.class_counts = {d: len(v) for d, v in self.cls2paths.items()}
        self.min_class = min(self.class_counts.values())

        # 2) transforms
        self.frame_tx = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 3) decide epoch size & build initial samples
        if self.mode == "random":
            self.clips_per_epoch = clips_per_epoch or (3 * self.min_class)
        else:  # deterministic
            # natural size = min_class (so every class can provide i-th image)
            self.clips_per_epoch = clips_per_epoch or self.min_class

        self._build_samples(seed=0)

    def set_epoch(self, epoch: int):
        """Call once per epoch from the training loop for fresh random sampling."""
        self._build_samples(seed=epoch)

    def _build_samples(self, seed: int):
        rng = random.Random(seed)
        self.samples: List[List[str]] = []  # each item: list of 10 file paths ordered 0..9

        if self.mode == "random":
            # With replacement: for each clip, pick one random path from each class 0..9
            for _ in range(self.clips_per_epoch):
                clip_paths = [rng.choice(self.cls2paths[d]) for d in range(10)]
                self.samples.append(clip_paths)
        else:
            # Deterministic pairing: clip i uses i-th image from each class (wrapping by modulus)
            # Ensures stable epoch regardless of differing class sizes.
            max_len = self.clips_per_epoch
            for i in range(max_len):
                clip_paths = []
                for d in range(10):
                    paths = self.cls2paths[d]
                    clip_paths.append(paths[i % len(paths)])
                self.samples.append(clip_paths)

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.frame_tx(img)

    def __getitem__(self, idx: int) -> torch.Tensor:
        paths = self.samples[idx]           # list of 10 paths in order 0..9
        frames = [self._load_image(p) for p in paths]
        # Stack into [10, C, H, W]
        return torch.stack(frames, dim=0)



        

class ImageFolderInstance(ImageFolder):
    def __getitem__(self, index):
        img, target = super(ImageFolderInstance, self).__getitem__(index)
        return img, target, index






class ImageFolderMask_kinetics(kinetics_singleFrame_Dataset):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask_kinetics, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask_kinetics, self).__getitem__(index)
                
        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output,) + (masks,)








class ImageFolderMask_lpw_kinetics(Temporal_RandStep_puzzle_kinetics_Dataset):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask_lpw_kinetics, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output, puzzle_images, sequence_tensor = super(ImageFolderMask_lpw_kinetics, self).__getitem__(index)
                
        masks = []
        for img in output + [puzzle_images[0]]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count 

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output,) + (masks,) + (puzzle_images,) + (sequence_tensor,)







class ImageFolderMask_puzzle(Dataset_puzzle):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask_puzzle, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output, puzzle_images = super(ImageFolderMask_puzzle, self).__getitem__(index)
                
        masks = []
        for img in output + [puzzle_images[0]]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count 

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output,) + (masks,) + (puzzle_images,)




class ImageFolderMask(Dataset):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count 

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output,) + (masks,)



class ImageFolderMask_Original(ImageFolder):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(ImageFolderMask_Original, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(ImageFolderMask_Original, self).__getitem__(index)
                
        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)



class VideoFolderMask(Video_Dataset):
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio, 
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(VideoFolderMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
            len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
            len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        output = super(VideoFolderMask, self).__getitem__(index)
                
        masks = []
        for img in output:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue
            
            high = self.get_pred_ratio() * H * W
            
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
            
            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return (output,) + (masks,)



class Temporal_innerclips_dataset(Dataset):
    _RULES: List[Tuple[int, int, int]] = [
        (8,   16, 10),      # clips  8–15  → 10 synthetic samples
        (16,  32, 20),      # clips 16–31  → 20
        (32,  10**9, 30),   # clips ≥32    → 30
    ]

    def __init__(self,
                 lmdb_path: str,
                 split_txt: str,
                 img_size: int = 224,
                 seq_len : int = 8,
                 transform: Optional[T.Compose] = None):
        super().__init__(lmdb_path, transform=None, size=img_size)

        self.seq_len = seq_len

        # 1) parse split file → vid_id → list[(start,len)]
        self.vid2clips: Dict[str, List[Tuple[int, int]]] = collections.defaultdict(list)
        with open(split_txt) as fh:
            for line in fh:
                vid_start, clip_len, *_ = line.strip().split()[:3]
                vid, start = vid_start.rsplit("_", 1)
                self.vid2clips[vid].append((int(start), int(clip_len)))

        # 2) store augment pipeline
        self.frame_tx = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
            utils.GaussianBlur(0.1),
            T.ToTensor(),                                    # 0-1 float
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # 3) build sample list for epoch-0
        self._build_samples(seed=0)

    def set_epoch(self, epoch: int):
        """Call once per epoch from the training loop for fresh sampling."""
        self._build_samples(seed=epoch)

    def _build_samples(self, seed: int):
        """
        Build self.samples so that **each original clip (start,L) contributes
        exactly ONE contiguous seq_len window**.

        · seq_len == 8  (contiguous frames required)
        · skip clips shorter than seq_len
        """
        rng = random.Random(seed)
        self.samples: List[List[bytes]] = []

        for vid, clips in self.vid2clips.items():
            for start, L in clips:
                if L < self.seq_len:                 # 不足 8 帧直接跳过
                    continue

                # 随机窗口起点：保证整段在 clip 内
                offset = 0 if L == self.seq_len else rng.randint(0, L - self.seq_len)

                # 连续 8 帧 key 列表
                keys = [
                    f"{vid}_{start + offset + i}".encode()
                    for i in range(self.seq_len)
                ]
                self.samples.append(keys)


    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        keys = self.samples[idx]                      # 8 frame-keys
        frames = [self.frame_tx(self.load_image_from_lmdb(k)) for k in keys]
        return torch.stack(frames, 0)                 # [8,3,H,W]