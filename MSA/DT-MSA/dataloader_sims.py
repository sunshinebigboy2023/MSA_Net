import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def sample_id_for_row(row):
    return f"{row['video_id']}_{int(row['clip_id']):04d}"


def sequence_id_for_row(row):
    return f"{row['split']}_{row['video_id']}"


def load_label_rows(label_path):
    with open(label_path, "r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["clip_id"] = int(row["clip_id"])
        row["label"] = float(row["label"])
        row["split"] = row["split"].strip().lower()
        row["sample_id"] = sample_id_for_row(row)
        row["sequence_id"] = sequence_id_for_row(row)
    return sorted(rows, key=lambda item: (item["split"], item["video_id"], item["clip_id"]))


def read_data(label_path, feature_root):
    rows = load_label_rows(label_path)
    names = [row["sample_id"] for row in rows]
    features = []
    feature_dim = -1

    for name in names:
        feature_path = os.path.join(feature_root, name + ".npy")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"Feature not found: {feature_path}")
        single_feature = np.load(feature_path).squeeze()
        if len(single_feature.shape) == 2:
            single_feature = np.mean(single_feature, axis=0)
        feature_dim = max(feature_dim, single_feature.shape[-1])
        features.append(single_feature)

    print(f"Input feature {os.path.basename(feature_root)} ===> dim is {feature_dim}; No. sample is {len(names)}")
    return dict(zip(names, features)), feature_dim


class SIMSDataset(Dataset):
    def __init__(self, label_path, audio_root, text_root, video_root):
        name2audio, adim = read_data(label_path, audio_root)
        name2text, tdim = read_data(label_path, text_root)
        name2video, vdim = read_data(label_path, video_root)
        self.adim = adim
        self.tdim = tdim
        self.vdim = vdim

        rows = load_label_rows(label_path)
        sequence_rows = defaultdict(list)
        for row in rows:
            sequence_rows[row["sequence_id"]].append(row)

        self.trainVids = sorted([seq for seq in sequence_rows if seq.startswith("train_")])
        self.valVids = sorted([seq for seq in sequence_rows if seq.startswith("valid_")])
        self.testVids = sorted([seq for seq in sequence_rows if seq.startswith("test_")])
        self.vids = self.trainVids + self.valVids + self.testVids

        self.max_len = 0
        self.videoAudioHost = {}
        self.videoTextHost = {}
        self.videoVisualHost = {}
        self.videoAudioGuest = {}
        self.videoTextGuest = {}
        self.videoVisualGuest = {}
        self.videoLabelsNew = {}
        self.videoSpeakersNew = {}

        for sequence_id in self.vids:
            ordered_rows = sorted(sequence_rows[sequence_id], key=lambda item: item["clip_id"])
            self.max_len = max(self.max_len, len(ordered_rows))
            sample_ids = [row["sample_id"] for row in ordered_rows]

            self.videoAudioHost[sequence_id] = np.array([name2audio[sample_id] for sample_id in sample_ids])
            self.videoTextHost[sequence_id] = np.array([name2text[sample_id] for sample_id in sample_ids])
            self.videoVisualHost[sequence_id] = np.array([name2video[sample_id] for sample_id in sample_ids])
            self.videoAudioGuest[sequence_id] = np.zeros((len(sample_ids), self.adim))
            self.videoTextGuest[sequence_id] = np.zeros((len(sample_ids), self.tdim))
            self.videoVisualGuest[sequence_id] = np.zeros((len(sample_ids), self.vdim))
            self.videoLabelsNew[sequence_id] = np.array([row["label"] for row in ordered_rows])
            self.videoSpeakersNew[sequence_id] = np.zeros((len(sample_ids),))

    def __getitem__(self, index):
        vid = self.vids[index]
        return torch.FloatTensor(self.videoAudioHost[vid]), \
               torch.FloatTensor(self.videoTextHost[vid]), \
               torch.FloatTensor(self.videoVisualHost[vid]), \
               torch.FloatTensor(self.videoAudioGuest[vid]), \
               torch.FloatTensor(self.videoTextGuest[vid]), \
               torch.FloatTensor(self.videoVisualGuest[vid]), \
               torch.FloatTensor(self.videoSpeakersNew[vid]), \
               torch.FloatTensor([1] * len(self.videoLabelsNew[vid])), \
               torch.FloatTensor(self.videoLabelsNew[vid]), \
               vid

    def __len__(self):
        return len(self.vids)

    def get_featDim(self):
        print(f"audio dimension: {self.adim}; text dimension: {self.tdim}; video dimension: {self.vdim}")
        return self.adim, self.tdim, self.vdim

    def get_maxSeqLen(self):
        print(f"max seqlen: {self.max_len}")
        return self.max_len

    def collate_fn(self, data):
        datnew = []
        dat = pd.DataFrame(data)
        for i in dat:
            if i <= 5:
                datnew.append(pad_sequence(dat[i]))
            elif i <= 8:
                datnew.append(pad_sequence(dat[i], True))
            else:
                datnew.append(dat[i].tolist())
        return datnew
