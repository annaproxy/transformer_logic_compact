import os
import json
import numpy as np
import torch


class LogicDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab_path="vocabulary", data_path="data/val", data_inputs=None):
        self.load_vocabulary(vocab_path)
        self.in_vocabulary_inverse = {v: k for k, v in self.in_vocabulary.items()}
        self.out_vocabulary_inverse = {v: k for k, v in self.out_vocabulary.items()}
        if data_inputs:
            self.inputs_to_data(data_inputs)
        else:
            self.file_to_data(data_path)
        self.batch_dim = 1

    def inputs_to_data(self, inputs):
        """Input is a tuple of two lists (inputs, outputs)"""
        self.in_lines, self.out_lines = inputs
        self.build_idxs()

    def file_to_data(self, path):
        with open(path + ".src") as f:
            self.in_lines = f.readlines()
        with open(path + ".tgt") as f:
            self.out_lines = f.readlines()
        self.build_idxs()

    def build_idxs(self):
        self.in_idxs = [
            [self.in_vocabulary[s] for s in sentence.split()]
            for sentence in self.in_lines
        ]
        self.out_idxs = [
            [self.out_vocabulary[s] for s in sentence.split()]
            for sentence in self.out_lines
        ]

    def load_vocabulary(self, path):
        with open(os.path.join(path, "in_vocabulary.json")) as f:
            self.in_vocabulary = json.load(f)
        with open(os.path.join(path, "out_vocabulary.json")) as f:
            self.out_vocabulary = json.load(f)

    def __len__(self):
        return len(self.in_idxs)

    def __getitem__(self, item: int):
        in_seq = self.in_idxs[item]
        out_seq = self.out_idxs[item]
        d = {
            "in": np.asarray(in_seq, np.int16),
            "out": np.asarray(out_seq, np.int16),
            "in_len": len(in_seq),
            "out_len": len(out_seq),
        }
        return d

    def sample_to_text(self, batch_outputs, position: int):
        scores, out_len = batch_outputs
        out = scores.argmax(-1)
        out = (
            out.select(self.batch_dim, position)[: out_len[position].item()]
            .cpu()
            .numpy()
        )
        return self.output_ids_to_text(out)

    def input_ids_to_text(self, idxs):
        return " ".join(self.in_vocabulary_inverse[s] for s in idxs)

    def output_ids_to_text(self, idxs):
        return " ".join(self.out_vocabulary_inverse[s] for s in idxs)
