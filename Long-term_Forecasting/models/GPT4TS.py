import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                gpt2_model_path = os.environ.get('GPT2_MODEL_PATH', 'gpt2')
                local_files_only = os.environ.get('GPT2_LOCAL_FILES_ONLY', '1') != '0'
                if local_files_only:
                    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
                    os.environ.setdefault('HF_HUB_OFFLINE', '1')
                    if not os.path.isdir(gpt2_model_path):
                        raise FileNotFoundError(
                            "GPT2_MODEL_PATH must be a local directory when GPT2_LOCAL_FILES_ONLY=1: "
                            "{}".format(gpt2_model_path)
                        )
                print("Loading GPT-2 from: {}".format(gpt2_model_path))
                self.gpt2 = GPT2Model.from_pretrained(
                    gpt2_model_path,
                    output_attentions=True,
                    output_hidden_states=True,
                    local_files_only=local_files_only
                )  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0


    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs

class MultiPeriodGPT4TS(nn.Module):
    def __init__(self, configs, device):
        super(MultiPeriodGPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.patch_sizes = self._parse_patch_sizes(configs)
        self.patch_nums = [(self.seq_len - patch_size) // self.stride + 2 for patch_size in self.patch_sizes]

        if configs.is_gpt:
            if configs.pretrain:
                gpt2_model_path = os.environ.get('GPT2_MODEL_PATH', 'gpt2')
                local_files_only = os.environ.get('GPT2_LOCAL_FILES_ONLY', '1') != '0'
                if local_files_only:
                    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
                    os.environ.setdefault('HF_HUB_OFFLINE', '1')
                    if not os.path.isdir(gpt2_model_path):
                        raise FileNotFoundError(
                            "GPT2_MODEL_PATH must be a local directory when GPT2_LOCAL_FILES_ONLY=1: "
                            "{}".format(gpt2_model_path)
                        )
                print("Loading GPT-2 from: {}".format(gpt2_model_path))
                self.gpt2 = GPT2Model.from_pretrained(
                    gpt2_model_path,
                    output_attentions=True,
                    output_hidden_states=True,
                    local_files_only=local_files_only
                )  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))

            total_patch_num = sum(self.patch_nums)
            if total_patch_num > self.gpt2.config.n_positions:
                raise ValueError(
                    "Total token length {} exceeds GPT2 max position {}. "
                    "Please increase stride or reduce patch sizes.".format(
                        total_patch_num, self.gpt2.config.n_positions
                    )
                )
        
        self.in_layers = nn.ModuleList([
            nn.Linear(patch_size, self.d_model) for patch_size in self.patch_sizes
        ])
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.period_embedding = nn.Embedding(len(self.patch_sizes), self.d_model)
        self.out_heads = nn.ModuleList([
            nn.Linear(self.d_model * patch_num, self.pred_len) for patch_num in self.patch_nums
        ])
        self.gate = nn.Linear(self.d_model, 1)
        
        if configs.is_gpt and configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        layers = [self.in_layers, self.padding_patch_layer, self.period_embedding, self.out_heads, self.gate]
        if configs.is_gpt:
            layers.append(self.gpt2)
        for layer in layers:
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

    def _parse_patch_sizes(self, configs):
        return [int(patch_size) for patch_size in configs.multi_patch.split(',')]

    def forward(self, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        period_tokens = []
        token_lengths = []
        for period_id, (patch_size, in_layer) in enumerate(zip(self.patch_sizes, self.in_layers)):
            patch_x = self.padding_patch_layer(x)
            patch_x = patch_x.unfold(dimension=-1, size=patch_size, step=self.stride)
            patch_x = rearrange(patch_x, 'b m n p -> (b m) n p')
            tokens = in_layer(patch_x)
            period_ids = torch.full(
                (tokens.shape[0], tokens.shape[1]),
                period_id,
                dtype=torch.long,
                device=tokens.device
            )
            tokens = tokens + self.period_embedding(period_ids)
            period_tokens.append(tokens)
            token_lengths.append(tokens.shape[1])

        outputs = torch.cat(period_tokens, dim=1)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        period_outputs = torch.split(outputs, token_lengths, dim=1)
        period_preds = []
        gate_scores = []
        for period_output, out_head in zip(period_outputs, self.out_heads):
            period_hidden = period_output.mean(dim=1)
            period_preds.append(out_head(period_output.reshape(B * M, -1)))
            gate_scores.append(self.gate(period_hidden))

        period_preds = torch.stack(period_preds, dim=1)
        gate_scores = torch.stack(gate_scores, dim=1)
        gate_weights = torch.softmax(gate_scores, dim=1)
        outputs = torch.sum(period_preds * gate_weights, dim=1)
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs
