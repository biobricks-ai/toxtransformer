import pathlib, torch, torch.nn as nn, torch.nn.functional as F
from cvae.models.multitask_transformer import PositionalEncoding, generate_custom_subsequent_mask, MultitaskTransformer, SelfiesPropertyValTokenizer
import cvae.models.multitask_transformer as mt
import cvae.utils
import json
import logging
import torch
import random
from collections import defaultdict

class MoE(nn.Module):

    def __init__(self, tokenizer, num_experts=2, k=2, hdim=32, nhead=8, dim_feedforward=32, noise_factor=.1, 
                 dropout_rate=.1, balance_loss_weight=1.0, diversity_loss_weight=10, expert_layers=4, output_size=None):
        super().__init__()
        self.tokenizer: SelfiesPropertyValTokenizer = tokenizer
        self.expert_layers = expert_layers
        self.output_size = output_size if output_size is not None else tokenizer.vocab_size
        mkexpert = lambda: MultitaskTransformer(tokenizer, hdim, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=expert_layers, output_size=self.output_size)
        self.experts = nn.ModuleList([mkexpert() for _ in range(num_experts)])
        self.gating_network = MultitaskTransformer(tokenizer, hdim, output_size=num_experts, nhead=nhead)
        self.hdim = hdim
        self.dim_feedforward = dim_feedforward
        self.nhead = nhead
        self.noise_factor = noise_factor
        self.balance_loss_weight = balance_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.balance_loss = 0.0
        self.diversity_loss = 0.0
        self.dropout_rate = dropout_rate
        self.current_train_step = 0
        self.num_experts = num_experts
        self.k = k
        

    def compute_gating_distribution(self, input, teach_forcing, expert_mask=None):
        gating_scores = self.gating_network(input, teach_forcing)
        if self.training and self.noise_factor > 0:
            gating_scores = gating_scores + torch.randn_like(gating_scores) * self.noise_factor

        if expert_mask is not None:
            expert_mask = expert_mask.view(1, 1, -1)
            gating_scores = gating_scores * expert_mask

        gating_distribution = F.softmax(gating_scores, dim=-1)
        return gating_distribution

    def compute_balance_loss(self, gating_distribution):
        eps = 1e-8
        expert_usage = gating_distribution.sum(dim=(0, 1))  # [num_experts]
        total_usage = expert_usage.sum()
        if total_usage < eps:
            return torch.tensor(0.0, device=gating_distribution.device)

        expert_usage = expert_usage / total_usage  # normalized

        target_usage = torch.ones_like(expert_usage) / self.num_experts
        balance_loss = F.kl_div(
            expert_usage.log(),
            target_usage,
            reduction='batchmean',
            log_target=False
        )

        return self.balance_loss_weight * balance_loss

    def compute_diversity_loss(self, expert_outputs):
        # expert_outputs: list of [batch, seq_len, vocab] tensors, one per expert
        valid_outputs = [eo for eo in expert_outputs if eo is not None]
        if len(valid_outputs) < 2:
            return torch.tensor(0.0, device=expert_outputs[0].device)

        # For each expert's output tensor [batch, seq_len, vocab], compute mean across batch and sequence dimensions
        # This gives us a single vector per expert of size [vocab], then stack all expert vectors into [num_experts, vocab]
        flattened = torch.stack([eo.mean(dim=(0, 1)) for eo in valid_outputs], dim=0)  # [E, V]

        # Normalize to unit length — if any vector is all zeros, skip
        norm = torch.norm(flattened, p=2, dim=1, keepdim=True)  # [E, 1]
        valid_mask = norm.squeeze() > 1e-6
        if valid_mask.sum() < 2:
            return torch.tensor(0.0, device=flattened.device)

        flattened = flattened[valid_mask]
        flattened = flattened / (norm[valid_mask] + 1e-8)  # [E_filtered, D]

        # Cosine similarity matrix: [E_filtered, E_filtered]
        similarity = torch.matmul(flattened, flattened.T)
        similarity = torch.clamp(similarity, -1.0, 1.0)

        # Keep only upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(similarity), diagonal=1)  # [E, E]
        n_pairs = mask.sum()

        # Penalize similarity → encourage diversity
        diversity_loss = (similarity ** 2 * mask).sum() / (n_pairs + 1e-8)
        diversity_loss = torch.tanh(diversity_loss)

        return self.diversity_loss_weight * diversity_loss

    def forward(self, input, teach_forcing=None):
        
        B, T = input.shape
        E = self.num_experts
        V = self.tokenizer.vocab_size

        # Compute expert outputs: List of [B, T, V]
        expert_outputs = [expert(input, teach_forcing) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # [B, T, V, E]

        # Compute gating scores: [B, T, E]
        gating_scores = self.gating_network(input, teach_forcing)
        if self.training and self.noise_factor > 0:
            gating_scores = gating_scores + torch.randn_like(gating_scores) * self.noise_factor

        # For balance loss
        soft_distribution = F.softmax(gating_scores, dim=-1)

        # Gumbel noise (optionally add more control here)
        if self.training and self.noise_factor > 0:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(gating_scores) + 1e-9) + 1e-9)
            gating_scores = gating_scores + gumbel_noise * self.noise_factor

        # Get top-2 experts: indices and values
        topk_values, topk_indices = torch.topk(gating_scores, k=self.k, dim=-1)  # [B, T, 2]

        # Softmax over top-k only
        topk_weights = F.softmax(topk_values, dim=-1)  # [B, T, 2]

        # Create one-hot mask to select top-k experts
        topk_mask = F.one_hot(topk_indices, num_classes=E).float()  # [B, T, 2, E]

        # Distribute weights into full [B, T, E] tensor
        routing_weights = (topk_weights.unsqueeze(-1) * topk_mask).sum(dim=2)  # [B, T, E]

        # Apply routing weights to expert outputs
        output = (stacked_outputs * routing_weights.unsqueeze(2)).sum(dim=-1)  # [B, T, V]

        # Auxiliary losses
        if self.training:
            self.balance_loss = self.compute_balance_loss(soft_distribution)
            self.diversity_loss = self.compute_diversity_loss(expert_outputs)
            if random.random() < 0.001:
                logging.info(f"balance_loss: {self.balance_loss.item()}, diversity_loss: {self.diversity_loss.item()}")
                
        return output
    
    def build_stratified_lossfn(self,label_smoothing=.001):
        ignore_index = self.tokenizer.pad_idx
        value_token_ids = torch.tensor(list(self.tokenizer.value_indexes().values()), dtype=torch.long)
        
        def lossfn(parameters, logits, output):
            batch_size, vocab_size, seq_len = logits.size()

            token_indices = torch.arange(seq_len, device=logits.device)
            is_value_position = (token_indices >= 2) & (token_indices % 2 == 0)
            is_value_position = is_value_position.unsqueeze(0).expand(batch_size, seq_len)

            is_not_pad = (output != ignore_index)
            final_mask = is_value_position & is_not_pad
            
            logits = logits.transpose(1, 2).contiguous()
            logits_selected = logits[final_mask]
            output_selected = output[final_mask]
            
            # Move value token IDs to device
            value_tokens_device = value_token_ids.to(logits.device, non_blocking=True)
            
            # Extract logits for just the two value tokens
            value_logits = logits_selected[:, value_tokens_device]
            
            # Create binary targets (0 for first value token, 1 for second)
            binary_targets = (output_selected == value_tokens_device[1]).long()
            
            # Standard cross entropy with label smoothing on 2 classes
            loss = torch.nn.functional.cross_entropy(value_logits,binary_targets, label_smoothing=label_smoothing)
            
            return loss

        return lossfn

    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        torch.save(self.state_dict(), path / "mtransformer.pt")

        config = {
            "num_experts": self.num_experts,
            "k": self.k,
            "hdim": self.hdim,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "noise_factor": self.noise_factor,
            "dropout_rate": self.dropout_rate,
            "balance_loss_weight": self.balance_loss_weight,
            "diversity_loss_weight": self.diversity_loss_weight,
            "expert_layers": self.expert_layers,
            "output_size": self.output_size
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return path

    @staticmethod
    def load(dirpath=pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        config = json.load(open(dirpath / "config.json"))
        model = MoE(tokenizer, **config)
        state_dict = torch.load(dirpath / 'mtransformer.pt', map_location='cpu', weights_only=True)

        model.load_state_dict(state_dict)
        model.eval()
        return model
