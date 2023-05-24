import torch
import torch.nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention, AttentionMask
from typing import Optional, Callable, Dict
from dataclasses import dataclass

# This file is based on PyTorch's internal implementation

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: ActivationFunction = F.relu,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        collect_self_attention=False,
        value_zero_index=None,
    ) -> torch.Tensor:
        attn_save = None
        if collect_self_attention:
            src2, attn_save = self.self_attn(
                src,
                src,
                AttentionMask(mask, None),
                need_weights=True,
                average_attn_weights=False,
                value_zero_index=value_zero_index,
            )
        else:
            src2 = self.self_attn(
                src, src, AttentionMask(mask, None), value_zero_index=value_zero_index
            )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if collect_self_attention:
            return src, attn_save
        return src

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.linear1.weight,
            gain=torch.nn.init.calculate_gain("relu")
            if self.activation is F.relu
            else 1.0,
        )
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation: ActivationFunction = F.relu,
        d_model_in: int = None,
    ):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        if d_model_in is not None and d_model_in != d_model:
            self.multihead_attn = MultiHeadAttention(
                d_model, nhead, input_size=d_model_in, dropout=dropout
            )
        else:
            self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.norm3 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.dropout3 = torch.nn.Dropout(dropout)

        self.activation = activation
        self.reset_parameters()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        full_target: Optional[torch.Tensor] = None,
        pos_offset: int = 0,
        collect_decoder_attention: bool = False,
        collect_decoder_hidden: bool = False,
        value_zero_index: int = None,
    ) -> torch.Tensor:
        assert pos_offset == 0 or tgt_mask is None
        attn_save_self = None
        attn_save_multi = None

        tgt2 = self.self_attn(
            tgt,
            tgt if full_target is None else full_target,
            mask=AttentionMask(None, tgt_mask),
            need_weights=collect_decoder_attention,
            average_attn_weights=False,
        )
        if collect_decoder_attention:
            tgt2, attn_save_self = tgt2
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Attention to encoder
        tgt2 = self.multihead_attn(
            tgt,
            memory,
            mask=AttentionMask(memory_key_padding_mask, None),
            need_weights=collect_decoder_attention,
            average_attn_weights=False,
            value_zero_index=value_zero_index,
        )
        if collect_decoder_attention:
            tgt2, attn_save_multi = tgt2

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        if collect_decoder_attention:
            return tgt, {"self": attn_save_self, "cross": attn_save_multi}
        else:
            return tgt

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(
            self.linear1.weight,
            gain=torch.nn.init.calculate_gain("relu")
            if self.activation is F.relu
            else 1.0,
        )
        torch.nn.init.xavier_uniform_(self.linear2.weight)


class TransformerDecoderBase(torch.nn.Module):
    @dataclass
    class State:
        step: int
        state: Dict[int, torch.Tensor]

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def create_state(
        self, batch_size: int, max_length: int, device: torch.device
    ) -> State:
        return self.State(
            0,
            {
                i: torch.empty([batch_size, max_length, self.d_model], device=device)
                for i in range(len(self.layers))
            },
        )

    def one_step_forward(self, state: State, data: torch.Tensor, *args, **kwargs):
        assert (
            data.shape[1] == 1
        ), f"For one-step forward should have one timesteps, but shape is {data.shape}"
        assert (
            state.step <= state.state[0].shape[1]
        ), f"State step was {state.step}, state shape was {state.state[0].shape[1]}"

        attn = {}
        for i, l in enumerate(self.layers):
            state.state[i][:, state.step : state.step + 1] = data  # TODO reset
            if data.requires_grad:
                # Need .clone() for greedy running when using Captum
                data = l(
                    data,
                    *args,
                    **kwargs,
                    full_target=state.state[i][:, : state.step + 1].clone(),
                    pos_offset=state.step,
                )
            else:
                data = l(
                    data,
                    *args,
                    **kwargs,
                    full_target=state.state[i][:, : state.step + 1],
                    pos_offset=state.step,
                )
                if kwargs["collect_decoder_attention"]:
                    data, attn0 = data
                    attn[i] = attn0
                elif kwargs["collect_decoder_hidden"]:
                    attn[i] = data.detach()
        state.step += 1
        if kwargs["collect_decoder_attention"] or kwargs["collect_decoder_hidden"]:
            return data, attn
        else:
            return data


class TransformerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int, *args, **kwargs):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [layer(*args, **kwargs) for _ in range(n_layers)]
        )

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for i, l in enumerate(self.layers):
            data = l(data, *args, **kwargs)
        return data

    def forward_collect_hidden(self, data: torch.Tensor, *args, **kwargs):
        hidden = {}
        value_zeroing = "value_zero_layer" in kwargs
        if value_zeroing:
            layer = kwargs["value_zero_layer"]
            idx = kwargs["value_zero_index"]
            # print(kwargs["value_zero_index"], kwargs["value_zero_layer"])

        for i, l in enumerate(self.layers):
            if value_zeroing:
                data = l(
                    data,
                    value_zero_index=None if i != layer else idx,
                    *args,
                    **{k: v for k, v in kwargs.items() if "value_zero" not in k},
                )
            else:
                data = l(data, *args, **kwargs)

            # Don't store everything on gpu
            hidden[i] = data.detach().cpu()

        return data, hidden

    def forward_collect_attention(self, data: torch.Tensor, *args, **kwargs):
        attn = {}
        for i, l in enumerate(self.layers):
            data, attn_current = l(data, collect_self_attention=True, *args, **kwargs)
            # Don't store everything on gpu
            attn[i] = attn_current.detach().cpu()
        return data, attn


class TransformerDecoder(TransformerDecoderBase):
    def __init__(self, layer, n_layers: int, d_model: int, *args, **kwargs):
        super().__init__(d_model)
        self.layers = torch.nn.ModuleList(
            [layer(d_model, *args, **kwargs) for _ in range(n_layers)]
        )

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


def TransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: TransformerEncoder(layer, *args, **kwargs)


def TransformerDecoderWithLayer(layer=TransformerDecoderLayer):
    return lambda *args, **kwargs: TransformerDecoder(layer, *args, **kwargs)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_model_decoder: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dim_feedforward_decoder: int = 2048,
        dropout: float = 0.1,
        activation: ActivationFunction = F.relu,
        encoder_layer=TransformerEncoderWithLayer(),
        decoder_layer=TransformerDecoderWithLayer(),
        activate_bottleneck: bool = False,
    ):
        super().__init__()
        self.activate_bottleneck = activate_bottleneck
        self.encoder = encoder_layer(
            n_layers=num_encoder_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = decoder_layer(
            n_layers=num_decoder_layers,
            d_model=d_model_decoder,
            nhead=nhead,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout,
            activation=activation,
            d_model_in=d_model,
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_length_mask: Optional[torch.Tensor] = None,
    ):
        # Simply set src_length_mask to 1 always?
        memory = self.encoder(src, src_length_mask)
        if self.activate_bottleneck:
            # Mask out everything after the first token
            # print("bottleneck activated ! :D")
            src_length_mask_bottleneck = src_length_mask.clone()
            src_length_mask_bottleneck[:, 1:] = True
            return self.decoder(tgt, memory, tgt_mask, src_length_mask_bottleneck)
        return self.decoder(tgt, memory, tgt_mask, src_length_mask)

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )
