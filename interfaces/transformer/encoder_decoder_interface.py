import torch
import torch.nn
from typing import Dict, Tuple
from models.encoder_decoder import add_eos
from models.transformer_enc_dec import TransformerResult
from ..model_interface import ModelInterface
from layers import cross_entropy

from ..encoder_decoder import EncoderDecoderResult


class TransformerEncDecInterface(ModelInterface):
    def __init__(
        self,
        model: torch.nn.Module,
        label_smoothing: float = 0.0,
    ):
        self.model = model
        self.label_smoothing = label_smoothing

    def loss(
        self, outputs: TransformerResult, ref: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        l = cross_entropy(
            outputs.data, ref, reduction="none", smoothing=self.label_smoothing
        )
        l = l.reshape_as(ref) * mask
        l = l.sum() / mask.sum()
        return l

    def decode_outputs(
        self, outputs: EncoderDecoderResult
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return outputs.outputs, outputs.out_lengths

    def __call__(
        self,
        data: Dict[str, torch.Tensor],
        train_eos: bool = True,
        force=False,
        collect_hidden: bool = False,
        collect_attention: bool = False,
        collect_decoder_hidden: bool = False,
        collect_decoder_attention: bool = False,
        value_zero_layer: int = None,
        value_zero_index: int = None,
        value_zero_index_decoder: int = None,
    ) -> EncoderDecoderResult:
        in_len = data["in_len"].long()
        out_len = data["out_len"].long()
        if "tree_pos_enc" in data:
            # No eos for tree
            in_with_eos = data["in"]
        else:
            in_with_eos = add_eos(data["in"], data["in_len"], self.model.encoder_eos)
            in_len += 1

        out_with_eos = add_eos(data["out"], data["out_len"], self.model.decoder_sos_eos)
        out_len += 1
        res = self.model(
            in_with_eos.transpose(0, 1),
            in_len,
            out_with_eos.transpose(0, 1),
            out_len,
            teacher_forcing=self.model.training or force,
            max_len=out_len.max().item(),  # TODO actually pass as an argument
            collect_hidden=collect_hidden,
            collect_attention=collect_attention,
            collect_decoder_attention=collect_decoder_attention,
            collect_decoder_hidden=collect_decoder_hidden,
            tree_pos_enc=data["tree_pos_enc"] if "tree_pos_enc" in data else None,
            value_zero_layer=value_zero_layer,
            value_zero_index=value_zero_index,
            value_zero_index_decoder=value_zero_index_decoder,
        )
        hidden_or_attn = None
        collecting = (
            collect_hidden
            or collect_attention
            or collect_decoder_attention
            or collect_decoder_hidden  # It's all just so tiresome
        )
        if collecting:
            res, hidden_or_attn = res
        res.data = res.data.transpose(0, 1)
        len_mask = ~self.model.generate_len_mask(
            out_with_eos.shape[0], out_len if train_eos else (out_len - 1)
        ).transpose(0, 1)

        loss = self.loss(res, out_with_eos, len_mask)
        if collecting:
            return EncoderDecoderResult(res.data, res.length, loss), hidden_or_attn
        else:
            return EncoderDecoderResult(res.data, res.length, loss)
