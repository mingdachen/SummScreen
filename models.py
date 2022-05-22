import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import model_utils

from decorators import auto_init_args, auto_init_pytorch
from transformers import LongformerConfig, LongformerModel
from transformer_xlm import CacheTransformer


class Base(nn.Module):
    def __init__(self, iter_per_epoch, experiment):
        super(Base, self).__init__()
        self.expe = experiment
        self.iter_per_epoch = iter_per_epoch
        self.eps = self.expe.config.eps
        self.config = {
            "attention_head_size": 64,
            "attn_layers": ["local", "lsh", "local", "lsh", "local", "lsh"],
            "axial_pos_embds": True,
            "sinusoidal_pos_embds": False,
            "axial_pos_embds_dim": [128, 384],
            "axial_pos_shape": [128, 256],
            "lsh_attn_chunk_length": 64,
            "local_attn_chunk_length": 64,
            "feed_forward_size": 1024,
            "hidden_act": "relu",
            "hidden_size": 512,
            "is_decoder": True,
            "num_hidden_layers": 6,
            "max_position_embeddings": 32768,
            "num_attention_heads": 2,
            "num_buckets": [64, 128],
            "num_hashes": 2,
            "lsh_attention_probs_dropout_prob": 0.05,
            "lsh_num_chunks_before": 1,
            "lsh_num_chunks_after": 0,
            "local_num_chunks_before": 1,
            "local_num_chunks_after": 0,
            "local_attention_probs_dropout_prob": 0.025,
            "hidden_dropout_prob": 0.05,
        }

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def to_tensor(self, inputs):
        if torch.is_tensor(inputs):
            return inputs.clone().detach().to(self.device)
        else:
            return torch.tensor(inputs, device=self.device)

    def to_tensors(self, *inputs):
        return [self.to_tensor(inputs_) if inputs_ is not None and inputs_.size
                else None for inputs_ in inputs]

    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_all_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def optimize(self, loss, update_param):
        loss.backward()
        if update_param:
            if self.expe.config.gclip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), self.expe.config.gclip)
            self.opt.step()
            if self.expe.config.wstep:
                self.scheduler.step()
            self.opt.zero_grad()

    def init_optimizer(self, opt_type, learning_rate, weight_decay):
        if opt_type.lower() == "adam":
            optimizer = torch.optim.Adam
        elif opt_type.lower() == "adamw":
            optimizer = torch.optim.AdamW
        elif opt_type.lower() == "rmsprop":
            optimizer = torch.optim.RMSprop
        elif opt_type.lower() == "sgd":
            optimizer = torch.optim.SGD
        else:
            raise NotImplementedError("invalid optimizer: {}".format(opt_type))

        if weight_decay:
            no_decay = ["bias", "layer_norm", "norm"]
            self.expe.log.info("following parameters do not have weight decay")
            self.expe.log.info("\n".join([n for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]))
            self.expe.log.info("*" * 20)
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": weight_decay,
                },
                {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            ]
        else:
            optimizer_grouped_parameters = filter(lambda p: p.requires_grad, self.parameters())

        opt = optimizer(
            params=optimizer_grouped_parameters,
            # weight_decay=weight_decay,
            lr=learning_rate)

        if self.expe.config.wstep:
            self.scheduler = \
                model_utils.get_linear_schedule_with_warmup(
                    opt, self.expe.config.wstep,
                    self.expe.config.n_epoch * self.iter_per_epoch)
            self.expe.log.info("training with learning rate scheduler - iterations per epoch: {}, total epochs: {}"
                               .format(self.iter_per_epoch, self.expe.config.n_epoch))
        return opt

    def save(self, dev_bleu, test_bleu, epoch, iteration=None, name="best"):
        save_path = os.path.join(self.expe.experiment_dir, name + ".ckpt")
        checkpoint = {
            "dev_bleu": dev_bleu,
            "test_bleu": test_bleu,
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": self.state_dict(),
            "opt_state_dict": self.opt.state_dict(),
            "config": self.expe.config
        }
        if self.expe.config.wstep:
            checkpoint["lr_scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, save_path)
        self.expe.log.info("model saved to {}".format(save_path))

    def load(self, checkpointed_state_dict=None, name="best", path=None, strict=True):
        if checkpointed_state_dict is None:
            base_path = self.expe.experiment_dir if path is None else path
            save_path = os.path.join(base_path, name + ".ckpt")
            checkpoint = torch.load(save_path,
                                    map_location=lambda storage, loc: storage)
            self.load_state_dict(checkpoint['state_dict'], strict=strict)
            self.opt.load_state_dict(checkpoint.get("opt_state_dict"))
            if self.expe.config.wstep:
                self.scheduler.load_state_dict(
                    checkpoint["lr_scheduler_state_dict"])
            self.expe.log.info("model loaded from {}. strict={}".format(save_path, strict))
            self.to(self.device)
            for state in self.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.expe.log.info("transferred model to {}".format(self.device))
            return checkpoint.get('epoch', 0), \
                checkpoint.get('iteration', 0), \
                checkpoint.get('dev_bleu', 0), \
                checkpoint.get('test_bleu', 0)
        else:
            self.load_state_dict(checkpointed_state_dict, strict=strict)
            self.expe.log.info("model loaded from checkpoint. strict={}".format(strict))
            self.to(self.device)
            self.expe.log.info("transferred model to {}".format(self.device))


class LongformerSeq2Seq(Base):
    @auto_init_pytorch
    @auto_init_args
    def __init__(self, vocab_size, iter_per_epoch, experiment):
        super(LongformerSeq2Seq, self).__init__(iter_per_epoch, experiment)
        self.config = {
            "attention_mode": "longformer",
            "attention_probs_dropout_prob": 0.1,
            "attention_window": [
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512,
                512
            ],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "ignore_attention_mask": False,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "max_position_embeddings": 16384,
            "model_type": "longformer",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "sep_token_id": 2,
            "type_vocab_size": 1,
            "vocab_size": 50265
        }
        self.config["vocab_size"] = vocab_size
        self.config["num_hidden_layers"] = self.expe.config.elayer
        self.config["hidden_size"] = self.expe.config.hsize
        self.config["attention_window"] = [512] * self.expe.config.elayer
        self.config["intermediate_size"] = self.expe.config.hsize * self.expe.config.fns
        self.config["num_attention_heads"] = 8
        lfconfig = LongformerConfig(**self.config)
        lfconfig.gradient_checkpointing = self.expe.config.gradient_checkpointing
        self.encoder = LongformerModel(lfconfig)

        self.decoder = CacheTransformer(
            n_words=vocab_size,
            bos_index=1,
            eos_index=2,
            pad_index=0,
            emb_dim=self.expe.config.hsize,
            ffnet_dim=self.expe.config.hsize * self.expe.config.fns,
            n_heads=8,
            n_layers=self.expe.config.dlayer,
            dropout=0.1,
            embed=None,
            share_embedding=True,
            attention_dropout=0.1,
            max_leng=1024 + 2,
            use_copy=False,
            alignment_heads=0,
            if_gelu=self.expe.config.gelu)

    def forward(self, input_data, input_attn_mask, global_attn_mask, tgt_inp, tgt_mask, tgt_tgt, eot_idx, eot_mask):
        input_data, global_attn_mask, input_attn_mask, tgt_inp, tgt_mask, tgt_tgt, eot_idx, eot_mask = \
            self.to_tensors(input_data, global_attn_mask, input_attn_mask, tgt_inp, tgt_mask, tgt_tgt, eot_idx, eot_mask)
        data_vec = self.encoder(input_ids=input_data.long(), attention_mask=input_attn_mask, global_attention_mask=global_attn_mask)[0]
        data_vec = data_vec[eot_idx.bool()]

        data_vec = data_vec.reshape(eot_mask.shape[0], eot_mask.shape[1], self.expe.config.hsize)

        pred_probs, _ = self.decoder.fwd(x=tgt_inp, src_enc=data_vec, src_mask=eot_mask)

        if self.expe.config.ls:
            loss_fn = model_utils.LabelSmoothingLoss(classes=self.vocab_size, smoothing=self.expe.config.ls, dim=-1)
            loss = loss_fn(pred_probs, tgt_tgt.long())
            loss = loss * tgt_mask
            loss = loss.sum(1) / tgt_mask.sum(1)
        else:
            batch_size, seq_len, vocab_size = pred_probs.shape
            tgt_mask = tgt_mask.reshape(-1)
            pred_probs = pred_probs.reshape(batch_size * seq_len, vocab_size)
            tgt = tgt_tgt.reshape(-1)

            loss = F.cross_entropy(pred_probs, tgt.long(), reduction="none")
            loss = loss * tgt_mask
            loss = loss.reshape(batch_size, seq_len).sum(1) / (tgt_mask.reshape(batch_size, seq_len)).sum(1)
        loss = loss.mean(0)
        return loss
