import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.
    These networks consider copying words
    directly from the source sequence.
    The copy generator is an extended version of the standard
    generator that computes three values.
    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.
    The model returns a distribution over the extend dictionary,
    computed as
    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`
    .. mermaid::
       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O
    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, pad_idx=0):
        super(CopyGenerator, self).__init__()
        # self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def forward(self, hidden, orig_prob, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.
        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        # batch_by_tlen, _ = hidden.size()
        # batch_by_tlen_, slen = attn.size()
        onehot_src_map = F.one_hot(src_map.long(), torch.max(src_map).long() + 1)
        batch, slen, cvocab = onehot_src_map.size()
        # aeq(batch_by_tlen, batch_by_tlen_)
        # aeq(slen, slen_)
        # bs, sql, vsize = orig_prob.shape
        prob = torch.softmax(orig_prob, 1)
        # Original probabilities.
        # logits = self.linear(hidden)
        # logits[:, self.pad_idx] = -float('inf')
        # prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden))
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy)
        mul_attn = torch.mul(attn, p_copy)
        copy_prob = torch.bmm(
            mul_attn.view(batch, -1, slen), # batch size x tgt len x src len
            onehot_src_map.float()) # batch size x src len x cvocab
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        return out_prob, copy_prob


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=0, eps=1e-10):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def forward(self, scores, align, target, src_tgt_map, label_smoothing):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            src_tgt_map: batch size x extended vocab size ([b, src vocab idx] = tgt vocab idx)
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        bs, sqlen = align.shape
        flat_align = align.reshape(-1)
        flat_target = target.reshape(-1)

        if label_smoothing:
            out_prob, copy_prob = scores

            scores, copy_mask = collapse_copy_scores(torch.cat([out_prob, copy_prob], 1), src_tgt_map, self.vocab_size)

            label_mask = copy_mask

            confidence = 1 - label_smoothing
            smoothing = label_smoothing / label_mask.sum(1, keepdim=True)

            tgt_labels = torch.zeros_like(scores)
            copy_labels = torch.zeros_like(scores)

            tgt_labels.scatter_(1, flat_target.unsqueeze(1).long(), 1)

            copy_ix = flat_align.unsqueeze(1) + self.vocab_size
            copy_labels.scatter_(1, copy_ix.long(), 1)
            non_copy = flat_align == self.unk_index
            if not self.force_copy:
                non_copy = non_copy | (flat_target != self.unk_index)

            final_labels = torch.where(
                non_copy.unsqueeze(1), tgt_labels, copy_labels
            )

            final_labels = final_labels * (confidence - smoothing) + smoothing
            final_labels = final_labels * label_mask

            # OLD
            # out_copy_prob = torch.zeros_like(out_prob)
            #
            # non_neg_src_tgt_map = src_tgt_map.clone()
            # non_neg_src_tgt_map[non_neg_src_tgt_map == -1] = 0
            #
            # non_neg_src_tgt_map = non_neg_src_tgt_map.unsqueeze(1).expand(-1, sqlen, -1).reshape(bs * sqlen, non_neg_src_tgt_map.shape[1])
            # out_copy_prob.scatter_(1, non_neg_src_tgt_map.long(), copy_prob)
            # out_copy_prob[:, 0] = 0
            #
            # scores = torch.cat([out_prob + out_copy_prob, copy_prob], 1)
            #
            # confidence = 1 - label_smoothing
            # smoothing = label_smoothing / self.vocab_size
            # tgt_labels = torch.zeros_like(scores)
            # copy_labels = torch.zeros_like(scores)
            # tgt_labels.scatter_(1, flat_target.unsqueeze(1).long(), 1)
            # # tgt_labels[:, 0] = 0
            # copy_ix = flat_align.unsqueeze(1) + self.vocab_size
            # copy_labels.scatter_(1, copy_ix.long(), 1)
            # # copy_labels[:, self.vocab_size] = 0
            # # soft_labels[align == self.unk_index] = smoothing
            # non_copy = flat_align == self.unk_index
            # if not self.force_copy:
            #     non_copy = non_copy | (flat_target != self.unk_index)
            #
            # final_labels = torch.where(
            #     non_copy.unsqueeze(1), tgt_labels, copy_labels
            # )
            #
            # use_out_label = torch.ones_like(out_prob)
            # not_use_out_label = torch.zeros_like(out_prob)
            # use_copy_label = torch.ones_like(copy_prob) * (src_tgt_map != -1).float()
            # not_use_copy_label = torch.zeros_like(copy_prob)
            #
            # label_mask = torch.where(
            #     non_copy.unsqueeze(1), torch.cat([use_out_label, not_use_copy_label], 1), torch.cat([not_use_out_label, use_copy_label], 1)
            # )
            #
            # final_labels = final_labels * (confidence - smoothing) + smoothing
            # final_labels = final_labels * label_mask
            loss = torch.sum(- (scores + self.eps).log() * final_labels, dim=1)
        else:
            scores = torch.cat(scores, 1)
            # probabilities assigned by the model to the gold targets
            vocab_probs = scores.gather(1, flat_target.unsqueeze(1).long()).squeeze(1)

            # probability of tokens copied from source
            copy_ix = flat_align.unsqueeze(1) + self.vocab_size
            copy_tok_probs = scores.gather(1, copy_ix.long()).squeeze(1)
            # Set scores for unk to 0 and add eps
            copy_tok_probs[flat_align == self.unk_index] = 0
            copy_tok_probs = copy_tok_probs + self.eps  # to avoid -inf logs

            # find the indices in which you do not use the copy mechanism
            non_copy = flat_align == self.unk_index
            if not self.force_copy:
                non_copy = non_copy | (flat_target != self.unk_index)

            probs = torch.where(
                non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
            )

            loss = -(probs + self.eps).log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[flat_target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(nn.Module):
    """Copy Generator Loss Computation."""
    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length):
        super(CopyGeneratorLossCompute, self).__init__()
        self.criterion = criterion
        self.generator = generator
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))

    def compute_loss(self, batch, output, target, copy_attn, align,
                      std_attn=None, coverage_attn=None):
        """Compute the loss.
        The args must match :func:`self._make_shard_state()`.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align = align.view(-1)
        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), batch.src_map
        )
        loss = self.criterion(scores, align, target)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss


def collapse_copy_scores(scores, src_tgt_vocab_map, vocab_size, keep_src_vocab_unk=True):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.

    src_tgt_vocab_map: batch size x src tgt vocab map size
    scores: (batch size * seq len) x dynamic vocab size
    """
    batch_size = src_tgt_vocab_map.shape[0]
    batch_size_by_seq_len = scores.shape[0]
    assert batch_size_by_seq_len % batch_size == 0, batch_size_by_seq_len % batch_size

    seq_len = batch_size_by_seq_len // batch_size
    offset = vocab_size

    fill = src_tgt_vocab_map[:, 1:].unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)
    pad = torch.ones(batch_size_by_seq_len, scores.shape[1] - fill.shape[1]).to(fill.device)
    padded_fill = torch.cat([pad, fill], 1)
    scores[padded_fill == -1] = 0

    non_neg_src_tgt_vocab_map = src_tgt_vocab_map.clone()
    non_neg_src_tgt_vocab_map[non_neg_src_tgt_vocab_map == -1] = 0

    blank = (offset + torch.arange(1, non_neg_src_tgt_vocab_map.shape[1]).unsqueeze(0).expand(batch_size_by_seq_len, -1) ).long()
    blank = blank.to(scores.device)
    fill = non_neg_src_tgt_vocab_map[:, 1:].long().unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)

    add_scores = torch.zeros_like(scores)
    indexed_scores = scores.gather(1, blank)
    add_scores.scatter_(1, fill, indexed_scores)
    if keep_src_vocab_unk:
        add_scores[:, 0] = 0
    scores = scores + add_scores
    # scores.index_add_(1, fill, scores.index_select(1, blank))

    scores_mask = torch.ones_like(scores)
    scores_mask.scatter_(1, blank, 0.0)
    # add_scores.scatter_(1, torch.nonzero(zero_blank), 0)
    # x_axis = torch.arange(blank.shape[0]).unsqueeze(1).to(scores.device).expand_as(blank)

    if keep_src_vocab_unk:
        # fill = src_tgt_vocab_map[:, 1:].unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, -1)
        # pad = torch.ones(batch_size_by_seq_len, scores_mask.shape[1] - fill.shape[1]).to(fill.device)
        # padded_fill = torch.cat([pad, fill], 1)
        # x_axis = x_axis[fill == 0].view(-1)
        # unblank = blank[fill == 0]#.view(-1)
        # print("unblank", unblank.shape)
        # print("scores_mask", scores_mask.shape)
        # scores_mask.scatter_(1, unblank, 1.0)
        scores_mask[padded_fill == 0] = 1
        # scores_mask[x_axis, y_axis] = 1

    # add_scores[add_scores == 1] = -float('Inf')
    scores = scores * scores_mask

    # scores.index_fill_(1, blank, -float('Inf'))
    # for b in range(scores.size(0)):
    #     blank = []
    #     fill = []
    #
    #     # if src_vocabs is None:
    #     #     src_vocab = batch.src_ex_vocab[b]
    #     # else:
    #     #     batch_id = batch_offset[b] if batch_offset is not None else b
    #     #     index = batch.indices.data[batch_id]
    #     #     src_vocab = src_vocabs[index]
    #
    #     for i in range(1, len(src_tgt_vocab_map[b])):
    #         ti = src_tgt_vocab_map[b][i]
    #         if ti != 0:
    #             blank.append(offset + i)
    #             fill.append(ti)
    #     if blank:
    #         blank = torch.Tensor(blank).type_as(scores).long()
    #         fill = torch.Tensor(fill).type_as(scores).long()
    #         score = scores[b]
    #         score.index_add_(0, fill, score.index_select(0, blank))
    #         score.index_fill_(0, blank, -float('Inf'))
    return scores, scores_mask


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = F.one_hot(target.long(), self.cls)
            # true_dist = torch.zeros_like(pred)
            true_dist = true_dist * self.confidence
            true_dist = true_dist + self.smoothing / (self.cls - 1)
            # true_dist.fill_()
            # true_dist.scatter_(2, , self.confidence)
        return torch.sum(-true_dist * pred, dim=self.dim)
