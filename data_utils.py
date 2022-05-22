import os
import pickle
import json
import statistics
from decorators import auto_init_args
from config import UNK_IDX, UNK_WORD
import numpy as np
import random


class DataHolder:
    @auto_init_args
    def __init__(self, train_data, dev_data, test_data, vocab):
        self.inv_vocab = {i: w for w, i in vocab.items()}


class DataProcessor:
    @auto_init_args
    def __init__(self, train_path, dev_path, test_path, bpe_vocab, experiment):
        self.expe = experiment

    def process(self):

        vocab = {UNK_WORD: UNK_IDX}
        for special_word in ["<bos>", "<eos>", "[SCENE_BREAK]", "[NEXT_ON]",
                             "[PREVIOUSLY_ON]", "[END_OF_TURN]", "<mask>"]:
            vocab[special_word] = len(vocab)

        with open(self.bpe_vocab) as fp:
            for line in fp:
                w, f = line.strip().split()
                if int(f) > 50:
                    vocab[w] = len(vocab)

        if self.expe.config.glossary_file:
            with open(self.expe.config.glossary_file) as fp:
                for line in fp:
                    w, f = line.strip().split("\t")
                    if w not in vocab and int(f) >= 10:
                        vocab[w] = len(vocab)

        self.expe.log.info("vocab size: {}".format(len(vocab)))

        train_data = self._load_instance(
            self.train_path, vocab, False,
            max_src_len=self.expe.config.max_src_len,
            max_tgt_len=self.expe.config.max_tgt_len)

        dev_data = self._load_instance(
            self.dev_path, vocab, True,
            max_src_len=self.expe.config.max_src_len,
            max_tgt_len=self.expe.config.max_tgt_len)

        data = DataHolder(
            train_data=np.array(train_data),
            dev_data=np.array(dev_data),
            test_data=None,
            vocab=vocab)

        return data

    def _load_instance(self, path, vocab, is_test,
                       max_src_len=None, max_tgt_len=None):
        all_instance = []
        src_unks = 0
        tgt_unks = 0
        src_tokens = 0
        tgt_tokens = 0
        src_len = []
        tgt_len = []
        src_per_len = []
        src_n_entry = []
        src_n_entity = []
        tgt_n_entity = []
        src_unique_ent = []
        tgt_unique_ent = []
        with open(path) as fp:
            for nline, line in enumerate(fp):
                if (nline + 1) % 4000 == 0:
                    self.expe.log.info("processed {} lines".format(nline + 1))
                if line.strip():
                    json_data = json.loads(line.strip())
                    src_ids = []
                    src_eot = []
                    src_entity_pos = []
                    src_ent_set = set()
                    src_n_ent = 0
                    src_n_entry.append(len(json_data["Transcript"]))
                    for trans in json_data["Transcript"]:
                        src_per_len_ = 0
                        for w in trans.split()[:max_src_len]:
                            src_tokens += 1
                            src_per_len_ += 1
                            src_eot.append(0)
                            if "ENTITY" in w:
                                src_ent_set.add(w)
                                src_n_ent += 1
                                src_entity_pos.append(len(src_ids))
                                src_ids.append(w)
                            elif w in vocab:
                                src_ids.append(vocab[w])
                            else:
                                src_ids.append(UNK_IDX)
                                src_unks += 1
                        src_per_len.append(src_per_len_)
                        src_ids.append(vocab["[END_OF_TURN]"])
                        src_eot.append(1)
                    src_ids = src_ids[:self.expe.config.max_tot_src_len]
                    src_eot = src_eot[:self.expe.config.max_tot_src_len]
                    src_len.append(len(src_ids))
                    src_n_entity.append(src_n_ent)
                    src_unique_ent.append(len(src_ent_set))
                    tgt_ids = []
                    tgt_ent_set = set()
                    tgt_entity_pos = []
                    tgt_n_ent = 0
                    for w in " ".join(json_data["Recap"]).split()[:max_tgt_len]:
                        tgt_tokens += 1
                        if "ENTITY" in w:
                            tgt_ent_set.add(w)
                            tgt_n_ent += 1
                            tgt_entity_pos.append(len(tgt_ids))
                            tgt_ids.append(w)
                        elif w in vocab:
                            tgt_ids.append(vocab[w])
                        else:
                            tgt_ids.append(UNK_IDX)
                            tgt_unks += 1
                    tgt_len.append(len(tgt_ids))
                    tgt_unique_ent.append(len(tgt_ent_set))
                    tgt_n_entity.append(tgt_n_ent)
                    if src_ids and tgt_ids:
                        all_instance.append(
                            {"idx": len(all_instance),
                             "src_ids": src_ids,
                             "src_eot": src_eot,
                             "tgt_ids": tgt_ids,
                             "src_entity_pos": src_entity_pos,
                             "tgt_entity_pos": tgt_entity_pos}
                        )
                        if is_test:
                            all_instance[-1]["tgt_tok"] = \
                                " ".join(json_data["Recap"]).replace("@@ ", "")
                    else:
                        self.expe.log.info(
                            "EMTPY FILE: {}".format(json_data["filename"]))

        self.expe.log.info("completed processing {} lines".format(len(all_instance)))
        self.expe.log.info("number of src entries - min: {}, max: {}, avg: {:.2f}"
                           .format(min(src_n_entry), max(src_n_entry),
                                   sum(src_n_entry) / len(src_n_entry)))
        self.expe.log.info("length of src per entry - min: {}, max: {}, avg: {:.2f}"
                          .format(min(src_per_len), max(src_per_len),
                                  sum(src_per_len) / len(src_per_len)))
        self.expe.log.info("#unk - src: {} ({:.2f} %) | tgt: {} ({:.2f} %)"
                           .format(src_unks, src_unks / src_tokens * 100,
                                   tgt_unks, tgt_unks / tgt_tokens * 100))
        self.expe.log.info("#anomynized entity - src total: {} (avg: {:.2f}) | tgt total: {} (avg: {:.2f})"
                           .format(sum(src_n_entity), sum(src_n_entity) / len(src_n_entity) if src_n_entity else 0.0,
                                   sum(tgt_n_entity), sum(tgt_n_entity) / len(tgt_n_entity) if tgt_n_entity else 0.0))
        self.expe.log.info("#unique anomynized entity - src total: {} (avg: {:.2f}) | tgt total: {} (avg: {:.2f})"
                           .format(sum(src_unique_ent), sum(src_unique_ent) / len(src_unique_ent) if src_unique_ent else 0.0,
                                   sum(tgt_unique_ent), sum(tgt_unique_ent) / len(tgt_unique_ent) if tgt_unique_ent else 0.0))
        self.expe.log.info("avg length - src: {:.2f} | tgt: {:.2f}"
                           .format(statistics.mean(src_len), statistics.mean(tgt_len)))
        self.expe.log.info("max length - src: {:.2f} | tgt: {:.2f}"
                           .format(max(src_len), max(tgt_len)))
        self.expe.log.info("min length - src: {:.2f} | tgt: {:.2f}"
                           .format(min(src_len), min(tgt_len)))
        return all_instance


class Minibatcher:
    @auto_init_args
    def __init__(self, data, save_dir, log, verbose,
                 filename, is_eval, batch_size, vocab, if_lm, mask_fraction,
                 random_perm,
                 *args, **kwargs):
        self._reset()
        self.load(filename)

    def __len__(self):
        return len(self.idx_pool) - self.init_pointer

    def save(self, filename="minibatcher.ckpt"):
        path = os.path.join(self.save_dir, filename)
        pickle.dump([self.pointer, self.idx_pool], open(path, "wb"))
        if self.verbose:
            self.log.info("minibatcher saved to: {}".format(path))

    def load(self, filename="minibatcher.ckpt"):
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, filename)
        else:
            path = None
        if self.save_dir is not None and os.path.exists(path):
            self.init_pointer, self.idx_pool = pickle.load(open(path, "rb"))
            self.pointer = self.init_pointer
            if self.verbose:
                self.log.info("loaded minibatcher from {}, init pointer: {}"
                              .format(path, self.init_pointer))
        else:
            if self.verbose:
                self.log.info("no minibatcher found at {}".format(path))

    def _reset(self):
        self.pointer = 0
        self.init_pointer = 0
        idx_list = np.arange(len(self.data))
        if self.is_eval:
            idx_list = idx_list[np.argsort([len(d["src_ids"]) for d in self.data])]
        else:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.data), self.batch_size)]

    def _pad_lm(self, data):
        max_len = 32768
        input_data = \
            np.zeros((len(data), max_len)).astype("float32")
        input_attn_mask = \
            np.zeros((len(data), max_len)).astype("float32")
        tgt_data = \
            np.zeros((len(data), max_len)).astype("float32")
        tgt_mask = \
            np.zeros((len(data), max_len)).astype("float32")

        # eot_idx = \
        #     np.zeros((len(data), max_len)).astype("float32")
        # eot_mask = \
        #     np.zeros((len(data), max_eot)).astype("float32")
        for i, d in enumerate(data):

            input_data[i, :len(d["src_ids"]) + len(d["tgt_ids"]) + 1] = \
                np.asarray(list(d["src_ids"]) + [self.vocab["<bos>"]] + list(d["tgt_ids"])).astype("float32")
            input_attn_mask[i, :len(d["src_ids"]) + len(d["tgt_ids"]) + 1] = 1.

            tgt_data[i, :len(d["src_ids"]) + len(d["tgt_ids"]) + 1] = \
                np.asarray(list(d["src_ids"]) + list(d["tgt_ids"]) + [self.vocab["<eos>"]] ).astype("float32")
            tgt_mask[i, len(d["src_ids"]):len(d["src_ids"]) + len(d["tgt_ids"]) + 1] = 1.0

        return [input_data, input_attn_mask, tgt_data, tgt_mask, [d["idx"] for d in data]]

    def _pad_seq2seq(self, data):
        max_src_len = 32768
        max_tgt_len = max([len(d["tgt_ids"]) for d in data])
        max_eot = max([sum(d["src_eot"]) for d in data])

        input_data = \
            np.zeros((len(data), max_src_len)).astype("float32")
        input_attn_mask = \
            np.zeros((len(data), max_src_len)).astype("float32")
        tgt_inp = \
            np.zeros((len(data), max_tgt_len + 1)).astype("float32")
        tgt_tgt = \
            np.zeros((len(data), max_tgt_len + 1)).astype("float32")
        tgt_mask = \
            np.zeros((len(data), max_tgt_len + 1)).astype("float32")

        eot_idx = \
            np.zeros((len(data), max_src_len)).astype("float32")
        eot_mask = \
            np.zeros((len(data), max_eot)).astype("float32")
        for i, d in enumerate(data):

            input_data[i, :len(d["src_ids"])] = \
                np.asarray(list(d["src_ids"])).astype("float32")
            input_attn_mask[i, :len(d["src_ids"])] = 1.

            if self.mask_fraction:
                rands = np.random.rand(len(list(d["tgt_ids"])))
                threshold = np.percentile(rands, self.mask_fraction * 100)
                mask_ids = rands < threshold
                inp_tgt_ids = np.array(list(d["tgt_ids"][:]))
                inp_tgt_ids[mask_ids] = self.vocab["<mask>"]
                inp_tgt_ids = list(inp_tgt_ids)
            else:
                inp_tgt_ids = list(d["tgt_ids"])

            tgt_inp[i, :len(d["tgt_ids"]) + 1] = \
                np.asarray([self.vocab["<bos>"]] + inp_tgt_ids).astype("float32")
            tgt_tgt[i, :len(d["tgt_ids"]) + 1] = \
                    np.asarray(list(d["tgt_ids"])  + [self.vocab["<eos>"]]).astype("float32")
            tgt_mask[i, :len(d["tgt_ids"]) + 1] = 1.0

            eot_idx[i, :len(d["src_eot"])] = d["src_eot"]
            eot_idx[i, len(d["src_eot"]): len(d["src_eot"]) + max_eot - sum(d["src_eot"])] = 1
            eot_mask[i, :sum(d["src_eot"])] = 1

        return [input_data, input_attn_mask, None, tgt_inp, tgt_mask,
                tgt_tgt, eot_idx, eot_mask, [d["idx"] for d in data]]

    def _pad_seq2seq_lf(self, data):
        entity_set = ["ENTITY{}".format(entity_id) for entity_id in range(100)]
        src_entity_set = entity_set[:]
        tgt_entity_set = entity_set[:]
        if self.random_perm:
            random.shuffle(src_entity_set)
            random.shuffle(tgt_entity_set)
        assert len(src_entity_set) == len(tgt_entity_set)
        src2tgt_entity_map = {ent1: ent2 for ent1, ent2 in zip(src_entity_set, tgt_entity_set)}

        max_src_len = max([len(d["src_ids"]) for d in data])
        max_tgt_len = max([len(d["tgt_ids"]) for d in data])
        max_eot = max([sum(d["src_eot"]) for d in data])
        max_pad_eot = max([max_eot - sum(d["src_eot"]) for d in data])

        input_data = \
            np.zeros((len(data), max_src_len + max_pad_eot)).astype("float32")
        input_attn_mask = \
            np.zeros((len(data), max_src_len + max_pad_eot)).astype("float32")
        global_attn_mask = \
            np.zeros((len(data), max_src_len + max_pad_eot)).astype("float32")

        tgt_inp = \
            np.zeros((len(data), max_tgt_len + 1)).astype("float32")
        tgt_tgt = \
            np.zeros((len(data), max_tgt_len + 1)).astype("float32")
        tgt_mask = \
            np.zeros((len(data), max_tgt_len + 1)).astype("float32")

        eot_idx = \
            np.zeros((len(data), max_src_len + max_pad_eot)).astype("float32")
        eot_mask = \
            np.zeros((len(data), max_eot)).astype("float32")
        for i, d in enumerate(data):

            # if self.random_perm:
            proc_src_ids = d["src_ids"][:]
            for ent_id in d["src_entity_pos"]:
                if ent_id < len(proc_src_ids):
                    proc_src_ids[ent_id] = self.vocab[src2tgt_entity_map[proc_src_ids[ent_id]]]

            input_data[i, :len(d["src_ids"])] = \
                np.asarray(list(proc_src_ids)).astype("float32")
            input_attn_mask[i, :len(d["src_ids"])] = 1.

            if self.mask_fraction:
                rands = np.random.rand(len(list(d["tgt_ids"])))
                threshold = np.percentile(rands, self.mask_fraction * 100)
                mask_ids = rands < threshold
                inp_tgt_ids = np.array(list(d["tgt_ids"][:]))
                inp_tgt_ids[mask_ids] = self.vocab["<mask>"]
                inp_tgt_ids = list(inp_tgt_ids)
            else:
                inp_tgt_ids = list(d["tgt_ids"])

            proc_tgt_ids = inp_tgt_ids
            for ent_id in d["tgt_entity_pos"]:
                if ent_id < len(proc_tgt_ids):
                    proc_tgt_ids[ent_id] = self.vocab[src2tgt_entity_map[proc_tgt_ids[ent_id]]]

            tgt_inp[i, :len(d["tgt_ids"]) + 1] = \
                np.asarray([self.vocab["<bos>"]] + proc_tgt_ids).astype("float32")
            tgt_tgt[i, :len(d["tgt_ids"]) + 1] = \
                np.asarray(proc_tgt_ids  + [self.vocab["<eos>"]]).astype("float32")
            tgt_mask[i, :len(d["tgt_ids"]) + 1] = 1.0

            global_attn_mask[i, :len(d["src_eot"])] = d["src_eot"]
            eot_idx[i, :len(d["src_eot"])] = d["src_eot"]
            eot_idx[i, len(d["src_eot"]): len(d["src_eot"]) + max_eot - sum(d["src_eot"])] = 1
            eot_mask[i, :sum(d["src_eot"])] = 1

        return [input_data, input_attn_mask, global_attn_mask, tgt_inp, tgt_mask,
                tgt_tgt, eot_idx, eot_mask, [d["idx"] for d in data]]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        data = self.data[idx]
        self.pointer += 1
        if self.if_lm:
            return self._pad_lm(data)
        else:
            if self.if_lf:
                return self._pad_seq2seq_lf(data)
            else:
                return self._pad_seq2seq(data)
