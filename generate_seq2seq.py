import sys
import train_helper
import data_utils
import config
import os
import torch
import models
from tqdm import tqdm

BEST_DEV_LOSS = TEST_LOSS = - 10000


def run(e):
    global BEST_DEV_LOSS, TEST_LOSS

    save_path = os.path.join(
        e.config.model_dir if e.config.model_dir else e.experiment_dir,
        e.config.ckpt_name)
    checkpoint = torch.load(save_path,
                            map_location=lambda storage, loc: storage)
    e.log.info("loaded from: {}".format(save_path))

    class dummy_exp:
        pass
    model_exp = dummy_exp()
    model_exp.log = e.log
    checkpoint["config"].debug = False
    checkpoint["config"].resume = True

    model_exp.config = checkpoint["config"]
    model_exp.experiment_dir = e.config.model_dir \
        if e.config.model_dir else e.experiment_dir
    model_exp.config.beam_size = e.config.beam_size
    model_exp.config.max_gen_len = e.config.max_gen_len
    model_exp.config.min_gen_len = e.config.min_gen_len
    model_exp.config.gradient_checkpointing = e.config.gradient_checkpointing
    for name in dir(e.config):
        if name.startswith("__"):
            continue
        if name not in dir(model_exp.config):
            value = getattr(e.config, name)
            e.log.info("update {} to {}".format(name, value))
            setattr(model_exp.config, name, value)

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_processor = data_utils.DataProcessor(
        train_path=e.config.train_path,
        dev_path=e.config.dev_path,
        test_path=e.config.test_path,
        bpe_vocab=e.config.vocab_file,
        experiment=model_exp)
    data = data_processor.process()

    eval_batch = data_utils.Minibatcher(
        data=data.dev_data,
        batch_size=e.config.batch_size,
        save_dir=e.experiment_dir,
        filename="minibatcher.ckpt",
        vocab=data.vocab,
        log=e.log,
        if_lm=False,
        mask_fraction=0.0,
        random_perm=False,
        if_lf=model_exp.config.model_type in ["longformer"],
        is_eval=True,
        verbose=True)

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if model_exp.config.model_type == "longformer":
        model_type = models.LongformerSeq2Seq
    else:
        raise ValueError("invalid model type: {}".format(
            model_exp.config.model_type))

    model = model_type(
        vocab_size=len(data.vocab),
        iter_per_epoch=100,
        experiment=model_exp)

    model.load(checkpointed_state_dict=checkpoint["state_dict"])
    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    model.eval()
    gen_fn = e.config.gen_prefix
    output_path = e.config.model_dir if e.config.model_dir else e.experiment_dir
    all_gen = {}
    gen_length = []
    with torch.no_grad():
        for nbatch, (input_data, input_attn_mask, global_attn_mask, tgt_inp, tgt_mask,
                tgt_tgt, eot_idx, eot_mask, batch_idx) in tqdm(enumerate(eval_batch), total=len(eval_batch)):
            if nbatch and nbatch % (len(eval_batch) // 10 + 1) == 0:
                e.log.info("evaluating progress: {}/{} = {:.2f} %".format(nbatch, len(eval_batch), nbatch / (len(eval_batch) + 1) * 100))
            # if e.config.debug:
            #     e.log.info("**** INPUT: {} ****".format(dev_inst["idx"]))
            #     e.log.info(" ".join([data.inv_vocab[idx] for idx in inp_ids[:100]]).replace("@@ ", "").replace("@@", ""))
            #     e.log.info("**** Reference ****")
            #     e.log.info(" ".join([data.inv_vocab[idx] for idx in dev_inst["tgt_ids"][:100]]).replace("@@ ", "").replace("@@", ""))
            #     e.log.info("************")

            input_data, input_attn_mask, global_attn_mask, eot_idx, eot_mask = \
                model.to_tensors(input_data, input_attn_mask, global_attn_mask, eot_idx, eot_mask)
            if model_exp.config.model_type == "longformer":
                data_vec = model.encoder(input_ids=input_data.long(),
                    attention_mask=input_attn_mask, global_attention_mask=global_attn_mask)[0]

            data_vec = data_vec[eot_idx.bool()]

            data_vec = data_vec.reshape(
                eot_mask.shape[0], eot_mask.shape[1], model_exp.config.hsize)
            out_ids = model.decoder._generate_beam(
                src_enc=data_vec,
                src_mask=eot_mask,
                beam_size=e.config.beam_size,
                trigram_blocking=e.config.trigram_blocking,
                min_len=e.config.min_gen_len,
                max_len=e.config.max_gen_len)[0].tolist()
            for out_id, idx in zip(out_ids, batch_idx):
                curr_gen = []
                for wid in out_id:
                    if wid == data.vocab["<bos>"]:
                        continue
                    if wid == data.vocab["<eos>"]:
                        break
                    curr_gen.append(data.inv_vocab[wid])
                all_gen[idx] = " ".join(curr_gen).replace("@@ ", "").replace("@@", "")
                gen_length.append(len(curr_gen))
                if e.config.debug:
                    e.log.info("**** GENERATION: {} (number of subword: {}) ****".format(idx, gen_length[-1]))
                    e.log.info(" ".join(curr_gen).replace("@@ ", "").replace("@@", ""))
                    e.log.info("**** REFERENCE: {}".format(data.dev_data[idx]["tgt_tok"]))
                    e.log.info("************")
    file_name = os.path.join(output_path, gen_fn + ".txt")
    ref_file_name = os.path.join(output_path, gen_fn + "_ref.txt")
    with open(file_name, "w") as fp, open(ref_file_name, "w") as ref_fp:
        for gen in sorted(all_gen.items(), key=lambda x: x[0]):
            fp.write(gen[1] + "\n")
            ref_fp.write(data.dev_data[gen[0]]["tgt_tok"] + "\n")
    e.log.info("generations saved to: {}".format(file_name))

    bleu_score = train_helper.run_multi_bleu(file_name, ref_file_name)
    e.log.info("avg generation steps: {}, bleu: {:.4f}".format(
        sum(gen_length) / len(gen_length), bleu_score))


if __name__ == '__main__':

    PARSED_CONFIG = config.get_base_parser().parse_args()

    def exit_handler(*args):
        print(PARSED_CONFIG)
        print("best dev loss: {:.4f}, test loss: {:.4f}"
              .format(BEST_DEV_LOSS, TEST_LOSS))
        sys.exit()

    train_helper.register_exit_handler(exit_handler)

    with train_helper.Experiment(PARSED_CONFIG,
                                 PARSED_CONFIG.save_prefix,
                                 forced_debug=True) as exp:

        exp.log.info("*" * 25 + " ARGS " + "*" * 25) #pylint: disable=W1201
        exp.log.info(PARSED_CONFIG)
        exp.log.info("*" * 25 + " ARGS " + "*" * 25) #pylint: disable=W1201

        run(exp)
