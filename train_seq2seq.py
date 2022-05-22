import sys
import train_helper
import data_utils
import config
import models

BEST_DEV_LOSS = TEST_LOSS = - 10000


def run(e):
    global BEST_DEV_LOSS, TEST_LOSS

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    data_processor = data_utils.DataProcessor(
        train_path=e.config.train_path,
        dev_path=e.config.dev_path,
        test_path=e.config.test_path,
        bpe_vocab=e.config.vocab_file,
        experiment=e)
    data = data_processor.process()

    train_batch = data_utils.Minibatcher(
        data=data.train_data,
        batch_size=e.config.batch_size,
        save_dir=e.experiment_dir,
        filename="minibatcher.ckpt",
        vocab=data.vocab,
        log=e.log,
        mask_fraction=e.config.mf,
        is_eval=False,
        if_lm=False,
        random_perm=e.config.random_perm,
        if_lf=e.config.model_type in ["longformer"],
        verbose=True)

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.model_type == "longformer":
        model_type = models.LongformerSeq2Seq
    else:
        raise ValueError("invalid model type: {}".format(e.config.model_type))

    model = model_type(
        vocab_size=len(data.vocab),
        iter_per_epoch=len(train_batch.idx_pool) // e.config.gcs,
        experiment=e)

    start_epoch = true_it = 0
    if e.config.resume:
        start_epoch, _, BEST_DEV_LOSS, TEST_LOSS = \
            model.load(name="latest")
        e.log.info(
            "resumed from previous checkpoint: start epoch: {}, "
            "iteration: {}, best dev loss: {:.3f}, test loss: {:.3f}."
            .format(start_epoch, true_it, BEST_DEV_LOSS, TEST_LOSS))

    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    dev_eval = train_helper.Evaluator(
        model=model,
        data=data.dev_data,
        vocab=data.vocab,
        if_lm=False,
        if_lf=e.config.model_type in ["longformer", "basic"],
        eval_batch_size=e.config.eval_batch_size,
        experiment=e)

    e.log.info("Training start ...")
    train_stats = train_helper.Tracker(["loss"])

    for epoch in range(start_epoch, e.config.n_epoch):
        for it, (input_data, input_attn_mask, global_attn_mask, tgt_inp, tgt_mask,
                tgt_tgt, eot_idx, eot_mask, _) in enumerate(train_batch):

            model.train()
            curr_it = train_batch.init_pointer + it + 1 + epoch * len(train_batch.idx_pool)
            true_it = curr_it // e.config.gcs
            full_division = ((curr_it % e.config.gcs) == 0) or (curr_it % len(train_batch.idx_pool) == 0)

            loss = model(input_data, input_attn_mask, global_attn_mask, tgt_inp, tgt_mask, tgt_tgt, eot_idx, eot_mask)
            model.optimize(loss / e.config.gcs, update_param=full_division)
            train_stats.update({"loss": loss}, len(input_data))
            if e.config.auto_disconnect and full_division:
                if e.elapsed_time > 3.5:
                    e.log.info("elapsed time: {:.3}(h), automatically exiting the program...".format(e.elapsed_time))
                    train_batch.save()
                    model.save(
                        dev_bleu=BEST_DEV_LOSS,
                        test_bleu=TEST_LOSS,
                        iteration=true_it,
                        epoch=epoch,
                        name="latest")
                    sys.exit()

            if (true_it % e.config.print_every == 0 or
                    curr_it % len(train_batch.idx_pool) == 0) and full_division:
                curr_lr = model.scheduler.get_last_lr()[0] if e.config.wstep \
                    else e.config.lr
                summarization = train_stats.summarize(
                    "epoch: {}, it: {} (max: {}), lr: {:.4e}"
                    .format(epoch, true_it % (len(train_batch.idx_pool) // e.config.gcs), len(train_batch.idx_pool) // e.config.gcs, curr_lr))
                e.log.info(summarization)

                train_stats.reset()

            if (true_it % e.config.eval_every == 0 or
                    curr_it % len(train_batch.idx_pool) == 0) and full_division:

                train_batch.save()
                model.save(
                    dev_bleu=BEST_DEV_LOSS,
                    test_bleu=TEST_LOSS,
                    iteration=true_it,
                    epoch=epoch,
                    name="latest")

                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                dev_loss = dev_eval.evaluate("gen_dev")

                e.log.info("*" * 25 + " DEV SET EVALUATION " + "*" * 25)

                if BEST_DEV_LOSS < dev_loss:
                    BEST_DEV_LOSS = dev_loss

                    model.save(
                        dev_bleu=BEST_DEV_LOSS,
                        test_bleu=TEST_LOSS,
                        iteration=true_it,
                        epoch=epoch,
                        name="best")

                e.log.info("best dev loss: {:.4f}, test loss: {:.4f}"
                           .format(BEST_DEV_LOSS, TEST_LOSS))
                train_stats.reset()
            if (true_it % e.config.save_every == 0) and full_division:
                train_batch.save()
                model.save(
                    dev_bleu=BEST_DEV_LOSS,
                    test_bleu=TEST_LOSS,
                    iteration=true_it,
                    epoch=epoch,
                    name="latest")

        train_batch.save()
        model.save(
            dev_bleu=BEST_DEV_LOSS,
            test_bleu=TEST_LOSS,
            iteration=true_it,
            epoch=epoch + 1,
            name="latest")

        time_per_epoch = (e.elapsed_time / (epoch - start_epoch + 1))
        time_in_need = time_per_epoch * (e.config.n_epoch - epoch - 1)
        e.log.info("elapsed time: {:.2f}(h), "
                   "time per epoch: {:.2f}(h), "
                   "time needed to finish: {:.2f}(h)"
                   .format(e.elapsed_time, time_per_epoch, time_in_need))
        train_stats.reset()


if __name__ == '__main__':

    PARSED_CONFIG = config.get_base_parser().parse_args()

    def exit_handler(*args):
        print(PARSED_CONFIG)
        print("best dev loss: {:.4f}, test loss: {:.4f}"
              .format(BEST_DEV_LOSS, TEST_LOSS))
        sys.exit()

    train_helper.register_exit_handler(exit_handler)

    with train_helper.Experiment(PARSED_CONFIG,
                                 PARSED_CONFIG.save_prefix) as exp:

        exp.log.info("*" * 25 + " ARGS " + "*" * 25)
        exp.log.info(PARSED_CONFIG)
        exp.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(exp)
