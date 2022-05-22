import time
import logging
import argparse
import os
import signal
import subprocess
import torch
import numpy as np
import data_utils
from config import get_base_parser, MULTI_BLEU_PERL
from decorators import auto_init_args


def register_exit_handler(exit_handler):
    import atexit

    atexit.register(exit_handler)
    signal.signal(signal.SIGTERM, exit_handler)
    signal.signal(signal.SIGINT, exit_handler)


def run_multi_bleu(input_file, reference_file):
    bleu_output = subprocess.check_output(
        "./{} {} < {}".format(MULTI_BLEU_PERL, reference_file, input_file),
        stderr=subprocess.STDOUT, shell=True).decode('utf-8')
    bleu = float(
        bleu_output.strip().split("\n")[-1]
        .split(",")[0].split("=")[1][1:])
    return bleu


class Tracker:
    @auto_init_args
    def __init__(self, names):
        assert len(names) > 0
        self.reset()

    def __getitem__(self, name):
        return self.values.get(name, 0) / self.counts.get(name, 0) if self.counts.get(name, 0) else 0

    def __len__(self):
        return len(self.names)

    def reset(self):
        self.values = dict({name: 0. for name in self.names})
        self.counts = dict({name: 0. for name in self.names})
        # self.counter = 0
        self.create_time = time.time()

    def update(self, named_values, count):
        """
        named_values: dictionary with each item as name: value
        """
        # self.counter += count
        for name, value in named_values.items():
            self.counts[name] += count
            self.values[name] += value.item() * count if isinstance(value, (torch.Tensor, np.ndarray)) else value * count

    def summarize(self, output=""):
        if output:
            output += ", "
        for name in self.names:
            output += "{}: {:.3f}, ".format(
                name, self.values[name] / self.counts[name] if self.counts[name] else 0)
        output += "elapsed time: {:.1f}(s)".format(
            time.time() - self.create_time)
        return output

    @property
    def stats(self):
        return {n: v / self.counts[n] if self.counts[n] else 0
                for n, v in self.values.items()}


class Experiment:
    @auto_init_args
    def __init__(self, config, experiments_prefix, forced_debug=False, logfile_name="log"):
        """Create a new Experiment instance.

        Modified based on: https://github.com/ex4sperans/mag

        Args:
            logfile_name: str, naming for log file. This can be useful to
                separate logs for different runs on the same experiment
            experiments_prefix: str, a prefix to the path where
                experiment will be saved
        """

        # get all defaults
        all_defaults = {}
        for key in vars(config):
            all_defaults[key] = get_base_parser().get_default(key)

        self.default_config = all_defaults

        config.resume = False
        if not config.debug and not forced_debug:
            if os.path.isdir(self.experiment_dir):
                print("log exists: {}".format(self.experiment_dir))
                config.resume = True

            print(config)
            self._makedir()

        # self._make_misc_dir()

    def _makedir(self):
        os.makedirs(self.experiment_dir, exist_ok=True)

    def _make_misc_dir(self):
        os.makedirs(self.config.vocab_file, exist_ok=True)

    @property
    def log_file(self):
        return os.path.join(self.experiment_dir, self.logfile_name)

    @property
    def experiment_dir(self):
        if self.config.debug or self.forced_debug:
            return "./"
        else:
            # get namespace for each group of args
            arg_g = dict()
            for group in get_base_parser()._action_groups:
                group_d = {a.dest: self.default_config.get(a.dest, None)
                           for a in group._group_actions}
                arg_g[group.title] = argparse.Namespace(**group_d)

            # skip default value
            identifier = ""
            for key, value in sorted(vars(arg_g["model_configs"]).items()):
                if hasattr(self.config, key) and getattr(self.config, key) != value:
                    identifier += key + str(getattr(self.config, key))
            return os.path.join(self.experiments_prefix, identifier)

    def register_directory(self, dirname):
        directory = os.path.join(self.experiment_dir, dirname)
        os.makedirs(directory, exist_ok=True)
        setattr(self, dirname, directory)

    def _register_existing_directories(self):
        for item in os.listdir(self.experiment_dir):
            fullpath = os.path.join(self.experiment_dir, item)
            if os.path.isdir(fullpath):
                setattr(self, item, fullpath)

    def __enter__(self):
        import socket
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if self.config.debug or self.forced_debug:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')
        else:
            print("log saving to", self.log_file)
            logging.basicConfig(
                filename=self.log_file,
                filemode='a+', level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M')

        self.log = logging.getLogger()
        self.start_time = time.time()
        self.log.info("HostName: {}".format(socket.gethostname()))
        return self

    def __exit__(self, *args):
        logging.shutdown()

    @property
    def elapsed_time(self):
        return (time.time() - self.start_time) / 3600


class Evaluator:
    def __init__(self, model, eval_batch_size, data, vocab, if_lm, if_lf, experiment):
        self.model = model
        self.expe = experiment

        self.data_iterator = data_utils.Minibatcher(
            batch_size=eval_batch_size,
            data=data,
            is_eval=True,
            save_dir=None,
            vocab=vocab,
            verbose=False,
            if_lm=if_lm,
            if_lf=if_lf,
            mask_fraction=0.0,
            random_perm=True,
            vocab_size=len(vocab),
            filename="devtesteval_minibatcher.ckpt",
            log=self.expe.log)
        self.eval_stats = Tracker(["loss"])

    def evaluate(self, gen_fn):
        self.model.eval()
        for nbatch, rets in enumerate(self.data_iterator):
            if nbatch and nbatch % (len(self.data_iterator) // 10 + 1) == 0:
                self.expe.log.info("evaluating progress: {}/{} = {:.2f} %".format(nbatch, len(self.data_iterator), nbatch / (len(self.data_iterator) + 1) * 100))
            with torch.no_grad():
                batch_loss = self.model(*rets[:-1])
            self.eval_stats.update({"loss": batch_loss}, len(rets[0]))
        eval_result = self.eval_stats["loss"]
        self.expe.log.info("loss: {:.4f}".format(eval_result))
        self.eval_stats.reset()
        return - eval_result
