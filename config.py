import argparse

UNK_IDX = 0
UNK_WORD = "UUUNKKK"

MULTI_BLEU_PERL = 'multi-bleu.perl'


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_base_parser():
    parser = argparse.ArgumentParser(
        description='Dialogue Summarization using PyTorch')
    parser.register('type', 'bool', str2bool)

    basic_group = parser.add_argument_group('basics')
    # Basics
    basic_group.add_argument('--debug', type="bool", default=False,
                             help='activation of debug mode (default: False)')
    basic_group.add_argument('--auto_disconnect', type="bool", default=True,
                             help='for slurm (default: True)')
    basic_group.add_argument('--save_prefix', type=str, default="experiments",
                             help='saving path prefix')
    basic_group.add_argument('--gen_prefix', type=str, default="gen",
                             help='generation saving path prefix')
    basic_group.add_argument('--model_dir', type=str, default=None,
                             help='model directory')
    basic_group.add_argument('--model_type', type=str, default="reformer",
                             help='model type')
    basic_group.add_argument('--tokenizer_type', type=str, default="reformer",
                             help='model type')
    basic_group.add_argument('--ckpt_name', type=str, default="best.ckpt",
                             help='checkpoint to load')

    data_group = parser.add_argument_group('data')
    # Data file
    data_group.add_argument('--train_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--train_pos_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--vocab_file', type=str, default=None,
                            help='bpe vocabulary file')
    data_group.add_argument('--glossary_file', type=str, default=None,
                            help='bpe glossary file')
    data_group.add_argument('--dev_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--test_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--dev_pos_path', type=str, default=None,
                            help='data file')
    data_group.add_argument('--test_pos_path', type=str, default=None,
                            help='data file')

    config_group = parser.add_argument_group('model_configs')
    config_group.add_argument('-lr', '--learning_rate',
                              dest='lr',
                              type=float,
                              default=1e-3,
                              help='learning rate')
    config_group.add_argument('-nlayer', '--num_hidden_layer',
                              dest='nlayer',
                              type=int,
                              default=6,
                              help='number of hidden layers')
    config_group.add_argument('-elayer', '--num_encoder_layer',
                              dest='elayer',
                              type=int,
                              default=6,
                              help='number of encoder layers')
    config_group.add_argument('-dlayer', '--num_decoder_layer',
                              dest='dlayer',
                              type=int,
                              default=6,
                              help='number of decoder layers')
    config_group.add_argument('-hsize', '--hidden_size',
                              dest='hsize',
                              type=int,
                              default=512,
                              help='size of hidden layers')
    config_group.add_argument('-adim1', '--axial_pos_embds_dim1',
                              dest='adim1',
                              type=int,
                              default=128,
                              help='axial_pos_embds_dim 1')
    config_group.add_argument('-adim2', '--axial_pos_embds_dim2',
                              dest='adim2',
                              type=int,
                              default=384,
                              help='axial_pos_embds_dim 2')
    config_group.add_argument('-wstep', '--warmup_steps',
                              dest='wstep', type=int, default=0,
                              help='learning rate warmup steps')
    config_group.add_argument('-nhash', '--num_hashes',
                              dest='nhash', type=int, default=2,
                              help='number of hashes')
    config_group.add_argument('-mf', '--mask_fraction',
                              dest='mf', type=float, default=0.0,
                              help='fraction of decoder input tokens being masked out')
    config_group.add_argument('--eps',
                              type=float,
                              default=1e-5,
                              help='safty for avoiding numerical issues')
    config_group.add_argument('-gclip', '--grad_clip',
                              dest='gclip',
                              type=float, default=1.0,
                              help='gradient clipping threshold')
    config_group.add_argument('--l2', type=float, default=0.,
                              help='l2 regularization')
    config_group.add_argument('-gcs', '--gradient_accumulation_steps',
                              dest='gcs', type=int, default=1,
                              help='gradient accumulation steps')
    config_group.add_argument('-lsh_dp', '--lsh_attention_probs_dropout_prob',
                              dest='lsh_dp', type=float, default=0.05,
                              help='lsh attention dropout prob')
    config_group.add_argument('-hdp', '--hidden_dropout_prob',
                              dest='hdp', type=float, default=0.05,
                              help='hidden state dropout prob')
    config_group.add_argument('-ls', '--label_smoothing',
                              dest='ls', type=float, default=0.0,
                              help='label smoothing')
    config_group.add_argument('-fns', '--feedforward_hidden_size',
                              dest='fns', type=int, default=2,
                              help='multiplier for feedforward neural network hidden size')
    config_group.add_argument('-gelu', '--use_gelu',
                              dest='gelu', type="bool", default=False,
                              help='whether to use gelu activation')
    config_group.add_argument('-fc', '--force_copy',
                              dest='fc', type="bool", default=False,
                              help='whether to force copy')

    setup_group = parser.add_argument_group('train_setup')
    setup_group.add_argument('--n_epoch', type=int, default=5,
                             help='number of epochs')
    setup_group.add_argument('--batch_size', type=int, default=20,
                             help='batch size')
    setup_group.add_argument('--eval_batch_size', type=int, default=20,
                             help='batch size')
    setup_group.add_argument('--max_src_len', type=int, default=None,
                             help='maximum length')
    setup_group.add_argument('--max_src_entry', type=int, default=None,
                             help='maximum length')
    setup_group.add_argument('--max_tot_src_len', type=int, default=None,
                             help='maximum length')
    setup_group.add_argument('--max_tgt_len', type=int, default=None,
                             help='maximum length')
    setup_group.add_argument('--random_perm', type="bool", default=False,
                             help='whether to randomly permute anonymized entity ids')
    setup_group.add_argument('--opt', type=str, default='adam',
                             choices=['sadam', 'adam', 'sgd', 'rmsprop', 'adamw'],
                             help='types of optimizer: adam (default), \
                             sgd, rmsprop')
    setup_group.add_argument('--beam_size', type=int, default=5,
                             help='beam size')
    setup_group.add_argument('--max_gen_len', type=int, default=800,
                             help='maximum generation length')
    setup_group.add_argument('--min_gen_len', type=int, default=0,
                             help='minimum generation length')
    setup_group.add_argument('--trigram_blocking', type="bool", default=False,
                             help='trigram blocking')
    setup_group.add_argument('--gradient_checkpointing', type="bool", default=False,
                             help='gradient checkpointing')

    misc_group = parser.add_argument_group('misc')
    # misc
    misc_group.add_argument('--print_every', type=int, default=500,
                            help='print training details after \
                            this number of iterations')
    misc_group.add_argument('--eval_every', type=int, default=5000,
                            help='evaluate model after \
                            this number of iterations')
    misc_group.add_argument('--save_every', type=int, default=2000,
                            help='save model after \
                            this number of iterations')

    return parser
