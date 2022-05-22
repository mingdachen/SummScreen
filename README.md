# SummScreen

- Tokenized dataset (including the anonymized SummScreen dataset): https://drive.google.com/file/d/1BvdIllGBo9d2-bzXQRzWuJXB04XPVmfF/view?usp=sharing
- Untokenized dataset (contains instances that were filtered out in the SummScreen dataset and you need to recover the train/dev/test splits by matching the file names to the ones in the tokenized dataset): https://drive.google.com/file/d/1tFpt32USOO2i1FWhtFTsyYyFzuRm2k36/view?usp=sharing

# Acknowledgement

SummScreen was sourced from [TVMegaSite](http://tvmegasite.net/), [Wikipedia](https://www.wikipedia.org/), [TVmaze](https://www.tvmaze.com/), and [ForeverDreaming](https://foreverdreaming.org/).

# Code for training the neural model baseline (longformer encoder + standard transformer decoder)
### Environment
- Python 3.7
- PyTorch 1.11.0
- HuggingFace Transformers 4.16.2

### Train
```sh scripts/train-seq2seq-lf.sh```

### Generate
```sh scripts/generate-seq2seq-lf.sh MODEL_DIR BEAM_SIZE```

### Evaluate
[file2rouge](https://github.com/pltrdy/files2rouge) and [multi-bleu](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl)

# Reference

```
@inproceedings{chen-etal-2022-summscreen,
    title = "{S}umm{S}creen: A Dataset for Abstractive Screenplay Summarization",
    author = "Chen, Mingda  and
      Chu, Zewei  and
      Wiseman, Sam  and
      Gimpel, Kevin",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.589",
    pages = "8602--8615",
    abstract = "We introduce SummScreen, a summarization dataset comprised of pairs of TV series transcripts and human written recaps. The dataset provides a challenging testbed for abstractive summarization for several reasons. Plot details are often expressed indirectly in character dialogues and may be scattered across the entirety of the transcript. These details must be found and integrated to form the succinct plot descriptions in the recaps. Also, TV scripts contain content that does not directly pertain to the central plot but rather serves to develop characters or provide comic relief. This information is rarely contained in recaps. Since characters are fundamental to TV series, we also propose two entity-centric evaluation metrics. Empirically, we characterize the dataset by evaluating several methods, including neural models and those based on nearest neighbors. An oracle extractive approach outperforms all benchmarked models according to automatic metrics, showing that the neural models are unable to fully exploit the input transcripts. Human evaluation and qualitative analysis reveal that our non-oracle models are competitive with their oracle counterparts in terms of generating faithful plot events and can benefit from better content selectors. Both oracle and non-oracle models generate unfaithful facts, suggesting future research directions.",
}
```
