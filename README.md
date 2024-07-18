# pay-better-attention-pytorch-keras

A PyTorch and Keras implementation of the Optimised, Efficient and Super Attention Variants from "You Need to Pay Better Attention: Rethinking the Mathematics of Attention Mechanism"

## pay-better-attention-pytorch

An implementation of Optimised, Efficient and Super Attention Variants from "You Need to Pay Better Attention: Rethinking the Mathematics of Attention Mechanism". This paper provides helpful optimizations to the Scaled Dot Product Attention (SDPA) mechanism by observing that in SDPA, there are consecutive matrix multiplications that can be compressed into a single learnt layer. Another observation the authors made is that learning a transformation between the K.T and Q values in Attention may provide performance boosts, which it empirically does.

This code has been verified against the Author's results, and I have published reimplementation results and a discussion of the paper at https://nmesac1019.medium.com/you-need-to-pay-better-attention-paper-discussion-49fb694f6881 as well as a paper summary at NMesaC.github.io

## Usage

In each implementation folder, there is a Jupyter Notebook with more details on how to test each Attention Layers performance on the IMDB Sentiment Analysis Dataset with a Transformer-Encoder-Only architecture.

If you are running the code locally, you can simply run either the keras_imdb_transformer_experiment.py or pytorch_imdb_transformer_experiment.py files and if you have the correct dependencies, then it will automatically begin the experiments.

## Acknowledgements

 I would like to thank Seyedpeyman Hosseini and Mehran Hosseini, the original paper authors, for being so kind as to assist me in figuring out how to actually replicate the results of their test bench and in general some questions about architecture differences. They were generous enough to help point out the differences in my test bench, data collection approach and hyperparameters, and I greatly appreciate their time and assistance. The validity of this codebase would not be possible with out their help, and I greatly appreciate it.

## Citation

```bibtex
@inproceedings{hosseini2024needpaybetterattention,
      title={You Need to Pay Better Attention: Rethinking the Mathematics of Attention Mechanism}, 
      author={Mehran Hosseini and Peyman Hosseini},
      year={2024},
      eprint={2403.01643},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.01643}, 
}
```

