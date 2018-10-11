# Propensity-weighted-click-prediction
Intern Project done in Cornell, supervised by Professor Thorsten Joachims.

Instructions to run the experiments on the covertype dataset.

- First download the dataset in the `./data/` folder. Use `wget` to download the data from `data_url`.
- Run `prepare.py`, `policy0.py`, `view.py` to process and reconstruct the dataset. You can use `python prepare.py && python policy0.py && python view.py`. Here the `policy0.py` is a 5-pass sampling. To modify it you have to modify the code. I haven't implemented options.
- Run `stat.py` to see the statistics of the reconstructed dataset.
- Run `main.py` to do training and evaluation.
  - `-v` verbose the training process.
  - `-alg` to select a algorithm from `mlp`(strong baseline), `pmlpb1`(advanced v1 and v2, controlled by another arg `-d`), `pmlpb2`(advanced v3), `pmlpb2h`(advanced v3 hardmax)
  - `-name` the name of the saved model.
  - `-clip` the clipping parameter on logged propensity `$p$`
  - `-wclip` the clipping parameter on `$\frac{\hat{p}}{p}$`
  - `-batch` the batch size
  - `-val` whether use validation set to do evaluation(Have bugs, not fixed yet, do not use)
  - `-wd` the weight decay parameter
  - `-lr` the learning rate
  - `-np` the number of epochs
  - `-seed` the random seed of tensorflow
  - `-v` whether to verbose the training process
  - `-save` if use -save x , `x` is an integer, it means save model after every `x` epochs
  - `-pred` whether to do inference on the test set
  - `-com` whether to evaluate the result.
  - `-vd` variance decay parameter. The result of current impelemntation is bad.
  - `-d` whether to detach the `\hat{p}`(Only for `-alg pmlpb1`). If use `-d` then, `pmlpb1` will be advanced v2, otherwise v1.
