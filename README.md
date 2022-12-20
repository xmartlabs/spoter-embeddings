# TITLE

This repository contains code for the Spoter embedding model
<!-- explained in this [blog post](link...). -->
The model is heavily based on [Spoter](https://github.com/matyasbohacek/spoter) which was presented in
[Sign Pose-Based Transformer for Word-Level Sign Language Recognition](https://openaccess.thecvf.com/content/WACV2022W/HADCV/html/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.html) with one of the main modifications being
that this is an embedding model instead of a classification model.
This allows for several zero-shot tasks on unseen Sign Language datasets from around the world.
<!-- More details about this are shown in the blog post mentioned above. -->

## Get Started

The recommended way of running code from this repo is by using Docker.

Run:
```
docker build -t spoter_embeddings .
docker run --rm -it --entrypoint=bash --gpus=all -v $PWD:/app spoter_embeddings
```

> Running without specifying the entrypoint will train the model with the hyperparameters specified in `train.sh`

If you prefer running in a **virtual environment** instead, then first install dependencies:

```shell
pip install -r requirements.txt
```

To train the model, run `train.sh` in Docker or your virtual env.

The hyperparameters with their descriptions can be found in the [train.py](link...) file.

## Data

Same as with SPOTER, this model works on top of sequences of signers' skeletal data extracted from videos.
This means that the input data has a much lower dimension compared to using videos directly, and therefore the model is
quicker and lighter, while you can choose any SOTA body pose model to preprocess video.
This makes our model lightweight and able to run in real-time (for example, it takes around 40ms to process a 4-second
25 FPS video inside a web browser using onnxruntime)

<!-- TODO: Mention dataset availability -->

![Alt Text](http://spoter.signlanguagerecognition.com/img/datasets_overview.gif)

## Modifications on [SPOTER](https://github.com/matyasbohacek/spoter)
Here is a list of the main modifications made on Spoter code and model architecture:

* The output layer is a linear layer but trained using triplet loss instead of CrossEntropyLoss. The output of the model
is therefore an embedding vector that can be used for several downstream tasks.
* We started using the keypoints dataset published by Spoter but later created new datasets using BlazePose from Mediapipe (as it is done in [Spoter 2](link...)). This improves results considerably.
* We select batches in a way that they contain several hard triplets and then compute the loss on all hard triplets found in each batch
* ...

## Tracking experiments with ClearML
The code supports tracking experiments, datasets, and models in a ClearML server.
If you want to do this make sure to pass the following arguments to train.py:

```
    --dataset_loader=clearml
    --tracker=clearml
```

Also make sure to correctly configure your clearml.conf file.
If using Docker, you can map it into Docker adding these volumes when running `docker run`:

```
-v $HOME/clearml.conf:/root/clearml.conf -v $HOME/.clearml:/root/.clearml
```


## License

The **code** is published under the [Apache License 2.0](./LICENSE) which allows for both academic and commercial use if
relevant License and copyright notice is included, our work is cited and all changes are stated.

The accompanying skeletal data of the [WLASL](https://arxiv.org/pdf/1910.11006.pdf) and [LSA64](https://core.ac.uk/download/pdf/76495887.pdf) datasets used for experiments are, however, shared under the [Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license allowing only for non-commercial usage.

<!-- ## Citation

If you find our work relevant, build upon it or compare your approaches with it, please cite our work as stated below:

```
@InProceedings{Bohacek_2022_WACV,
    author    = {Boh\'a\v{c}ek, Maty\'a\v{s} and Hr\'uz, Marek},
    title     = {Sign Pose-Based Transformer for Word-Level Sign Language Recognition},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
    month     = {January},
    year      = {2022},
    pages     = {182-191}
}
``` -->
