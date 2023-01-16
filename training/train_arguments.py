import argparse


def get_default_args():
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument("--experiment_name", type=str, default="lsa_64_spoter",
                        help="Name of the experiment after which the logs and plots will be named")
    parser.add_argument("--num_classes", type=int, default=100, help="Number of classes to be recognized by the model")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Hidden dimension of the underlying Transformer model")
    parser.add_argument("--seed", type=int, default=379,
                        help="Seed with which to initialize all the random components of the training")

    # Embeddings
    parser.add_argument("--classification_model", action='store_true', default=False,
                        help="Select SPOTER model to train, pass only for original classification model")
    parser.add_argument("--vector_length", type=int, default=32,
                        help="Number of features used in the embedding vector")
    parser.add_argument("--epoch_iters", type=int, default=-1,
                        help="Iterations per epoch while training embeddings. Will loop through dataset once if -1")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size during training and validation")
    parser.add_argument("--hard_triplet_mining", type=str, default=None,
                        help="Strategy to select hard triplets, options [None, in_batch]")
    parser.add_argument("--triplet_loss_margin", type=float, default=1,
                        help="Margin used in triplet loss margin (See documentation)")
    parser.add_argument("--normalize_embeddings", action='store_true', default=False,
                        help="Normalize model output to keep vector length to one")
    parser.add_argument("--filter_easy_triplets", action='store_true', default=False,
                        help="Filter easy triplets in online in batch triplets")

    # Data
    parser.add_argument("--dataset_name", type=str, default="", help="Dataset name")
    parser.add_argument("--dataset_project", type=str, default="Sign Language Recognition", help="Dataset project name")
    parser.add_argument("--training_set_path", type=str, default="",
                        help="Path to the training dataset CSV file (relative to root dataset)")
    parser.add_argument("--validation_set_path", type=str, default="", help="Path to the validation dataset CSV file")
    parser.add_argument("--dataset_loader", type=str, default="local",
                        help="Dataset loader to use, options: [clearml, local]")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=1300, help="Number of epochs to train the model for")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the model training")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout used in transformer layer")
    parser.add_argument("--augmentations_prob", type=float, default=0.5, help="How often to use data augmentation")

    # Checkpointing
    parser.add_argument("--save_checkpoints_every", type=int, default=-1,
                        help="Determines every how many epochs the weight checkpoints are saved. If -1 only best model \
                            after final epoch")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="SGD",
                        help="Optimizer used during training, options: [SGD, ADAM]")

    # Tracker
    parser.add_argument("--tracker", type=str, default="none",
                        help="Experiment tracker to use, options: [clearml, none]")

    # Scheduler
    parser.add_argument("--scheduler_factor", type=float, default=0,
                        help="Factor for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_patience", type=int, default=10,
                        help="Patience for the ReduceLROnPlateau scheduler")
    parser.add_argument("--scheduler_warmup", type=int, default=400,
                        help="Warmup epochs before scheduler starts")

    # Gaussian noise normalization
    parser.add_argument("--gaussian_mean", type=float, default=0, help="Mean parameter for Gaussian noise layer")
    parser.add_argument("--gaussian_std", type=float, default=0.001,
                        help="Standard deviation parameter for Gaussian noise layer")

    # Batch Sorting
    parser.add_argument("--start_mining_hard", type=int, default=None, help="On which epoch to start hard mining")
    parser.add_argument("--hard_mining_pre_batch_multipler", type=int, default=16,
                        help="How many batches should be computed at once")
    parser.add_argument("--hard_mining_pre_batch_mining_count", type=int, default=5,
                        help="How many times to loop through a list of computed batches")
    parser.add_argument("--hard_mining_scheduler_triplets_threshold", type=float, default=0,
                        help="Enables batching grouping scheduler if > 0. Defines threshold for when to decay the \
                            distance threshold of the batch sorter")

    return parser
