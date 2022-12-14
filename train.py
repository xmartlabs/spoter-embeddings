
from datetime import datetime
import os
import os.path as op
import argparse
import random
import logging
import json
from datasets.dataset_loader import LocalDatasetLoader
from tracking.tracker import Tracker
import torch
import multiprocessing
import numpy as np
import torch.nn as nn
import torch.optim as optim
# import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import copy

from datasets import CzechSLRDataset, SLREmbeddingDataset, collate_fn_triplet_padd, collate_fn_padd
from models import SPOTER, SPOTER_EMBEDDINGS, train_epoch, evaluate, train_epoch_embedding, \
    train_epoch_embedding_online, evaluate_embedding
from training.online_batch_mining import BatchAllTripletLoss
from training.batching_scheduler import BatchingScheduler
from training.gaussian_noise import GaussianNoise
from training.train_utils import train_setup, create_embedding_scatter_plots
from training.train_arguments import get_default_args

PROJECT_NAME = "spoter"
CLEAR_ML = "clear_ml"


def is_pre_batch_sorting_enabled(args):
    return args.start_mining_hard is not None and args.start_mining_hard > 0


def get_tracker(tracker_name, project, experiment_name):
    if tracker_name == CLEAR_ML:
        from tracking.clearml_tracker import ClearMLTracker
        return ClearMLTracker(project_name=project, experiment_name=experiment_name)
    else:
        return Tracker(project_name=project, experiment_name=experiment_name)


def get_dataset_loader(loader_name):
    if loader_name == CLEAR_ML:
        from datasets.clearml_dataset_loader import ClearMLDatasetLoader
        return ClearMLDatasetLoader()
    else:
        return LocalDatasetLoader()


def build_data_loader(dataset, batch_size, shuffle, collate_fn, generator):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn,
                      generator=generator, pin_memory=torch.cuda.is_available(), num_workers=multiprocessing.cpu_count())


def train(args, tracker: Tracker):
    tracker.execute_remotely(queue_name="default")
    # Initialize all the random seeds
    gen = train_setup(args.seed, args.experiment_name)

    # Set device to CUDA only if applicable
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Construct the model
    if not args.classification_model:
        slrt_model = SPOTER_EMBEDDINGS(
            features=args.vector_length,
            hidden_dim=args.hidden_dim,
            norm_emb=args.normalize_embeddings,
            dropout=args.dropout
        )
        model_type = 'embed'
        if args.hard_triplet_mining == "None":
            cel_criterion = nn.TripletMarginLoss(margin=args.triplet_loss_margin, p=2)
        elif args.hard_triplet_mining == "in_batch":
            cel_criterion = BatchAllTripletLoss(
                device=device,
                margin=args.triplet_loss_margin,
                filter_easy_triplets=bool(args.filter_easy_triplets)
            )
    else:
        slrt_model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim)
        model_type = 'classif'
        cel_criterion = nn.CrossEntropyLoss()
    slrt_model.to(device)

    if args.optimizer == "SGD":
        optimizer = optim.SGD(slrt_model.parameters(), lr=args.lr)
    elif args.optimizer == "ADAM":
        optimizer = optim.Adam(slrt_model.parameters(), lr=args.lr)

    if args.scheduler_factor > 0:
        mode = 'min' if args.classification_model else 'max'
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=args.scheduler_factor,
            patience=args.scheduler_patience
        )
    else:
        scheduler = None

    if args.hard_mining_scheduler_triplets_threshold > 0:
        batching_scheduler = BatchingScheduler(triplets_threshold=args.hard_mining_scheduler_triplets_threshold)
    else:
        batching_scheduler = None

    # Ensure that the path for checkpointing and for images both exist
    Path("out-checkpoints/" + args.experiment_name + "/").mkdir(parents=True, exist_ok=True)
    Path("out-img/").mkdir(parents=True, exist_ok=True)

    # Training set
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    dataset_loader = get_dataset_loader(args.dataset_loader)
    dataset_folder = dataset_loader.get_dataset_folder(args.dataset_project, args.dataset_name)
    training_set_path = op.join(dataset_folder, args.training_set_path)

    with open(op.join(dataset_folder, 'id_to_label.json')) as fid:
        id_to_label = json.load(fid)
    id_to_label = {int(key): value for key, value in id_to_label.items()}

    if not args.classification_model:
        batch_size = args.batch_size
        val_batch_size = args.batch_size
        if args.hard_triplet_mining == "None":
            train_set = SLREmbeddingDataset(training_set_path, triplet=True, transform=transform, augmentations=True,
                                            augmentations_prob=args.augmentations_prob)
            collate_fn_train = collate_fn_triplet_padd
        elif args.hard_triplet_mining == "in_batch":
            train_set = SLREmbeddingDataset(training_set_path, triplet=False, transform=transform, augmentations=True,
                                            augmentations_prob=args.augmentations_prob)
            collate_fn_train = collate_fn_padd
            if is_pre_batch_sorting_enabled(args):
                batch_size *= args.hard_mining_pre_batch_multipler
        train_val_set = SLREmbeddingDataset(training_set_path, triplet=False)
        # Train dataloader for validation
        train_val_loader = build_data_loader(train_val_set, val_batch_size, False, collate_fn_padd, gen)
    else:
        train_set = CzechSLRDataset(training_set_path, transform=transform, augmentations=True)
        batch_size = 1
        val_batch_size = 1
        collate_fn_train = None

    train_loader = build_data_loader(train_set, batch_size, True, collate_fn_train, gen)

    # Validation set
    validation_set_path = op.join(dataset_folder, args.validation_set_path)

    if args.classification_model:
        val_set = CzechSLRDataset(validation_set_path)
        collate_fn_val = None
    else:
        val_set = SLREmbeddingDataset(validation_set_path, triplet=False)
        collate_fn_val = collate_fn_padd

    val_loader = build_data_loader(val_set, val_batch_size, False, collate_fn_val, gen)

    # MARK: TRAINING
    train_acc, val_acc = 0, 0
    losses, train_accs, val_accs = [], [], []
    lr_progress = []
    top_val_acc = -999
    top_model_saved = True

    print("Starting " + args.experiment_name + "...\n\n")
    logging.info("Starting " + args.experiment_name + "...\n\n")

    if is_pre_batch_sorting_enabled(args):
        mini_batch_size = int(batch_size / args.hard_mining_pre_batch_multipler)
    else:
        mini_batch_size = None
    enable_batch_sorting = False
    pre_batch_mining_count = 1
    for epoch in range(1, args.epochs + 1):
        start_time = datetime.now()
        if not args.classification_model:
            train_kwargs = {"model": slrt_model,
                            "epoch_iters": args.epoch_iters,
                            "train_loader": train_loader,
                            "val_loader": val_loader,
                            "criterion": cel_criterion,
                            "optimizer": optimizer,
                            "device": device,
                            "scheduler": scheduler if epoch >= args.scheduler_warmup else None,
                            }
            if args.hard_triplet_mining == "None":
                train_loss, val_silhouette_coef = train_epoch_embedding(**train_kwargs)
            elif args.hard_triplet_mining == "in_batch":
                if epoch == args.start_mining_hard:
                    enable_batch_sorting = True
                    pre_batch_mining_count = args.hard_mining_pre_batch_mining_count
                train_kwargs.update(dict(enable_batch_sorting=enable_batch_sorting,
                                         mini_batch_size=mini_batch_size,
                                         pre_batch_mining_count=pre_batch_mining_count,
                                         batching_scheduler=batching_scheduler if enable_batch_sorting else None))

                train_loss, val_silhouette_coef, triplets_stats = train_epoch_embedding_online(**train_kwargs)

                tracker.log_scalar_metric("triplets", "valid_triplets", epoch, triplets_stats["valid_triplets"])
                tracker.log_scalar_metric("triplets", "used_triplets", epoch, triplets_stats["used_triplets"])
                tracker.log_scalar_metric("triplets_pct", "pct_used", epoch, triplets_stats["pct_used"])
            tracker.log_scalar_metric("train_loss", "loss", epoch, train_loss)
            losses.append(train_loss)

            # calculate acc on train dataset
            silhouette_coefficient_train = evaluate_embedding(slrt_model, train_val_loader, device)

            tracker.log_scalar_metric("silhouette_coefficient", "train", epoch, silhouette_coefficient_train)
            train_accs.append(silhouette_coefficient_train)

            val_accs.append(val_silhouette_coef)
            tracker.log_scalar_metric("silhouette_coefficient", "val", epoch, val_silhouette_coef)

        else:
            train_loss, _, _, train_acc = train_epoch(slrt_model, train_loader, cel_criterion, optimizer, device)
            tracker.log_scalar_metric("train_loss", "loss", epoch, train_loss)
            tracker.log_scalar_metric("acc", "train", epoch, train_acc)
            losses.append(train_loss)
            train_accs.append(train_acc)

            _, _, val_acc = evaluate(slrt_model, val_loader, device)
            val_accs.append(val_acc)
            tracker.log_scalar_metric("acc", "val", epoch, val_acc)

        logging.info(f"Epoch time: {datetime.now() - start_time}")
        logging.info("[" + str(epoch) + "] TRAIN  loss: " + str(train_loss) + " acc: " + str(train_accs[-1]))
        logging.info("[" + str(epoch) + "] VALIDATION  acc: " + str(val_accs[-1]))

        logging.info("")

        lr_progress.append(optimizer.param_groups[0]["lr"])
        tracker.log_scalar_metric("lr", "lr", epoch, lr_progress[-1])

        if val_accs[-1] > top_val_acc:
            top_val_acc = val_accs[-1]
            top_model_name = "checkpoint_" + model_type + "_" + str(epoch) + ".pth"
            top_model_dict = {
                "name": top_model_name,
                "epoch": epoch,
                "val_acc": val_accs[-1],
                "config_args": args,
                "state_dict": copy.deepcopy(slrt_model.state_dict()),
            }
            top_model_saved = False

        # Save checkpoint if it is the best on validation and delete previous checkpoints
        if args.save_checkpoints_every > 0 and epoch % args.save_checkpoints_every == 0 and not top_model_saved:
            torch.save(
                top_model_dict,
                "out-checkpoints/" + args.experiment_name + "/" + top_model_name
            )
            top_model_saved = True
            logging.info("Saved new best checkpoint: " + top_model_name)

    # save top model if checkpoints are disabled
    if not top_model_saved:
        torch.save(
            top_model_dict,
            "out-checkpoints/" + args.experiment_name + "/" + top_model_name
        )
        logging.info("Saved new best checkpoint: " + top_model_name)

    # Log scatter plots
    if not args.classification_model and args.hard_triplet_mining == "in_batch":
        logging.info("Generating Scatter Plot.")
        best_model = slrt_model
        best_model.load_state_dict(top_model_dict["state_dict"])
        create_embedding_scatter_plots(tracker, best_model, train_loader, val_loader, device, id_to_label, epoch,
                                       top_model_name)
    logging.info("The experiment is finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("", parents=[get_default_args()], add_help=False)
    args = parser.parse_args()
    tracker = get_tracker(args.tracker, PROJECT_NAME, args.experiment_name)
    train(args, tracker)
    tracker.finish_run()
