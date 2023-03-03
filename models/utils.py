import numpy as np
import torch
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from training.batch_sorter import sort_batches
from utils import get_logger


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):

    pred_correct, pred_all = 0, 0
    running_loss = 0.0
    model.train(True)
    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs).expand(1, -1, -1)
        loss = criterion(outputs[0], labels[0])
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            pred_correct += 1
        pred_all += 1

    epoch_loss = running_loss / len(dataloader)
    model.train(False)
    if scheduler:
        scheduler.step(epoch_loss)

    return epoch_loss, pred_correct, pred_all, (pred_correct / pred_all)


def train_epoch_embedding(model, epoch_iters, train_loader, val_loader, criterion, optimizer, device, scheduler=None):

    running_loss = []
    model.train(True)
    for i, (anchor, positive, negative, a_mask, p_mask, n_mask) in enumerate(train_loader):
        optimizer.zero_grad()

        anchor_emb = model(anchor.to(device), a_mask.to(device))
        positive_emb = model(positive.to(device), p_mask.to(device))
        negative_emb = model(negative.to(device), n_mask.to(device))

        loss = criterion(anchor_emb.to(device), positive_emb.to(device), negative_emb.to(device))
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

        if i == epoch_iters:
            break

    epoch_loss = np.mean(running_loss)

    # VALIDATION
    model.train(False)
    val_silhouette_coef = evaluate_embedding(model, val_loader, device)

    if scheduler:
        scheduler.step(val_silhouette_coef)

    return epoch_loss, val_silhouette_coef


def train_epoch_embedding_online(model, epoch_iters, train_loader, val_loader, criterion, optimizer, device,
                                 scheduler=None, enable_batch_sorting=False, mini_batch_size=None,
                                 pre_batch_mining_count=1, batching_scheduler=None):

    running_loss = []
    iter_used_triplets = []
    iter_valid_triplets = []
    iter_pct_used = []
    model.train(True)
    mini_batch = mini_batch_size or train_loader.batch_size
    for i, (inputs, labels, masks) in enumerate(train_loader):
        labels_size = labels.size()[0]
        batch_loop_count = int(labels_size / mini_batch)
        if batch_loop_count == 0:
            continue
        # Second condition is added so that we only run batch sorting if we have a full batch
        if enable_batch_sorting:
            if labels_size < train_loader.batch_size:
                trim_count = labels_size % mini_batch
                inputs = inputs[:-trim_count]
                labels = labels[:-trim_count]
                masks = masks[:-trim_count]
            embeddings = None
            with torch.no_grad():
                for j in range(batch_loop_count):
                    batch_embed = compute_batched_embeddings(model, device, inputs, masks, mini_batch, j)
                    if embeddings is None:
                        embeddings = batch_embed
                    else:
                        embeddings = torch.cat([embeddings, batch_embed], dim=0)
                inputs, labels, masks = sort_batches(inputs, labels, masks, embeddings, device,
                                                     mini_batch_size=mini_batch_size, scheduler=batching_scheduler)
                del embeddings
                del batch_embed
            mining_loop_count = pre_batch_mining_count
        else:
            mining_loop_count = 1
        for k in range(mining_loop_count):
            for j in range(batch_loop_count):
                optimizer.zero_grad(set_to_none=True)
                batch_labels = labels[mini_batch * j:mini_batch * (j + 1)]
                if batch_labels.size()[0] == 0:
                    break
                embeddings = compute_batched_embeddings(model, device, inputs, masks, mini_batch, j)
                loss, valid_triplets, used_triplets = criterion(embeddings, batch_labels)

                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                if valid_triplets > 0:
                    iter_used_triplets.append(used_triplets)
                    iter_valid_triplets.append(valid_triplets)
                    iter_pct_used.append((used_triplets * 100) / valid_triplets)

        if epoch_iters > 0 and i * batch_loop_count * pre_batch_mining_count >= epoch_iters:
            print("Breaking out because of epoch_iters filter")
            break

    epoch_loss = np.mean(running_loss)
    mean_used_triplets = np.mean(iter_used_triplets)
    triplets_stats = {
        "valid_triplets": np.mean(iter_valid_triplets),
        "used_triplets": mean_used_triplets,
        "pct_used": np.mean(iter_pct_used)
    }

    if batching_scheduler:
        batching_scheduler.step(mean_used_triplets)

    # VALIDATION
    model.train(False)
    with torch.no_grad():
        val_silhouette_coef = evaluate_embedding(model, val_loader, device)

    if scheduler:
        scheduler.step(val_silhouette_coef)

    return epoch_loss, val_silhouette_coef, triplets_stats


def compute_batched_embeddings(model, device, inputs, masks, mini_batch, iteration):
    batch_inputs = inputs[mini_batch * iteration:mini_batch * (iteration + 1)]
    batch_masks = masks[mini_batch * iteration:mini_batch * (iteration + 1)]

    return model(batch_inputs.to(device), batch_masks.to(device)).squeeze(1)


def evaluate(model, dataloader, device, print_stats=False):

    logger = get_logger(__name__)

    pred_correct, pred_all = 0, 0
    stats = {i: [0, 0] for i in range(101)}

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        # Statistics
        if int(torch.argmax(torch.nn.functional.softmax(outputs, dim=2))) == int(labels[0][0]):
            stats[int(labels[0][0])][0] += 1
            pred_correct += 1

        stats[int(labels[0][0])][1] += 1
        pred_all += 1

    if print_stats:
        stats = {key: value[0] / value[1] for key, value in stats.items() if value[1] != 0}
        print("Label accuracies statistics:")
        print(str(stats) + "\n")
        logger.info("Label accuracies statistics:")
        logger.info(str(stats) + "\n")

    return pred_correct, pred_all, (pred_correct / pred_all)


def evaluate_embedding(model, dataloader, device):
    val_embeddings = []
    labels_emb = []

    for i, (inputs, labels, masks) in enumerate(dataloader):
        inputs = inputs.to(device)
        masks = masks.to(device)

        outputs = model(inputs, masks)
        for n in range(outputs.shape[0]):
            val_embeddings.append(outputs[n, 0].cpu().detach().numpy())
            labels_emb.append(labels.detach().numpy()[n])

    silhouette_coefficient = silhouette_score(
        X=np.array(val_embeddings),
        labels=np.array(labels_emb).reshape(len(labels_emb))
    )

    return silhouette_coefficient


def embeddings_scatter_plot(model, dataloader, device, id_to_label, perplexity=40, n_iter=1000):

    val_embeddings = []
    labels_emb = []

    with torch.no_grad():
        for i, (inputs, labels, masks) in enumerate(dataloader):
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs, masks)
            for n in range(outputs.shape[0]):
                val_embeddings.append(outputs[n, 0].cpu().detach().numpy())
                labels_emb.append(id_to_label[int(labels.detach().numpy()[n])])

    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=n_iter)
    tsne_results = tsne.fit_transform(np.array(val_embeddings))

    return tsne_results, labels_emb


def embeddings_scatter_plot_splits(model, dataloaders, device, id_to_label, perplexity=40, n_iter=1000):

    labels_split = {}
    embeddings_split = {}
    splits = list(dataloaders.keys())
    with torch.no_grad():
        for split, dataloader in dataloaders.items():
            labels_str = []
            embeddings = []
            for i, (inputs, labels, masks) in enumerate(dataloader):
                inputs = inputs.to(device)
                masks = masks.to(device)

                outputs = model(inputs, masks)
                for n in range(outputs.shape[0]):
                    embeddings.append(outputs[n, 0].cpu().detach().numpy())
                    labels_str.append(id_to_label[int(labels.detach().numpy()[n])])
            labels_split[split] = labels_str
            embeddings_split[split] = embeddings

    tsne = TSNE(n_components=2, verbose=0, perplexity=perplexity, n_iter=n_iter)
    all_embeddings = np.vstack([embeddings_split[split] for split in splits])
    tsne_results = tsne.fit_transform(all_embeddings)
    tsne_results_dict = {}
    curr_index = 0
    for split in splits:
        len_embeddings = len(embeddings_split[split])
        tsne_results_dict[split] = tsne_results[curr_index: curr_index + len_embeddings]
        curr_index += len_embeddings

    return tsne_results_dict, labels_split


def evaluate_top_k(model, dataloader, device, k=5):

    pred_correct, pred_all = 0, 0

    for i, data in enumerate(dataloader):
        inputs, labels = data
        inputs = inputs.squeeze(0).to(device)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(inputs).expand(1, -1, -1)

        if int(labels[0][0]) in torch.topk(outputs, k).indices.tolist()[0][0]:
            pred_correct += 1

        pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)
