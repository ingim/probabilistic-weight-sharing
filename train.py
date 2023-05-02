import torch
import torchvision
import torch.utils.data
import model

import torchmetrics

from tqdm import tqdm
import wandb
import argparse
import torch.jit as jit
import multi_mnist

from torch.profiler import profile, record_function, ProfilerActivity


def main():
    parser = argparse.ArgumentParser(description='Relational Parameter Sharing')
    parser.add_argument('-l', '--log', help='Description for foo argument', action='store_true')
    args = parser.parse_args()

    batch_size = 256

    l1_dim = 128
    l2_dim = 32

    batch_size_test = batch_size
    mode = 'pps'
    lr = 5e-4
    num_f = 5
    num_epoch = 1000
    temperature = 0.05
    use_bias = False
    save_model = True
    run = None
    ortho_loss_weight = 0.3

    var_size = [2048, 64, 1]

    input_size = 32 * 96  # 3072

    if args.log:
        run = wandb.init(project="probabilistic-parameter-sharing",
                         entity="ingim",
                         save_code=True,
                         config={
                             'mode': mode,
                             "learning_rate": lr,
                             "num_epoch": num_epoch,
                             "batch_size": batch_size,
                             "num_f": num_f,
                             "l1_dim": l1_dim,
                             "l2_dim": l2_dim,
                             "var_size": var_size,
                             "ortho_loss_weight": ortho_loss_weight,
                             "use_bias": use_bias,
                             "save_model": save_model,
                             "input_size": input_size,
                             "temperature": temperature
                         })

    baseline = model.Simple(input_size, 1, 10, mode=mode, num_f=num_f, l1_dim=l1_dim, l2_dim=l2_dim, var_size=var_size,
                            bias=use_bias,
                            temperature=temperature).cuda()

    # baseline = model.VerySimple(7056, 10).cuda()
    # baseline = torch.jit.trace(baseline, torch.empty(batch_size, input_size).cuda())

    dataset_train = multi_mnist.MultiMnistDataset('./dataset/multimnist/train', num_digits=3)
    dataset_val = multi_mnist.MultiMnistDataset('./dataset/multimnist/val', num_digits=3)

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                               num_workers=16, pin_memory=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size_test, shuffle=False,
                                             num_workers=16, pin_memory=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=baseline.parameters(), lr=lr)

    # define some evaluation metrics
    precision = torchmetrics.classification.BinaryPrecision().cuda()
    recall = torchmetrics.classification.BinaryRecall().cuda()
    f1 = torchmetrics.classification.BinaryRecall().cuda()
    accuracy = torchmetrics.classification.BinaryAccuracy().cuda()

    for epoch in range(num_epoch):

        print(f'Epoch {epoch}')

        avg_loss = 0
        avg_acc = 0
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        total_samples = 0

        for i, batch in enumerate(tqdm(loader_train)):
            optimizer.zero_grad()
            total_samples += 1

            img, target = batch

            img = img.cuda()
            target = target.cuda()

            img = torch.flatten(img, start_dim=1)

            # (n, in) -> (n, num_task, out)
            logits, ortho_loss = baseline(img)

            # (n, num_task, out) = (n, num_task, 1) -> (n, num_task)
            # logits.squeeze(-1)

            loss = criterion(logits.squeeze(-1), target) + ortho_loss_weight * ortho_loss
            loss.backward()

            # calculate metrics
            preds = torch.round(torch.sigmoid(logits.squeeze(-1)))

            #
            label = target.long()

            train_accuracy = accuracy(preds, label).item()
            train_precision = precision(preds, label).item()
            train_recall = recall(preds, label).item()
            train_f1 = f1(preds, label).item()

            optimizer.step()

            avg_loss += loss.item()
            avg_acc += train_accuracy
            avg_precision += train_precision
            avg_recall += train_recall
            avg_f1 += train_f1

            if args.log:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_accuracy": train_accuracy,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_f1_score": train_f1,
                })

        avg_loss /= total_samples
        avg_acc /= total_samples
        avg_precision /= total_samples
        avg_recall /= total_samples
        avg_f1 /= total_samples

        print(
            f'[Train] loss: {avg_loss}, accuracy: {avg_acc}, precision: {avg_precision}, recall: {avg_recall}, f1: {avg_f1}')

        with torch.no_grad():

            avg_loss = 0
            avg_acc = 0
            avg_precision = 0
            avg_recall = 0
            avg_f1 = 0
            total_samples = 0

            for i, batch in enumerate(tqdm(loader_val)):
                img, target = batch

                total_samples += 1

                img = img.cuda()
                target = target.cuda()

                img = torch.flatten(img, start_dim=1)

                # (n, in) -> (n, num_task, out)
                logits, ortho_loss = baseline(img)

                # (n, num_task, out) = (n, num_task, 1) -> (n, num_task)
                # logits.squeeze(-1)

                loss = criterion(logits.squeeze(-1), target) + ortho_loss_weight * ortho_loss

                # calculate metrics
                preds = torch.round(torch.sigmoid(logits.squeeze(-1)))

                label = target.long()

                valid_accuracy = accuracy(preds, label).item()
                valid_precision = precision(preds, label).item()
                valid_recall = recall(preds, label).item()
                valid_f1 = f1(preds, label).item()

                avg_loss += loss.item()
                avg_acc += valid_accuracy
                avg_precision += valid_precision
                avg_recall += valid_recall
                avg_f1 += valid_f1

            avg_loss /= total_samples
            avg_acc /= total_samples
            avg_precision /= total_samples
            avg_recall /= total_samples
            avg_f1 /= total_samples

            print(
                f'[Validation] loss: {avg_loss}, accuracy: {avg_acc}, precision: {avg_precision}, recall: {avg_recall}, f1: {avg_f1}')
            if args.log:
                wandb.log({
                    "val_loss": avg_loss,
                    "val_accuracy": avg_acc,
                    "val_precision": avg_precision,
                    "val_recall": avg_recall,
                    "val_f1_score": avg_f1
                })

        if save_model and args.log:
            torch.save(baseline.state_dict(), f'./checkpoint/{run.name}_{epoch}.pt')
        # print(img.shape, target.shape)


if __name__ == "__main__":
    main()
