import torch
from tqdm.autonotebook import tqdm
from sklearn.metrics import f1_score
import copy
import wandb


def evaluate(model, eval_loader, loss_fn, device):
    model = model.eval()
    r_loss = 0.0
    preds = []
    ans = []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            x_batch = batch['image'].to(device)
            y_batch = batch['label'].to(device)

            y_pred = model(x_batch).squeeze()
            predicted = (y_pred > 0.5).float()

            loss = loss_fn(y_pred.float().to(device), y_batch.float())
            r_loss += loss.item()

            preds.extend(predicted.cpu().numpy())
            ans.extend(y_batch.cpu().numpy())

        f1_metric = f1_score(ans, preds, average='binary')

    return f1_metric, r_loss / len(eval_loader)


def train_model(model, optimizer, loss_fn, train_loader, val_loader, device, conf):
    train_loss_history = []
    val_loss_history = []
    train_f1_history = []
    val_f1_history = []
    cur_f1 = 0
    num_epochs = conf.epochs
    wandb.watch(model, loss_fn, log="all", log_freq=10)
    seen_img = 0
    best_model_wts = None

    for ep in range(num_epochs):
        model = model.train()
        running_loss = 0.0
        preds_train = []
        ans_train = []

        for i, batch in enumerate(train_loader):

            x_batch = batch['image'].to(device)
            y_batch = batch['label'].to(device)

            optimizer.zero_grad()

            y_pred = model(x_batch).squeeze()
            predicted = (y_pred > 0.5).float()
            loss = loss_fn(y_pred.float().to(device), y_batch.float())

            if i % 5 == 0:
                print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    ep + 1,
                    num_epochs,
                    i * len(y_batch),
                    len(train_loader.dataset),
                    100. * i / len(train_loader),
                    loss),
                    end='')

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            seen_img += 128
            if i % 5 == 0:
                figure_log('train_loss', loss, seen_img, ep)

            preds_train.extend(predicted.cpu().numpy())
            ans_train.extend(y_batch.cpu().numpy())

        train_f1 = f1_score(ans_train, preds_train, average='binary')
        train_loss = running_loss / len(train_loader)
        print(f'\navg train loss:{train_loss:.3f}, f1 on train:{train_f1:.3f}')
        train_loss_history.append(train_loss)
        train_f1_history.append(train_f1)

        figure_log('avg_train_loss', train_loss, seen_img, ep)
        figure_log('train_f1', train_f1, seen_img, ep)

        val_f1, val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f'avg val loss:{val_loss:.3f}, f1 on validation:{val_f1:.3f}\n')
        val_loss_history.append(val_loss)
        val_f1_history.append(val_f1)

        figure_log('avg_val_loss', val_loss, seen_img, ep)
        figure_log('val_f1', val_f1, seen_img, ep)

        if val_f1 > cur_f1:
            cur_f1 = val_f1
            best_model_wts = copy.deepcopy(model)

    return best_model_wts, train_loss_history, train_f1_history, val_loss_history, val_f1_history


def figure_log(mode, value, example_ct, epoch):
    if mode == 'train_loss':
        wandb.log({"epoch": epoch, "train_loss_every_5_batches": value}, step=example_ct)
    elif mode == 'avg_train_loss':
        wandb.log({"epoch": epoch, "avg_train_loss": value}, step=example_ct)
    elif mode == 'train_f1':
        wandb.log({"epoch": epoch, "train_f1": value}, step=example_ct)
    elif mode == 'avg_val_loss':
        wandb.log({"epoch": epoch, "avg_val_loss": value}, step=example_ct)
    elif mode == 'val_f1':
        wandb.log({"epoch": epoch, "val_f1": value}, step=example_ct)
    else:
        print('unknown mode')
