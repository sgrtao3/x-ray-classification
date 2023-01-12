from sklearn.metrics import accuracy_score
import torch
import numpy as np
import torch.nn.functional as F
from livelossplot import PlotLosses
from pycm import ConfusionMatrix


def validate(model, criterion, data_loader, imshape=(-1, 1, 299, 299),
             device='cpu'):
    """
    Validate the accuracy of a model on a previously unseen dataset

    Args:
        model (nn.Module): Model
        criterion (torch.nn.optim): Criterion for loss function calculation
        data_loader (DataLoader): DataLoader for input validation data
        imshape (tuple, optional): Shape of input image.
                                   Defaults to (-1, 1, 299, 299)
        device (str, optional): Device to run on. Defaults to 'cpu'.

    Returns:
        float: validation loss (normalized to dataset length)
        float: validation accuracy (normalized to dataset length)
    """
    model.eval()
    validation_loss, validation_accuracy = 0., 0.
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            out = model(X.view(*imshape))
            loss = criterion(out, y)
            validation_loss += loss*X.size(0)
            y_pred = F.log_softmax(out, dim=1).max(1)[1]
            validation_accuracy += accuracy_score(y.cpu().numpy(), y_pred.cpu()
                                                  .numpy())*X.size(0)

    return validation_loss/len(data_loader.dataset), validation_accuracy / \
        len(data_loader.dataset)


def train(model, optimizer, criterion, data_loader, imshape=(-1, 1, 299, 299),
          device='cpu'):
    """
    Train a model given a labeled dataset

    Args:
        model (nn.Module): Model
        criterion (torch.nn.optim): Criterion for loss function calculation
        data_loader (DataLoader): DataLoader for input validation data
        imshape (tuple, optional): Shape of input image.
                                   Defaults to (-1, 1, 299, 299)
        device (str, optional): Device to run on. Defaults to 'cpu'.

    Returns:
        float: validation loss (normalized to dataset length)
        float: validation accuracy (normalized to dataset length)
    """
    model.train()
    train_loss, train_accuracy = 0, 0
    i = 0
    for X, y in data_loader:
        if i % 100 == 0:
            print(i)
        i += 1
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X.view(*imshape))
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss*X.size(0)
        y_pred = F.log_softmax(out, dim=1).max(1)[1]
        train_accuracy += accuracy_score(y.cpu().numpy(), y_pred.detach()
                                         .cpu().numpy())*X.size(0)

    return train_loss/len(data_loader.dataset), train_accuracy/len(data_loader.
                                                                   dataset)


def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any
    randomness from cuda kernels
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def predict(model, test_loader, max=True, imshape=(-1, 1, 299, 299),
            device='cpu'):
    """
    Generate predictions for a trained model

    Args:
        model (nn.Module): (Trained) model
        test_loader (DataLoader): DataLoader to load in test data
        max (bool, optional): Get only max prediction. Defaults to True.
        imshape (tuple, optional): Image shape. Defaults to (-1, 1, 299, 299).
        device (str, optional): Device to run on. Defaults to 'cpu'.

    Returns:
        list: List of prediction
    """
    model.eval()

    preds = []

    for X in test_loader:
        with torch.no_grad():
            X = X.to(device)
            out = model(X.view(*imshape))
            if max:
                y_pred = F.log_softmax(out, dim=1).max(1)[1]
                preds.append(y_pred.item())
            else:
                y_pred = F.softmax(out, dim=1)
                preds.append([pred.cpu().numpy() for pred in y_pred][0])

    preds = np.array(preds)
    return preds


def preds_to_file(preds, filename):
    """
    Function to write a list of predictions to file in Kaggle submission format

    Args:
        preds (list): predictions
        filename (string): name of submission file
    """
    with open(filename, "w") as file:
        for i, y in enumerate(preds):
            file.write(f"test_{i},{y}\n")


def preds_average(pred_lists, weights_list):
    """
    Get average prediction over a set of lists of predictions with weigths.

    Args:
        pred_lists (list): List of predictions
        weights_list (list): List of weights

    Returns:
        list: Resulting predictions
    """
    pred_lists = np.prod([pred_lists, weights_list], axis=0)
    return np.argmax(sum(pred_lists), axis=1)


def train_validate(train_loader, validation_loader, lr, momentum,  model,
                   optimizer, criterion, epochs=30,  device='cpu', seed=42):
    """
    Subroutine to do both the training and the validation for a certain
    number of epochs

    Args:
        train_loader (DataLoader): DataLoader of the train dataset
        validation_loader (DataLoader): Dataloader of the validation dataset
        lr (float): Learning rate
        momentum (float): Momentum
        model (nn.Module): Model to run training and validation with
        optimizer (torch.optim): Optimization algorithm
        criterion (torch.nn.modules.loss): Loss criterion
        epochs (int, optional): [description]. Defaults to 30.
        device (str, optional): [description]. Defaults to 'cpu'.
        seed (int, optional): [description]. Defaults to 42.

    Returns:
        nn.Module: Trained model
        dict: Logs of run
    """
    # Set up the model and send it to the GPU (or CPU if one isn't available)
    set_seed(seed)
    model = model.to(device)

    liveloss = PlotLosses()
    for epoch in range(epochs):
        logs = {}
        train_loss, train_accuracy = train(model, optimizer, criterion,
                                           train_loader, device=device)

        # Update the logs for the training data
        logs['' + 'log loss'] = train_loss.item()
        logs['' + 'accuracy'] = train_accuracy.item()

        # Update the logs for the validation data
        validation_loss, validation_accuracy = validate(model, criterion,
                                                        validation_loader,
                                                        device=device)
        logs['val_' + 'log loss'] = validation_loss.item()
        logs['val_' + 'accuracy'] = validation_accuracy.item()
        liveloss.update(logs)

        liveloss.draw()

    return model, logs


def evaluate(model, data_loader, device='cpu'):
    """
    Evaluate model given labeled input data

    Args:
        model (nn.Module): Model to evaluate
        data_loader (DataLoader): DataLoader for input data
        device (str, optional): Device to use. Defaults to 'cpu'.

    Returns:
        np.array: array with predictions
        np.array: array with actual values
    """
    model.eval()
    ys, y_preds = [], []
    for X, y in data_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            a2 = model(X)
            y_pred = F.log_softmax(a2, dim=1).max(1)[1]
            ys.append(y.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())

    return np.concatenate(y_preds, 0),  np.concatenate(ys, 0)


def auc(model, validation_loader):
    """
    Calculate area under Receiver Operating Curve metric for a model

    Args:
        model (nn.Module): Model that ROC-AUC is being calculated for
        validation_loader (DataLoader): DataLoader to load in labeled
                                        validation set

    Returns:
        float: ROC-AUC value
    """
    y_pred, y_gt = evaluate(model, validation_loader)
    cm = ConfusionMatrix(actual_vector=y_gt, predict_vector=y_pred)

    return cm.AUNU


def train_validate_AUCROC(train_loader, validation_loader, lr, momentum,
                          model, optimizer, criterion, epochs=30,
                          device='cpu', seed=42):
    """
    Similar to train_validate function. However, this function uses
    ROC-AUC metric rather than accuracy

    Args:
        train_loader (DataLoader): DataLoader of the train dataset
        validation_loader (DataLoader): Dataloader of the validation dataset
        lr (float): Learning rate
        momentum (float): Momentum
        model (nn.Module): Model to run training and validation with
        optimizer (torch.optim): Optimization algorithm
        criterion (torch.nn.modules.loss): Loss criterion
        epochs (int, optional): [description]. Defaults to 30.
        device (str, optional): [description]. Defaults to 'cpu'.
        seed (int, optional): [description]. Defaults to 42.

    Returns:
        nn.Module: Trained model
        dict: Logs of run
    """
    # Set up the model and send it to the GPU (or CPU if one isn't available)
    set_seed(seed)
    model = model.to(device)

    liveloss = PlotLosses()
    for epoch in range(epochs):
        logs = {}
        train_loss, train_accuracy = train(model, optimizer, criterion,
                                           train_loader, device=device)

        # Update the logs for the training data
        logs['' + 'log loss'] = train_loss.item()
        logs['' + 'accuracy'] = train_accuracy.item()

        # Update the logs for the validation data
        auc_score = auc(model, validation_loader)
        logs['val_' + 'auc'] = auc_score
        # logs['val_' + 'accuracy'] = validation_accuracy.item()
        liveloss.update(logs)

        liveloss.draw()

    return model, logs


"""
Function not yet implemented

def roc_auc_score_multiclass(actual_class, pred_class, average="macro"):
    # Create a set of the individual classes from the actual_class list
    unique_class = set(actual_class)
    roc_auc_dict = {}

    for per_class in unique_class:
        # Create a list of all classes except the current class
        other_classes = [x for x in unique_class if x != per_class]

        # Mark current class as one and all other classes as 0
        new_actual_class = [0 if x in other_classes else 1 for x in
                            actual_class]
        new_pred_class = [0 if x in other_classes else 1 for x in pred_class]

        # sklearn metric roc_auc_score to find score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class,
                                average=average)

        roc_auc_dict[per_class] = roc_auc
    # Returns a dictionary of the classes and their scores based on roc_auc
    return roc_auc_dict
"""
