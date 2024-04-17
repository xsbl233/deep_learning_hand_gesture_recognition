from __future__ import unicode_literals, print_function, division
import sys
import numpy
import pickle
import torch
import itertools
import time
import math
from torch.utils.data import Dataset, DataLoader

import os


class HandGestureNet(torch.nn.Module):
    """
    [Devineau et al., 2018] Deep Learning for Hand Gesture Recognition on Skeletal Data

    Summary
    -------
        Deep Learning Model for Hand Gesture classification using pose data only (no need for RGBD)
        The model computes a succession of [convolutions and pooling] over time independently on each of the 66 (= 22 * 3) sequence channels.
        Each of these computations are actually done at two different resolutions, that are later merged by concatenation
        with the (pooled) original sequence channel.
        Finally, a multi-layer perceptron merges all of the processed channels and outputs a classification.

    TL;DR:
    ------
        input ------------------------------------------------> split into n_channels channels [channel_i]
            channel_i ----------------------------------------> 3x [conv/pool/dropout] low_resolution_i
            channel_i ----------------------------------------> 3x [conv/pool/dropout] high_resolution_i
            channel_i ----------------------------------------> pooled_i
            low_resolution_i, high_resolution_i, pooled_i ----> output_channel_i
        MLP(n_channels x [output_channel_i]) -------------------------> classification

    Article / PDF:
    --------------
        https://ieeexplore.ieee.org/document/8373818

    Please cite:
    ------------
        @inproceedings{devineau2018deep,
            title={Deep learning for hand gesture recognition on skeletal data},
            author={Devineau, Guillaume and Moutarde, Fabien and Xi, Wang and Yang, Jie},
            booktitle={2018 13th IEEE International Conference on Automatic Face \& Gesture Recognition (FG 2018)},
            pages={106--113},
            year={2018},
            organization={IEEE}
        }
    """

    def __init__(self, n_channels=66, n_classes=14, dropout_probability=0.2):

        super(HandGestureNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dropout_probability = dropout_probability

        # Layers ----------------------------------------------
        self.all_conv_high = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.all_conv_low = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=8, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool1d(2),

            torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.dropout_probability),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.all_residual = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2),
            torch.nn.AvgPool1d(2)
        ) for joint in range(n_channels)])

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=9 * n_channels * 12, out_features=1936),
            # <-- 12: depends of the sequences lengths (cf. below)
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=1936, out_features=n_classes)
        )

        # Initialization --------------------------------------
        # Xavier init
        for module in itertools.chain(self.all_conv_high, self.all_conv_low, self.all_residual):
            for layer in module:
                if layer.__class__.__name__ == "Conv1d":
                    torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                    torch.nn.init.constant_(layer.bias, 0.1)

        for layer in self.fc:
            if layer.__class__.__name__ == "Linear":
                torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
                torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, input):
        """
        This function performs the actual computations of the network for a forward pass.

        Arguments
        ---------
            input: a tensor of gestures of shape (batch_size, duration, n_channels)
                   (where n_channels = 3 * n_joints for 3D pose data)
        """

        # Work on each channel separately
        all_features = []

        for channel in range(0, self.n_channels):
            input_channel = input[:, :, channel]
            # Add a dummy (spatial) dimension for the time convolutions
            # Conv1D format : (batch_size, n_feature_maps, duration)
            input_channel = input_channel.unsqueeze(1)

            high = self.all_conv_high[channel](input_channel)
            low = self.all_conv_low[channel](input_channel)
            ap_residual = self.all_residual[channel](input_channel)

            # Time convolutions are concatenated along the feature maps axis
            output_channel = torch.cat([
                high,
                low,
                ap_residual
            ], dim=1)
            all_features.append(output_channel)

        # Concatenate along the feature maps axis
        all_features = torch.cat(all_features, dim=1)

        # Flatten for the Linear layers
        all_features = all_features.view(-1,
                                         9 * self.n_channels * 12)  # <-- 12: depends of the initial sequence length (100).
        # If you have shorter/longer sequences, you probably do NOT even need to modify the modify the network architecture:
        # resampling your input gesture from T timesteps to 100 timesteps will (surprisingly) probably actually work as well!

        # Fully-Connected Layers
        output = self.fc(all_features)

        return output


def load_data(filepath='./shrec_data.pckl'):
    """
    Returns hand gesture sequences (X) and their associated labels (Y).
    Each sequence has two different labels.
    The first label  Y describes the gesture class out of 14 possible gestures (e.g. swiping your hand to the right).
    The second label Y describes the gesture class out of 28 possible gestures (e.g. swiping your hand to the right with your index pointed, or not pointed).
    """
    file = open(filepath, 'rb')
    data = pickle.load(file, encoding='latin1')  # <<---- change to 'latin1' to 'utf8' if the data does not load
    file.close()
    return data['x_train'], data['x_test'], data['y_train_14'], data['y_train_28'], data['y_test_14'], data['y_test_28']

class GestureDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '{:02d}m {:02d}s'.format(int(m), int(s))


def get_accuracy(model, x, y_ref):
    """Get the accuracy of the pytorch model on a batch"""
    acc = 0.
    model.eval()
    with torch.no_grad():
        predicted = model(x)
        _, predicted = predicted.max(dim=1)
        #print(_, '!\n', predicted)
        acc = 1.0 * (predicted == y_ref).sum().item() / y_ref.shape[0]
    return acc



# -------------
# Training
# -------------


def train(model, criterion, optimizer, dataloader,
          x_train, y_train, x_test, y_test,
          force_cpu=False, num_epochs=5):
    # use a GPU (for speed) if you have one
    device = torch.device("cuda") if torch.cuda.is_available() and not force_cpu else torch.device("cpu")
    model = model.to(device)
    print(x_train.size(),x_test.size(), y_train.size(),y_test.size())
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    # (bonus) log accuracy values to visualize them in tensorboard:
    writer = SummaryWriter()

    # Training starting time
    start = time.time()

    print('[INFO] Started to train the model.')
    print('Training the model on {}.'.format('GPU' if device == torch.device('cuda') else 'CPU'))

    for ep in range(num_epochs):

        # Ensure we're still in training mode
        model.train()

        current_loss = 0.0

        for idx_batch, batch in enumerate(dataloader):
            # Move data to GPU, if available
            x, y = batch
            x, y = x.to(device), y.to(device)

            # zero the gradient parameters
            optimizer.zero_grad()

            # forward
            y_pred = model(x)

            # backward + optimize
            # backward
            loss = criterion(y_pred, y)
            loss.backward()
            # optimize
            optimizer.step()
            # for an easy access
            current_loss += loss.item()

        train_acc = get_accuracy(model, x_train, y_train)
        test_acc = get_accuracy(model, x_test, y_test)

        writer.add_scalar('data/accuracy_train', train_acc, ep)
        writer.add_scalar('data/accuracy_test', test_acc, ep)
        print(
            'Epoch #{:03d} | Time elapsed : {} | Loss : {:.4e} | Accuracy_train : {:.2f}% | Accuracy_test : {:.2f}% '.format(
                ep + 1, time_since(start), current_loss, 100 * train_acc, 100 * test_acc))

    print('[INFO] Finished training the model. Total time : {}.'.format(time_since(start)))

if __name__ == '__main__':
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:true'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


    # (bonus) plot acc with tensorboard
    #   Command to start tensorboard if installed (requires tensorflow):
    #   $  tensorboard --logdir ./runs
    try:
        from tensorboardX import SummaryWriter
    except:
        # tensorboardX is not installed, just fail silently
        class SummaryWriter():
            def __init__(self):
                pass

            def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
                pass

    n_classes = 14
    duration = 100
    n_channels = 63
    learning_rate = 1e-3


    x_train, x_test, y_train_14, y_train_28, y_test_14, y_test_28 = load_data('dhg_data.pckl')

    y_train_14, y_test_14 = numpy.array(y_train_14), numpy.array(y_test_14)
    y_train_28, y_test_28 = numpy.array(y_train_28), numpy.array(y_test_28)
    if n_classes == 14:
        y_train = y_train_14
        y_test = y_test_14
    elif n_classes == 28:
        y_train = y_train_28
        y_test = y_test_28

    print(x_train.size,x_test.size)
    print(y_train.size,y_test.size)


    # -------------
    # Network instantiation
    # -------------
    model = HandGestureNet(n_channels=n_channels, n_classes=n_classes)


    # ------------------------
    # Create pytorch datasets and dataloaders:
    # ------------------------
    # Convert from numpy to torch format
    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    y_train, y_test = torch.from_numpy(y_train), torch.from_numpy(y_test)

    print(x_train.size(), x_test.size(), y_train.size(), y_test.size())
    # Ensure the label values are between 0 and n_classes-1
    if y_train.min() > 0:
        y_train = y_train - 1
    if y_test.min() > 0:
        y_test = y_test - 1

    # Ensure the data type is correct
    x_train, x_test = x_train.float(), x_test.float()
    y_train, y_test = y_train.long(), y_test.long()
    # Create the datasets
    train_dataset = GestureDataset(x=x_train, y=y_train)
    test_dataset = GestureDataset(x=x_test, y=y_test)

    # Pytorch dataloaders are used to group dataset items into batches
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=False,num_workers  = 8,pin_memory = True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True,drop_last=False,num_workers  = 8,pin_memory = True)


    # -----------------------------------------------------
    # Loss function & Optimizer
    # -----------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    # Please adjust the training epochs count, and the other hyperparams (lr, dropout, ...), for a non-overfitted training according to your own needs.
    # tip: use tensorboard to display the accuracy (see cells above for tensorboard usage)

    num_epochs = 1000

    print(x_train.size(), x_test.size(), y_train.size(), y_test.size())
    train(model=model, criterion=criterion, optimizer=optimizer, dataloader=train_dataloader,
          x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
          num_epochs=num_epochs, force_cpu=True)

    torch.save(model.state_dict(), 'gesture_pretrained_model.pt')

    # Reminder: first redefine/load the HandGestureNet class before you use it, if you want to use it elsewhere
    model = HandGestureNet(n_channels=n_channels, n_classes=n_classes)
    model.load_state_dict(torch.load('gesture_pretrained_model.pt'))
    model.eval()

    # make predictions
    with torch.no_grad():
        demo_gesture_batch = torch.randn(32, duration, n_channels)
        predictions = model(demo_gesture_batch)
        _, predictions = predictions.max(dim=1)
        print("Predicted gesture classes: {}".format(predictions.tolist()))
