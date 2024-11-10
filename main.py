"""
To train and test the model
"""

import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trans

import DVSLip
import models

parser = argparse.ArgumentParser()    
parser.add_argument('-f', dest='filename', default='test', type=str, help='filename to store the model')
parser.add_argument('-t', dest='is_test', action='store_true', default=False, help='test only')
parser.add_argument('-e', dest='epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('-a', dest='is_ann', action='store_true', default=False, help='ann network')
parser.add_argument("-d", dest="is_delayed", action="store_true", default=False, help="delayed network")
parser.add_argument("--axonal", dest="has_axonal_delay", action="store_true", default=False, help="axonal-delayed network")
parser.add_argument('--actreg', default=0.0, type=float, help='activity regularization for SNNs')
parser.add_argument('--finetune', action='store_true', default=False, help='restart training from the given model')
parser.add_argument('--nbframes', default=30, type=int, help='nb of frames for data pre-processing')
parser.add_argument('-b', dest='batch_size', default=32, type=int, help='training batch_size')
parser.add_argument('--augS', action='store_true', default=False, help='spatial data augmentation (for training)')
parser.add_argument('--augT', action='store_true', default=False, help='temporal data augmentation (for training)')
parser.add_argument('--ternact', dest='ternact', action='store_true', default=False, help='SNN with ternary activations')
parser.add_argument('--useBN', dest='useBN', action='store_true', default=False, help='use batch norm in Conv layers')
parser.add_argument('--front', action='store_true', default=False, help='train front end (resnet) only')
parser.add_argument('--NObidirectional', action='store_true', default=False, help='NO bidirectional GRU')
parser.add_argument('--singlegate', action='store_true', default=False, help='NO two gates GRU (single gate)')
parser.add_argument('--hybridsign', action='store_true', default=False, help='hybrid signed SNN : bin frontend / tern backend')
parser.add_argument('--hybridANN', action='store_true', default=False, help='hybrid ANN-SNN : SNN frontend / ANN backend')
parser.add_argument('--nowarmup', action='store_true', default=False, help='no warmup epoch')
parser.add_argument('--Tnbmask', default=6, type=int, help='nb of masks for temporal data augmentation')
parser.add_argument('--Tmaxmasklength', default=18, type=int, help='maximale length of each mask for temporal data augmentation')
parser.add_argument("--round", action="store_true", default=False, help="round positions")
parser.add_argument('--checkpoint_name', type=str, help="checkpoint model name")
parser.add_argument("--change", action="store_true", default=False, help="change state dict")

args = parser.parse_args()
device = torch.device("cuda:0")
dtype = torch.float
SAVE_PATH_MODEL_BEST = os.getcwd() + '/' + args.filename + '.pt'
IS_CHANGED = False

MODEL_BASE_PATH = os.path.expanduser('~/paper_runs')

## DATASET
####################################################################
train_data_root = "/home/tahaf/data/DVS-Lip/train"
test_data_root = "/home/tahaf/data/DVS-Lip/test"
training_words = DVSLip.get_training_words()
label_dct = {k:i for i,k in enumerate(training_words)}
## label_dct: {'education': 0, 'between': 1, 'london': 2, 'allow': 3, 'military': 4, 'warning': 5, 'little': 6, 'press': 7, 'missing': 8, 'numbers': 9, 'change': 10, 'support': 11, 'immigration': 12, 'started': 13, 'still': 14, 'attacks': 15, 'called': 16, 'another': 17, 'security': 18, 'minutes': 19, 'point': 20, 'general': 21, 'judge': 22, 'hundreds': 23, 'spend': 24, 'described': 25, 'million': 26, 'having': 27, 'young': 28, 'syria': 29, 'evening': 30, 'american': 31, 'difference': 32, 'russian': 33, 'taken': 34, 'potential': 35, 'russia': 36, 'terms': 37, 'banks': 38, 'leaders': 39, 'welcome': 40, 'house': 41, 'labour': 42, 'words': 43, 'challenge': 44, 'taking': 45, 'worst': 46, 'everything': 47, 'really': 48, 'needs': 49, 'america': 50, 'allowed': 51, 'under': 52, 'thing': 53, 'happened': 54, 'price': 55, 'syrian': 56, 'benefit': 57, 'paying': 58, 'right': 59, 'tomorrow': 60, 'capital': 61, 'question': 62, 'germany': 63, 'meeting': 64, 'these': 65, 'couple': 66, 'saying': 67, 'billion': 68, 'majority': 69, 'think': 70, 'accused': 71, 'giving': 72, 'action': 73, 'become': 74, 'economic': 75, 'times': 76, 'different': 77, 'perhaps': 78, 'benefits': 79, 'court': 80, 'water': 81, 'death': 82, 'during': 83, 'chief': 84, 'happen': 85, 'being': 86, 'years': 87, 'election': 88, 'ground': 89, 'england': 90, 'exactly': 91, 'should': 92, 'spent': 93, 'several': 94, 'number': 95, 'around': 96, 'significant': 97, 'legal': 98, 'heavy': 99}

train_dataset = DVSLip.DVSLipDataset(train_data_root, label_dct, train=True, augment_spatial=args.augS, augment_temporal=args.augT, T=args.nbframes, Tnbmask=args.Tnbmask, Tmaxmasklength=args.Tmaxmasklength)
test_dataset = DVSLip.DVSLipDataset(test_data_root, label_dct, train=False, augment_spatial=False, augment_temporal=False, T=args.nbframes)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True, pin_memory=True) 
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, shuffle=False, pin_memory=True) 
# print(len(train_dataloader), len(test_dataloader)) # 14896 train / 4975 test so 466 / 156 iter with bs 32


### MODEL ##
###############################################################
model = models.SCNN(args, 100, useBN=args.useBN, ternact=args.ternact, ann=args.is_ann, front=args.front, NObidirectional=args.NObidirectional, singlegate=args.singlegate, hybridsign=args.hybridsign, hybridANN=args.hybridANN, delayed=args.is_delayed, axonal_delay=args.has_axonal_delay)
model.cuda()

print(model)
param_flatten = torch.cat([param.data.view(-1) for param in model.parameters()], 0)
print("nb param:", param_flatten.size())
print(args)


def train(model, loss_fn, optimizer, train_dataloader, valid_dataloader, nb_epochs, scheduler=None, warmup_epochs=0):
    """ 
    Train the model
    """
    if warmup_epochs > 0:
        for g in optimizer.param_groups:
            g['lr'] /= len(train_dataloader)*warmup_epochs
        warmup_itr = 1
    best_val = 0; best_epoch = 0

    for e in range(nb_epochs):
        epoch_start = time.time()
        local_loss = []
        train_accs = []
            
        for ni, (x_batch, y_batch) in enumerate(train_dataloader):
            model.train()
            x_batch = x_batch.to(device, dtype, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            output, loss_act, spike_act = model(x_batch)
            
            log_p_y = torch.mean(output, dim=1)
            loss_val = loss_fn(log_p_y, y_batch) 

            if args.actreg > 0:
                loss_val += args.actreg * loss_act

            am = torch.argmax(log_p_y, dim=1)
            tmp = np.mean((y_batch==am).detach().cpu().numpy())
            train_accs.append(tmp)

            local_loss.append(loss_val.item())

            optimizer.zero_grad()
            loss_val.backward()

            if ni % 100 == 0:
                print("iter %i loss %.4f loss_act %.4f lr: %.5f"%(ni, loss_val.item(), loss_act.item(), optimizer.param_groups[0]["lr"]))
                # if not args.is_ann:
                #     for i in range(len(spike_act)):
                #         print(spike_act[i], end = ' ')
                #     print("")

            optimizer.step()

            model.clamp()

            if e < warmup_epochs:
                for g in optimizer.param_groups:
                    g['lr'] *= (warmup_itr+1)/(warmup_itr)
                warmup_itr += 1


        mean_loss = np.mean(local_loss)
        train_accuracy = np.mean(train_accs)
        print("Epoch %i: loss=%.5f, Training accuracy=%.3f"%(e+1, mean_loss, train_accuracy))
        
        valid_accuracy = compute_classification_accuracy(model, valid_dataloader, valid=True)
        print("Validation accuracy=%.3f"%(valid_accuracy))

        if scheduler is not None and e >= warmup_epochs:
            scheduler.step()
        
        if valid_accuracy > best_val:
            print("better valid_accuracy")
            torch.save(model.state_dict(), SAVE_PATH_MODEL_BEST)
            best_val = valid_accuracy
            best_epoch = e

        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        print("Epoch (training) took {:.3f}s".format(epoch_duration))

        with open('res_' + args.filename + '.txt', 'a') as f:
            f.write(
                "epoch %i: train: %.2f, val: %.2f, loss: %.5f, loss_reg: %.3f, lr: %.5f, epoch duration: %.3f\n"
                % (
                    e+1,
                    train_accuracy*100, 
                    valid_accuracy*100, 
                    mean_loss, 
                    loss_act, 
                    optimizer.param_groups[0]["lr"],
                    epoch_duration
                )
            )
                 

    with open('res_' + args.filename + '.txt', 'a') as f:
        f.write("best epoch, accu (val): %i %.2f"%(best_epoch +1, best_val*100))
        f.write('\n')
    exit()

def compute_classification_accuracy(model, dataloader, valid=False):
    """ 
    Evaluate the model on the given dataset (accuracy and spike rate). 
    If valid=True, do not compute the spike rate. 
    """
    accs = np.array([])
    avg_spike_act = None
    avg_in_act = []

    model.eval()
    with torch.no_grad():
        for ni, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device, dtype, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            output, loss_act, spike_act = model(x_batch)

            log_p_y = torch.mean(output, dim=1)

            am = torch.argmax(log_p_y, dim=1)
            tmp = (y_batch==am).view(-1).detach().cpu().numpy()
            accs = np.concatenate((accs, tmp))

            if avg_spike_act == None:
                avg_spike_act = [ [] for i in range(len(spike_act))]
            for i,l in enumerate(spike_act):
                avg_spike_act[i].append(l)
            avg_in_act.append(x_batch.abs().mean().detach().cpu().numpy())

            # if ni % 20 == 0:
            #     print("testing iter...", ni, np.mean(accs))

    testaccu = np.mean(accs)
    print("Test accuracy:", testaccu)

    if len(spike_act) == 22:
        layer_names = ["conv1", "pool1", "conv2_1", "conv2_11", "conv2_2", "conv2_21", "conv3_1", "conv3_11", "conv3_2", "conv3_21", "conv4_1", "conv4_11", "conv4_2","conv4_21","conv5_1","conv5_11","conv5_2", "conv5_21","avgpool","gru1", "gru2", "gru3"]
    else:
        layer_names = ["conv1", "pool1", "conv2_1", "conv2_11", "conv2_2", "conv2_21", "conv3_1", "conv3_11", "conv3_2", "conv3_21", "conv4_1", "conv4_11", "conv4_2","conv4_21","conv5_1","conv5_11","conv5_2", "conv5_21","avgpool"]
    if not args.is_test and not valid:
            with open('res_' + args.filename + '.txt', 'a') as f:
                f.write("INPUT activity: %0.4f \n"%(np.mean(avg_in_act)))
                for i,l in enumerate(avg_spike_act):
                    f.write("avg spike activity %s: %0.4f \n"%(layer_names[i], np.mean(avg_spike_act[i])))
                f.write("# avg spike activity ALL: %0.4f \n"%(np.mean(avg_spike_act)))
    else:
        print("INPUT activity: %0.4f"%(np.mean(avg_in_act)))
        for i,l in enumerate(avg_spike_act):
            print("avg spike activity %s: %0.4f"%(layer_names[i], np.mean(avg_spike_act[i])))
        print("# avg spike activity ALL: %0.4f"%(np.mean(avg_spike_act)))

    return testaccu

def adjust_state_dict(state_dict):
    ## TODO: FIX ME!
    def is_changed(key):
        return key in ["layer2_1.conv1.weight", "layer2_1.conv2.weight", "layer2_2.conv1.weight",
                        "layer2_2.conv2.weight", "layer3_1.conv1.weight", "layer3_1.conv2.weight",
                        "layer3_1.downsample.weight", "layer3_2.conv1.weight", "layer3_2.conv2.weight",
                        "layer4_1.conv1.weight", "layer4_1.conv2.weight", "layer4_1.downsample.weight",
                        "layer4_2.conv1.weight", "layer4_2.conv2.weight", "layer5_1.conv1.weight",
                        "layer5_1.conv2.weight", "layer5_1.downsample.weight", "layer5_2.conv1.weight",
                        "layer5_2.conv2.weight"]
    
    keys = list(state_dict.keys())
    for key in keys:
        if is_changed(key) and IS_CHANGED:
            if args.is_delayed:
                param = state_dict[key]
                param = param.unsqueeze(dim=-1)
                state_dict[key] = param
            elif args.has_axonal_delay:
                new_key = key[:-7] + ".conv.weight"
                state_dict[new_key] = state_dict.pop(key)
    
    return state_dict



## TRAINING PARAMETERS
########################################################################
if not args.is_test:
    print("training filename:", args.filename)
    print("training iter:", len(train_dataloader))
    if args.finetune:
        state_dict = torch.load(SAVE_PATH_MODEL_BEST)
        model.load_state_dict(adjust_state_dict(state_dict), strict=False) # strict=False ignores the unmatching keys in both state_dict
        print("FINE TUNE: ######## FILE LOADED:", SAVE_PATH_MODEL_BEST)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    if args.is_ann:
        if args.finetune:
            lr = 1e-4 * (args.batch_size / 32) 
        else:
            lr = 3e-4 * (args.batch_size / 32)
        final_lr = 5e-6 * (args.batch_size / 32)
    else:
        if args.finetune:
            lr = 1e-4 * (args.batch_size / 32)
            final_lr = 5e-6 * (args.batch_size / 32)
        else:
            lr = 3e-4 * (args.batch_size / 32)
            final_lr = lr

    if args.nowarmup:
        warmup_epochs = 0
    else:
        warmup_epochs = 1

    position_params = []
    bn_params = []
    others_params = []
    for name, param in model.named_parameters():
        if "bn" in name:  # no weight decay on batch norm param
            bn_params.append(param)
        elif name.endswith(".P") or name.endswith(".SIG") and param.requires_grad:
            position_params.append(param)
        else:
            others_params.append(param)
    param_groups = [
        {"params": others_params, "lr": lr, "weight_decay": 1e-4},
        {"params": bn_params, "lr": lr, "weight_decay": 0.0},
        {"params": position_params, "lr": lr * 100, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.Adam(param_groups)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=final_lr, last_epoch=-1)
    
    train(model, loss_fn, optimizer, train_dataloader, test_dataloader, args.epochs, scheduler, warmup_epochs)

else:
    model_path = os.path.join(MODEL_BASE_PATH, args.checkpoint_name + '.pt')
    model_state_dict = torch.load(model_path, map_location='cuda')
    if 'model' in model_state_dict.keys():
        model_state_dict = model_state_dict['model']
    if args.change:
        keys = list(model_state_dict.keys())
        for key in keys:
            new_key = key
            if 'downsample' in key:
                new_key = key.replace('downsample.', 'downsample.conv.')
            elif 'conv' in key and 'layer' in key and '_' in key:
                phrases = key.rsplit('.', 1)
                new_key = phrases[0] + '.conv.' + phrases[1]
            model_state_dict[new_key] = model_state_dict.pop(key)

    model.load_state_dict(model_state_dict, strict=True)
    if args.round:
        with torch.no_grad():
            model.round_pos()
    print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
    valid_accuracy = compute_classification_accuracy(model, test_dataloader, valid=True)
    print(f'Test Accuracy: {valid_accuracy}')
    print('###############################')
## LOAD MODEL AND FINAL TEST
######################################
# model.load_state_dict(torch.load(SAVE_PATH_MODEL_BEST), strict=True)
# print("######## FILE LOADED:", SAVE_PATH_MODEL_BEST)

# test_accuracy = compute_classification_accuracy(model, test_dataloader)
# print("test accuracy:" + str(test_accuracy) + "\n")

# if not args.is_test:
#     with open('res_' + args.filename + '.txt', 'a') as f:
#         f.write("\nTest accuracy(noerror) %s \n"%(test_accuracy))
