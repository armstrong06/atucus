import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

sys.path.append("/home/armstrong/Research/git_repos/patprob/swag_modified")
from swag import seismic_data, models, utils, losses
from swag.posteriors import SWAG

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

# parser.add_argument(
#     "--dataset", type=str, default="CIFAR10", help="dataset name (default: CIFAR10)"
# )
parser.add_argument(
    "--data_path",
    type=str,
    default=None,
    required=True,
    metavar="PATH",
    help="path to datasets location (default: None)",
)

parser.add_argument(
    "--train_dataset",
    help="Name of the training dataset",
)
parser.add_argument(
    "--validation_dataset",
    help="Name of the validation dataset",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    metavar="N",
    help="number of workers (default: 4)",
)

parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)
parser.add_argument(
    "--load_model",
    type=str,
    default=None,
    metavar="INTMODEL",
    help="Path to initial model to load (default: None)",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to resume training from (default: None)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    metavar="N",
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--save_freq",
    type=int,
    default=25,
    metavar="N",
    help="save frequency (default: 25)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    metavar="N",
    help="evaluation frequency (default: 5)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)

parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
parser.add_argument(
    "--swa_start",
    type=float,
    default=161,
    metavar="N",
    help="SWA start epoch number (default: 161)",
)
parser.add_argument(
    "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
)
parser.add_argument(
    "--swa_c_epochs",
    type=int,
    default=1,
    metavar="N",
    help="SWA model collection frequency/cycle length in epochs (default: 1)",
)
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument(
    "--max_num_models",
    type=int,
    default=20,
    help="maximum number of SWAG models to save",
)

parser.add_argument(
    "--swa_resume",
    type=str,
    default=None,
    metavar="CKPT",
    help="checkpoint to restor SWA from (default: None)",
)
parser.add_argument(
    "--loss",
    type=str,
    default="MSE",
    help="loss to use for training model (default: Mean-square error)",
)

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument("--no_schedule", action="store_true", help="store schedule")

args = parser.parse_args()

args.device = None

use_cuda = torch.cuda.is_available()

if use_cuda:
    args.device = torch.device("cuda:0")
else:
    args.device = torch.device("cpu")

print("--no_schedule:", args.no_schedule)

param_columns = ["BatchSize", "SGD_lr", "WD", "Mom", "SWA_lr", "K"]
param_values = [args.batch_size,
                args.lr_init, 
                args.wd, 
                args.momentum, 
                args.swa_lr, 
                args.max_num_models
]

table = tabulate.tabulate([param_values], param_columns, tablefmt="simple", floatfmt="8.5f")
table = table.split("\n")
table = "\n".join([table[1]] + table)
print(table)

print("Preparing directory %s" % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

# Random seeds 
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
print("Set seed to", args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading datasets %s and %s from %s" % (args.train_dataset, args.validation_dataset, args.data_path))
loaders = seismic_data.loaders(
    args.train_dataset,
    args.validation_dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
)

print("Preparing model")
print(*model_cfg.args)

model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
model.to(args.device)

if args.load_model is not None:
    print("Loading model:", args.load_model)
    check_point = torch.load(args.load_model)
    print("Starting loss:", check_point["training_loss"])
    model.load_state_dict(check_point['model_state_dict'])

if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True

if args.swa:
    print("SWAG training")
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        *model_cfg.args,
        **model_cfg.kwargs
    )
    swag_model.to(args.device)
else:
    print("SGD training")


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


# use a slightly modified loss function that allows input of model
if args.loss == "MSE":
    criterion = losses.mse_loss

optimizer = torch.optim.SGD(
    model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print("Resume training from %s" % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

if args.swa and args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(
        model_cfg.base,
        no_cov_mat=args.no_cov_mat,
        max_num_models=args.max_num_models,
        *model_cfg.args,
        **model_cfg.kwargs
    )
    #loading=True,
    swag_model.to(args.device)
    swag_model.load_state_dict(checkpoint["state_dict"])
    start_epoch = checkpoint["epoch"]

columns = ["ep", "lr", "tr_loss", "tr_rms", "va_loss", "va_rms", "time", "mem_usage"]
if args.swa:
    columns = columns[:-2] + ["swa_va_loss"] + ["swa_va_rms"] + columns[-2:]
    swag_res = {"loss": None, "rms": None}

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.0
values_over_epochs = []

if args.load_model is not None:
    print("Evaluating the initial model...")
    test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda, regression=True)
    train_res = utils.eval(loaders["train"], model, criterion, cuda=use_cuda, regression=True)
    values = [
        0,
        args.lr_init,
        train_res["loss"],
        train_res["rms"],
        test_res["loss"],
        test_res["rms"],
        None,
        None,
    ]

    if args.swa:
        values = values[:-2] + [None] + [None] + values[-2:]
    
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.5f")
    table = table.split("\n")
    table = "\n".join([table[1]] + table)
   
    print(table)
    values_over_epochs.append(values)

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict(),
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        lr = schedule(epoch)
    elif (args.swa and (epoch + 1) > args.swa_start):
        lr = args.swa_lr
    else:
        lr = args.lr_init

    utils.adjust_learning_rate(optimizer, lr)

    if (args.swa and (epoch + 1) > args.swa_start) and args.cov_mat:
        train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, cuda=use_cuda, regression=True)
    else:
        # why is this the same? - Alysha
        train_res = utils.train_epoch(loaders["train"], model, criterion, optimizer, cuda=use_cuda, regression=True)

    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        test_res = utils.eval(loaders["test"], model, criterion, cuda=use_cuda, regression=True)
    else:
        test_res = {"loss": None, "rms":None}

    if (
        args.swa
        and (epoch + 1) > args.swa_start
        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
    ):
        # sgd_preds, sgd_targets = utils.predictions(loaders["test"], model)
        sgd_res = utils.predict(loaders["test"], model, regression=True)
        sgd_preds = sgd_res["predictions"]
        sgd_targets = sgd_res["targets"]
        print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            # TODO: rewrite in a numerically stable way
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (
                n_ensembled + 1
            ) + sgd_preds / (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(model)
        if (
            epoch == 0
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epochs - 1
        ):
            swag_model.sample(0.0)
            utils.bn_update(loaders["train"], swag_model)
            swag_res = utils.eval(loaders["test"], swag_model, criterion, regression=True)
        else:
            swag_res = {"loss": None, "rms":None}

    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
        )
        if args.swa and epoch + 1 > args.swa_start:
            utils.save_checkpoint(
                args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
            )

    time_ep = time.time() - time_ep
    
    if use_cuda:
        memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)
        
    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["rms"],
        test_res["loss"],
        test_res["rms"],
        time_ep,
        memory_usage,
    ]
    if args.swa:
        values = values[:-2] + [swag_res["loss"]] + [swag_res["rms"]] + values[-2:]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.5f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
    values_over_epochs.append(values)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    # I think args.epochs > args.swa_start should be epoch > args.swa_start
    if args.swa and epoch + 1 > args.swa_start:
        utils.save_checkpoint(
            args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
        )

if args.swa:
    np.savez(
        os.path.join(args.dir, "sgd_ens_preds.npz"),
        predictions=sgd_ens_preds,
        targets=sgd_targets,
    )

if args.swa:
    residuals = sgd_targets - sgd_ens_preds
    of_mean, of_std = utils.compute_outer_fence_mean_standard_deviation(residuals)
    print("Stats for sgd ensemble residuals")
    print("Mean    STD    OF_Mean    OF_STD     RMS")
    print(np.mean(residuals), np.std(residuals), of_mean, of_std, np.sqrt(np.sum(residuals**2)/len(residuals)))

#np.savetxt(os.path.join(args.dir, "train_metrics.out"), np.array(values_over_epochs))
df = pd.DataFrame(values_over_epochs, columns=columns)
df.to_csv(os.path.join(args.dir, "train_metrics.csv"), index=None)
