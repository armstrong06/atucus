import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import tqdm
import tabulate
import sys
sys.path.append("/home/armstrong/Research/git_repos/patprob/swag_modified")
from swag import seismic_data, losses, models, utils
from swag.posteriors import SWAG #, KFACLaplace
import pandas as pd
import time

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument("--file", type=str, default=None, required=True, help="checkpoint")

parser.add_argument(
    "--train_dataset",
    help="Name of the training dataset",
)
parser.add_argument(
    "--validation_dataset",
    help="Name of the validation dataset",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="/scratch/datasets/",
    metavar="PATH",
    help="path to datasets location (default: None)",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument("--split_classes", type=int, default=None)
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
    default="PPicker",
    metavar="MODEL",
    help="model name (default: PPicker)",
)
parser.add_argument(
    "--method",
    type=str,
    default="SWAG",
    choices=["SWAG", "KFACLaplace", "SGD", "HomoNoise", "Dropout", "SWAGDrop"],
    required=True,
)
parser.add_argument(
    "--save_path",
    type=str,
    default=None,
    required=True,
    help="path to npz results file",
)
parser.add_argument("--N", type=int, default=30)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument(
    "--cov_mat", action="store_true", help="use sample covariance for swag"
)
parser.add_argument("--use_diag", action="store_true", help="use diag cov for swag")

parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)
parser.add_argument(
    "--max_num_models",
    type=int,
    default=20,
    help="maximum number of SWAG models to save",
)
args = parser.parse_args()

dir = os.path.dirname(args.save_path)
print("Preparing directory %s" % dir)
if not os.path.exists(dir):
    os.makedirs(dir)

eps = 1e-12
if args.cov_mat:
    args.cov_mat = True
else:
    args.cov_mat = False

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading datasets %s and %s from %s" % (args.train_dataset, args.validation_dataset, args.data_path))
loaders = seismic_data.loaders(
    args.train_dataset,
    args.validation_dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    shuffle_train=False,
)

"""if args.split_classes is not None:
    num_classes /= 2
    num_classes = int(num_classes)"""

print("Preparing model")
if args.method in ["SWAG", "HomoNoise", "SWAGDrop"]:
    model = SWAG(
        model_cfg.base,
        no_cov_mat=not args.cov_mat,
        max_num_models=args.max_num_models,
        *model_cfg.args,
        **model_cfg.kwargs
    )
elif args.method in ["SGD", "Dropout", "KFACLaplace"]:
    model = model_cfg.base(*model_cfg.args, **model_cfg.kwargs)
else:
    assert False
model.cuda()


def train_dropout(m):
    if type(m) == torch.nn.modules.dropout.Dropout:
        m.train()


print("Loading model %s" % args.file)
checkpoint = torch.load(args.file)
model.load_state_dict(checkpoint["state_dict"])

if args.method == "KFACLaplace":
    print(len(loaders["train"].dataset))
    model = KFACLaplace(
        model, eps=5e-4, data_size=len(loaders["train"].dataset)
    )  # eps: weight_decay

    t_input, t_target = next(iter(loaders["train"]))
    t_input, t_target = (
        t_input.cuda(non_blocking=True),
        t_target.cuda(non_blocking=True),
    )

if args.method == "HomoNoise":
    std = 0.01
    for module, name in model.params:
        mean = module.__getattr__("%s_mean" % name)
        module.__getattr__("%s_sq_mean" % name).copy_(mean ** 2 + std ** 2)


predictions = np.zeros((len(loaders["test"].dataset), args.N))
targets = np.zeros((len(loaders["test"].dataset), 1))
print(targets.size)

residual_summary = []

starttime = time.time()
for i in range(args.N):
    print("%d/%d" % (i + 1, args.N))
    if args.method == "KFACLaplace":
        ## KFAC Laplace needs one forwards pass to load the KFAC model at the beginning
        model.net.load_state_dict(model.mean_state)

        if i == 0:
            model.net.train()

            loss, _ = losses.cross_entropy(model.net, t_input, t_target)
            loss.backward(create_graph=True)
            model.step(update_params=False)

    # Do if Method is NOT SGD or Dropout
    if args.method not in ["SGD", "Dropout"]:
        sample_with_cov = args.cov_mat and not args.use_diag
        model.sample(scale=args.scale, cov=sample_with_cov)

    if "SWAG" in args.method:
        utils.bn_update(loaders["train"], model)

    model.eval()
    if args.method in ["Dropout", "SWAGDrop"]:
        model.apply(train_dropout)
        # torch.manual_seed(i)
        # utils.bn_update(loaders['train'], model)

    k = 0
    for input, target in tqdm.tqdm(loaders["test"]):
        input = input.cuda(non_blocking=True)
        ##TODO: is this needed?
        # if args.method == 'Dropout':
        #    model.apply(train_dropout)
        torch.manual_seed(i)

        if args.method == "KFACLaplace":
            output = model.net(input)
        else:
            output = model(input)

        with torch.no_grad():
            predictions[k : k + input.size()[0], i:i+1] = output.cpu().numpy()
            
        targets[k : (k + target.size(0))] = target.numpy()
        k += input.size()[0]

    print(time.time()-starttime)

    # Residual stats from the i'th set of predictions
    residuals = targets[:, 0] - predictions[:, i]
    of_mean, of_std = utils.compute_outer_fence_mean_standard_deviation(residuals)
    # Residual stats from the i'th ensemble (predictions from 0 to i)
    pred_mean_N = np.mean(predictions[:, 0:i+1], axis=1)
    # pred_std_N = np.std(predictions[:, 0:i+1], axis=1)
    N_ensemble_residuals = targets[:, 0] - pred_mean_N
    of_mean_N, of_std_N = utils.compute_outer_fence_mean_standard_deviation(N_ensemble_residuals)
    residual_summary.append([np.mean(residuals), np.std(residuals), of_mean, of_std, np.sqrt(np.sum(residuals**2)/len(residuals)),
    np.mean(N_ensemble_residuals), np.std(N_ensemble_residuals), of_mean_N, of_std_N, np.sqrt(np.sum(N_ensemble_residuals**2)/len(N_ensemble_residuals))])
    
    print("Residual Mean:", np.mean(residuals))
    print("Residual STD:", np.std(residuals))
    print("Residual RMS:", np.sqrt(np.sum(residuals**2)/len(residuals)))

# mean of sampled predictions - don't do this since storing all of them
# predictions /= args.N

print("total time:", time.time()-starttime)
#final_residuals = targets - predictions
pred_mean = np.mean(predictions, axis=1)
pred_std = np.std(predictions, axis=1)

# Don't feel like this has the same meaning for regression
# entropies = -np.sum(np.log(predictions + eps) * predictions, axis=1)
np.savez(args.save_path, predictions=predictions, targets=targets,
         prediction_mean=pred_mean, prediction_std=pred_std)

resids = targets[:, 0] - pred_mean
of_mean, of_std = utils.compute_outer_fence_mean_standard_deviation(resids)
print("Stats for sgd ensemble residuals")
print("Mean    STD    OF_Mean    OF_STD,    RMS")
print(np.mean(resids), np.std(resids), of_mean, of_std, np.sqrt(np.sum(resids**2)/len(resids)))

df = pd.DataFrame(residual_summary, columns=["i_residual_mean", "i_residual_std", "i_of_mean", "i_of_std", "i_rms",
                                                "ens_residual_mean", "ens_residual_std", "ens_of_mean", "ens_of_std", "ens_rms"])
df.to_csv(f"{args.save_path}_residual_summary_{args.N}.csv", index=False)