""" main.py

Learns ODE from scrna data

"""
import os
import matplotlib
# matplotlib.use("Agg")
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from lib.growth_net import GrowthNet
from lib import utils
from lib.visualize_flow import visualize_transform
from lib.viz_scrna import (
    save_trajectory,
    trajectory_to_video,
    save_vectors,
)
from lib.viz_scrna import save_trajectory_density

from geomloss import SamplesLoss

# from train_misc import standard_normal_logprob
from train_misc import (
    count_nfe,
    count_parameters,
    count_total_time,
    add_spectral_norm,
    spectral_norm_power_iteration,
    create_regularization_fns,
    get_regularization,
    append_regularization_to_log,
    build_model_vanilla,
)

from eval_utils import evaluate_kantorovich_v2

import dataset
from parse import parser





def get_transforms(device, args, model, integration_times):
    """
    Given a list of integration points,
    returns a function giving integration times
    """

    def sample_fn(z, logpz=None):
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in integration_times
        ]
        if logpz is not None:
            # TODO this works right?
            for it in int_list:
                z, logpz = model(z, logpz, integration_times=it, reverse=True)
            return z, logpz
        else:
            for it in int_list:
                z = model(z, integration_times=it, reverse=False)
            return z

    def density_fn(x, logpx=None):
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in integration_times[::-1]
        ]
        if logpx is not None:
            for it in int_list:
                x, logpx = model(x, logpx, integration_times=it, reverse=False)
            return x, logpx
        else:
            for it in int_list:
                x = model(x, integration_times=it, reverse=False)
            return x

    return sample_fn, density_fn


def compute_loss(device, args, model, logger, full_data, train_loss_fn, regularization_coeffs):
    """
    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """

    loss = torch.zeros(size=(1, 1)).to(device)
    #FORWARD
    z = None
    for i, (itp, tp) in enumerate(zip(args.int_tps[:-1], args.timepoints[:-1])): #dont integrate the last one obviously
        # tp counts down from last
        integration_times = torch.tensor([itp, itp + args.time_scale])
        integration_times = integration_times.type(torch.float32).to(device)
        # integration_times.requires_grad = True

        # load data and add noise
        if i != args.leaveout_timepoint:
            idx = args.data.sample_index(n="all", label_subset=tp) #used to be n=args.batch_size, want to use all data points 
            x = args.data.get_data()[idx]
            x = torch.from_numpy(x).type(torch.float32).to(device)
        else:
            x = z
        if args.training_noise > 0.0:
            x += np.random.randn(*x.shape) * args.training_noise

        # transform to next timepoint
        z, = model(x, integration_times=integration_times) #add comma cuz unpacking tuple
        if args.timepoints[i+1] != args.leaveout_timepoint:
            ground_truth_ix = args.data.sample_index(n="all", label_subset=args.timepoints[i+1])
            ground_truth = args.data.get_data()[ground_truth_ix]
            gt = torch.from_numpy(ground_truth).type(torch.float32).to(device)
            loss += train_loss_fn(z, gt)
    
    if len(regularization_coeffs) > 0:
        # Only regularize on the last timepoint
        reg_states_f = get_regularization(model, regularization_coeffs)
        reg_loss_f = sum(
            reg_state * coeff
            for reg_state, coeff in zip(reg_states_f, regularization_coeffs)
            if coeff != 0
        )

    #BACKWARD
    z = None
    for i, (itp, tp) in enumerate(zip(args.int_tps[::-1][:-1], args.timepoints[::-1][:-1])): #dont integrate the last one obviously
        # tp counts down from last
        integration_times = torch.tensor([itp - args.time_scale, itp])
        integration_times = integration_times.type(torch.float32).to(device)

        # load data and add noise
        if i != args.leaveout_timepoint:
            idx = args.data.sample_index(n="all", label_subset=tp) #used to be n=args.batch_size, want to use all data points 
            x = args.data.get_data()[idx]
            x = torch.from_numpy(x).type(torch.float32).to(device)
        else:
            x = z
        if args.training_noise > 0.0:
            x += np.random.randn(*x.shape) * args.training_noise

        # transform to next timepoint
        next_tp_index_reverse = len(args.timepoints)-i-2
        z, = model(x, integration_times=integration_times, reverse=True) #add comma cuz unpacking tuple
        if args.timepoints[next_tp_index_reverse] != args.leaveout_timepoint:
            ground_truth_ix = args.data.sample_index(n="all", label_subset=args.timepoints[next_tp_index_reverse])
            ground_truth = args.data.get_data()[ground_truth_ix]
            gt = torch.from_numpy(ground_truth).type(torch.float32).to(device)
            loss += train_loss_fn(z, gt)
    
    if len(regularization_coeffs) > 0:
        # Only regularize on the last timepoint
        reg_states_b = get_regularization(model, regularization_coeffs)
        reg_loss_b = sum(
            reg_state * coeff
            for reg_state, coeff in zip(reg_states_b, regularization_coeffs)
            if coeff != 0
        )

    if len(regularization_coeffs) > 0:
        reg_loss = reg_loss_f + reg_loss_b
        reg_states = tuple([x+y for x,y in zip(reg_states_f, reg_states_b)])
    else:
        reg_loss = 0
        reg_states = tuple()
    return loss, reg_loss, reg_states


def train(
    device, args, model, regularization_coeffs, regularization_fns, logger
):
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)

    full_data = (
        torch.from_numpy(
            args.data.get_data()[args.data.get_times() != args.leaveout_timepoint]
        )
        .type(torch.float32)
        .to(device)
    )

    best_loss = float("inf")
    

    end = time.time()
    train_loss_fn = SamplesLoss("sinkhorn", p=2, blur=1.0, backend="online")
    for itr in range(1, args.niters + 1):
        model.train()
        optimizer.zero_grad()

        #step the input mapping...take care to account for the regularizedODE wrapper
        odefunc = model.chain[0].odefunc
        if len(regularization_coeffs) > 0:
            odefunc = odefunc.odefunc
        odefunc.diffeq.feature_mapping.step(itr / args.niters)

        # Train
        if args.spectral_norm:
            spectral_norm_power_iteration(model, 1)

        loss, reg_loss, reg_states = compute_loss(device, args, model, logger, full_data, train_loss_fn, regularization_coeffs)
        loss_meter.update(loss.item())

        nfe_forward = count_nfe(model)

        loss = loss + reg_loss
        loss.backward()
        optimizer.step()

        # Eval
        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)
        time_meter.update(time.time() - end)

        log_message = (
            "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) |"
            " NFE Forward {:.0f}({:.1f})"
            " | NFE Backward {:.0f}({:.1f})".format(
                itr,
                time_meter.val,
                time_meter.avg,
                loss_meter.val,
                loss_meter.avg,
                nfef_meter.val,
                nfef_meter.avg,
                nfeb_meter.val,
                nfeb_meter.avg,
            )
        )
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(
                log_message, regularization_fns, reg_states
            )
        
        logger.info(log_message)

        if itr % args.val_freq == 0 or itr == args.niters:
            with torch.no_grad():
                train_eval(
                    device, args, model, itr, best_loss, logger, full_data, train_loss_fn
                )

        if itr % args.viz_freq == 0:
            with torch.no_grad():
                    visualize(device, args, model, itr)

        if itr % args.save_freq == 0:
            chkpt = {
                "state_dict": model.state_dict(),
            }

            utils.save_checkpoint(
                chkpt,
                args.save,
                epoch=itr,
            )
        end = time.time()
    logger.info("Training has finished.")


def train_eval(device, args, model, itr, best_loss, logger, full_data, train_loss_fn):
    model.eval()
    test_loss = compute_loss(device, args, model, logger, full_data, train_loss_fn)
    emd_backward, emd_forward = evaluate_kantorovich_v2(device, args, model)
    test_nfe = count_nfe(model)
    log_message = "[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f} | EMD F/B {:.4f}/{:.4f}".format(
        itr, test_loss.item(), test_nfe, emd_forward, emd_backward
    )
    logger.info(log_message)
    utils.makedirs(args.save)
    with open(os.path.join(args.save, "train_eval.csv"), "a") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow((itr, test_loss))

    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        chkpt = {
            "state_dict": model.state_dict(),
        }
        torch.save(
            chkpt,
            os.path.join(args.save, "checkpt.pth"),
        )


def visualize(device, args, model, itr):
    model.eval()
    plt.clf()
    plt.figure(figsize=(9, 3))
    LOW = -4
    HIGH = 4
    npts = 100
    nrows = 2
    ncols = len(args.timepoints)
    d = {}
    #plot ground truth first
    for i, tp in enumerate(args.timepoints):
        ax = plt.subplot(nrows, ncols, i+1, aspect="equal")
        idx = args.data.sample_index(n="all", label_subset=tp)
        gt_samples = args.data.get_data()[idx]
        gt_samples = torch.from_numpy(gt_samples).type(torch.float32).to(device)
        d[i] = gt_samples
        ax.hist2d(gt_samples[:, 0].cpu().numpy(), gt_samples[:, 1].cpu().numpy(), range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
        ax.invert_yaxis()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.set_title(f"t={i}")
        if i == 0:
            ax.set_ylabel(f"Ground Truth")
    #plot predicted
    for i, (itp, tp) in enumerate(zip(args.int_tps, args.timepoints)):
        ax = plt.subplot(nrows, ncols, ncols+i+1, aspect="equal")
        if i == 0:
            gt_samples = d[i]
            ax.hist2d(gt_samples[:, 0].cpu().numpy(), gt_samples[:, 1].cpu().numpy(), range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
            ax.invert_yaxis()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_ylabel(f"Advected Samples")
        else:
            gt_samples = d[i-1]
            integration_times = torch.tensor([itp - args.time_scale, itp])
            integration_times = integration_times.type(torch.float32).to(device)
            advected_samples, = model(gt_samples, integration_times=integration_times) #add comma cuz unpacking tuple
            ax.hist2d(advected_samples[:, 0].cpu().numpy(), advected_samples[:, 1].cpu().numpy(), range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
            ax.invert_yaxis()
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    fig_filename = os.path.join(
            args.save, "figs", "{:04d}.jpg".format(itr, i)
        )
    utils.makedirs(os.path.dirname(fig_filename))
    plt.savefig(fig_filename)
    plt.close()

def plot_output(device, args, model):
    save_traj_dir = os.path.join(args.save, "trajectory")
    # logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    data_samples = args.data.get_data()[args.data.sample_index(2000, 0)]
    np.random.seed(42)
    start_points = args.data.base_sample()(1000, 2)
    # idx = args.data.sample_index(50, 0)
    # start_points = args.data.get_data()[idx]
    # start_points = torch.from_numpy(start_points).type(torch.float32)
    save_vectors(
        args.data.base_density(),
        model,
        start_points,
        args.data.get_data(),
        args.data.get_times(),
        args.save,
        skip_first=(not args.data.known_base_density()),
        device=device,
        end_times=args.int_tps,
        ntimes=100,
    )
    save_trajectory(
        args.data.base_density(),
        args.data.base_sample(),
        model,
        data_samples,
        save_traj_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=25,
    )
    trajectory_to_video(save_traj_dir)

    density_dir = os.path.join(args.save, "density2")
    save_trajectory_density(
        args.data.base_density(),
        model,
        data_samples,
        density_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=25,
        memory=0.1,
    )
    trajectory_to_video(density_dir)


def main(args):
    # logger
    print(args.no_display_loss)
    utils.makedirs(args.save)

    logger = utils.get_logger(
        logpath=os.path.join(args.save, "logs"),
        filepath=os.path.abspath(__file__),
        displaying=~args.no_display_loss,
    )

    logger.info(args)

    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    if args.use_cpu:
        device = torch.device("cpu")

    args.data = dataset.SCData.factory(args.dataset, args)

    args.timepoints = args.data.get_unique_times()
    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_vanilla(args, args.data.get_shape()[0], regularization_fns).to(
        device
    )

    if args.spectral_norm:
        add_spectral_norm(model)

    logger.info(model)
    n_param = count_parameters(model)
    logger.info("Number of trainable parameters: {}".format(n_param))

    train(
        device,
        args,
        model,
        regularization_coeffs,
        regularization_fns,
        logger,
    )

    if args.data.data.shape[1] == 2:
        plot_output(device, args, model)


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)
