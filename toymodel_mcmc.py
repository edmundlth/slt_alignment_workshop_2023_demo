import jax.numpy as jnp
from jax import random
import jax
import numpyro
import numpyro.distributions as dist

import numpy as np
import scipy
import torch
from scipy.spatial.transform import Rotation

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn_extra.cluster import KMedoids
from sklearn.manifold import TSNE

import argparse
import functools
import gc
import os
import time
import json
import logging
logger = logging.getLogger("__main__")  
logging.basicConfig(level=logging.INFO)  # Lower the severity level to INFO

from mcmc_utils import (
    MCMCConfig, 
    linspaced_itemps_by_n, 
    run_mcmc, 
    rlct_estimate_regression, 
    plot_polygon, 
    plot_rlct_regression, 
    count_convhull_vertices, 
    make_tally, 
    generate_2d_ngon_vertices, 
    compute_bayesian_loss, 
    compute_functional_variance, 
    compute_gibbs_loss, 
    compute_waic, 
    compute_wbic, 
)

class ToyModelsDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, n, gamma):
        self.num_samples = num_samples
        self.n = n
        self.data = []
        self.targets = []

        for _ in range(num_samples):
            a = torch.randint(1, self.n + 1, (1,)).item()
            lambda_val = torch.rand(1).item()

            # Create input vector x
            x = torch.zeros(self.n)
            x[a - 1] = lambda_val

            gaussian_noise = torch.empty_like(x)
            gaussian_noise.normal_(mean=0,std=1/np.sqrt(gamma))
            y = x + gaussian_noise

            self.data.append(x)
            self.targets.append(y)
        self.data = torch.stack(self.data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    

def normalize_W_3D(W):
    # Compute the magnitudes of the columns
    magnitudes = np.linalg.norm(W, axis=0)

    # Identify the index of the column with the largest magnitude
    idx_max_magnitude = np.argmax(magnitudes)

    # Compute the azimuth and elevation of the largest column
    azimuth = np.arctan2(W[1, idx_max_magnitude], W[0, idx_max_magnitude])
    elevation = np.arcsin(W[2, idx_max_magnitude]/magnitudes[idx_max_magnitude])

    # Create a rotation matrix
    rotation = Rotation.from_euler('zyx', [-azimuth, -elevation, 0])
    R = rotation.as_matrix()

    # Rotate the matrix
    W_rotated = np.dot(R, W)

    # Normalize the rotated matrix
    W_normalised = W_rotated / magnitudes

    # Compute the azimuth and elevation of each column
    azimuths = np.arctan2(W_normalised[1], W_normalised[0])
    elevations = np.arcsin(W_normalised[2])

    # Create a structured array for sorting
    angles = np.zeros(W.shape[1], dtype=[('azimuth', float), ('elevation', float)])
    angles['azimuth'] = azimuths
    angles['elevation'] = elevations

    # Sort the columns according to their angles
    return W_rotated[:, np.argsort(angles, order=['azimuth', 'elevation'])]


def normalize_W(W):
    # Compute the magnitudes of the columns
    magnitudes = np.linalg.norm(W, axis=0)

    # Identify the index of the column with the largest magnitude
    idx_max_magnitude = np.argmax(magnitudes)

    # Compute the angle between the largest column and the x-axis
    angle = np.arctan2(W[1, idx_max_magnitude], W[0, idx_max_magnitude])

    # Create a rotation matrix (clockwise rotation)
    R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

    # Rotate the matrix
    W_rotated = np.dot(R, W)

    # Sort the columns according to their angles from the x-axis
    W_normalised = W_rotated / magnitudes
    angles = np.arctan2(W_normalised[1], W_normalised[0])

    return W_rotated[:, np.argsort(angles)]

def compute_l2_distance(W1, W2, m=2, normalise_func=normalize_W):
    # Normalize the matrices
    W1_normalized = normalise_func(W1.reshape(m, -1))
    W2_normalized = normalise_func(W2.reshape(m, -1))
    # Compute the L2 distance between the normalized matrices
    return np.linalg.norm(W1_normalized - W2_normalized)

@jax.jit
def toymodel_forward(W, b, X):
    # Compute ReLU(W^T W x + b), where x is now a row of X
    mu = jnp.matmul(jnp.transpose(W), jnp.matmul(W, jnp.transpose(X))) + b
    mu = jax.nn.relu(mu)
    return jnp.transpose(mu)

@jax.jit
def toymodel_log_likelihood(W, b, X, sigma_obs=1.0):
    mu = toymodel_forward(W, b, X)
    return dist.Normal(mu, sigma_obs).log_prob(X)


@jax.jit
def toymodel_forward_no_bias(W, X):
    # Compute ReLU(W^T W x + b), where x is now a row of X
    mu = jnp.matmul(jnp.transpose(W), jnp.matmul(W, jnp.transpose(X)))
    mu = jax.nn.relu(mu)
    return jnp.transpose(mu)

@jax.jit
def toymodel_log_likelihood_no_bias(W, X, sigma_obs=1.0):
    mu = toymodel_forward_no_bias(W, X)
    return dist.Normal(mu, sigma_obs).log_prob(X)



def build_model(m, n, prior_w_center=None, prior_b_center=None, prior_std=1.0, sigma_obs=1.0, no_bias=False):
    if prior_w_center is None:
        prior_w_center = jnp.zeros((m, n))
    if prior_b_center is None:
        prior_b_center = jnp.zeros((n, 1))

    def model_with_bias(X, itemp=1.0):
        # Prior over W and b
        W = numpyro.sample(
            "W", dist.Normal(prior_w_center, prior_std * jnp.ones((m, n)))
        )
        b = numpyro.sample(
            "b", dist.Normal(prior_b_center, prior_std * jnp.ones((n, 1)))
        )
        mu = toymodel_forward(W, b, X)
        numpyro.sample("X", dist.Normal(mu, sigma_obs / jnp.sqrt(itemp)), obs=X)
        return
    
    def model_without_bias(X, itemp=1.0):
        # Prior over W
        W = numpyro.sample(
            "W", dist.Normal(prior_w_center, prior_std * jnp.ones((m, n)))
        )
        mu = toymodel_forward_no_bias(W, X)
        numpyro.sample("X", dist.Normal(mu, sigma_obs / jnp.sqrt(itemp)), obs=X)
        return
    
    if no_bias:
        return model_without_bias
    else:
        return model_with_bias


def main(args, rngseed):
    logger.info(f"Running experiment for rngseed: {rngseed}")
    # set rngseeds
    np.random.seed(rngseed)
    torch.manual_seed(rngseed)
    rng_key = random.PRNGKey(rngseed)

    # parse cmd arguments
    m = args.m
    num_training_data = args.num_training_data
    n = args.n
    
    ####################################################
    # Generate data
    ####################################################
    X = jnp.array(ToyModelsDataset(num_samples=num_training_data, n=n, gamma=100).data)
    X_test = jnp.array(ToyModelsDataset(num_samples=args.num_test_data, n=n, gamma=100).data)

    ####################################################
    # Run MCMC
    ####################################################
    mcmc_config = MCMCConfig(
        args.num_posterior_samples, args.num_warmup, args.num_chains, args.thinning
    )
    if args.prior_center_k is not None:
        prior_center = generate_2d_ngon_vertices(args.prior_center_k, rot=np.pi / 4, pad_to=n)
    else:
        prior_center = None
    model = build_model(m, n, prior_w_center=prior_center, prior_std=args.prior_std, sigma_obs=args.sigma_obs, no_bias=args.no_bias)
    mcmc = run_mcmc(model, X, rng_key, mcmc_config)
    posterior_samples = mcmc.get_samples()
    

    ####################################################
    # Collect results from MCMC samples
    ####################################################
    Ws = posterior_samples["W"]
    num_mcmc_samples = len(Ws)
    if "b" in posterior_samples:
        bs = posterior_samples["b"] # TODO: handle cases where --no_bias is set. 
    else:
        bs = np.zeros((num_mcmc_samples, n, 1))
    # - Convex hull configurations
    convhull_vertices_nums = [count_convhull_vertices(W) for W in Ws]
    convhull_vertices_tally = make_tally(convhull_vertices_nums)
    
    logger.info(f"Num MCMC samples: {num_mcmc_samples}")
    num_unique_convexhull = len(set(convhull_vertices_nums))
    
    logger.info(f"Distinct convex hulls (num={num_unique_convexhull}): {sorted(set(convhull_vertices_nums))}")

    # - Frobenius norms
    frobenius_norms = np.array([np.sum(W**2) for W in Ws])

    # - Clustering
    vecs = np.array([W.flatten() for W in Ws])
    
    # - k-Medoid clustering
    if args.num_cluster is not None:
        n_cluster = args.num_cluster
    else:
        n_cluster = num_unique_convexhull
    logger.info(f"Running k={n_cluster}-Medoid clustering...")
    if m == 2:
        normalise_func = normalize_W
    elif m == 3:
        normalise_func = normalize_W_3D
    else:
        raise NotImplementedError("Program not implemented for m > 3.")
    metric = lambda W1, W2: compute_l2_distance(W1, W2, m=m, normalise_func=normalise_func)
    kmedoids = KMedoids(n_clusters=n_cluster, metric=metric).fit(vecs)
    cluster_centers = kmedoids.cluster_centers_
    

    # t-SNE for visualization
    logger.info("Running tSNE...")
    tsne = TSNE(n_components=2, metric=metric)
    vecs_2d = tsne.fit_transform(vecs)

    # Bayes obervables
    if args.no_bias: # TODO: handle this no-bias ugly code...
        log_likelihood_fn = jax.jit(
            lambda W, bs, X: toymodel_log_likelihood_no_bias(W, X, sigma_obs=args.sigma_obs)
        )
    else:
        log_likelihood_fn = jax.jit(functools.partial(
                toymodel_log_likelihood, sigma_obs=args.sigma_obs
            ))
            
    
    loglike_array = np.hstack(# dimension = (num test samples, num mcmc samples)
        [log_likelihood_fn(Ws[i], bs[i], X_test) for i in range(len(Ws))]
    )
    gibbs_loss = compute_gibbs_loss(loglike_array)
    logger.info(f"Gibbs loss: {gibbs_loss}")

    bayes_loss = compute_bayesian_loss(loglike_array)
    logger.info(f"Bayes loss: {bayes_loss}")
    
    # This is loglikelihood array for training data now. 
    # It overwrites the previous loglike_array to free up memory.
    del loglike_array
    gc.collect()

    loglike_array = np.hstack( 
        [log_likelihood_fn(Ws[i], bs[i], X) for i in range(len(Ws))]
    )
    
    gibbs_train_loss = compute_gibbs_loss(loglike_array)
    logger.info(f"Gibbs train loss: {gibbs_train_loss}")

    bayes_train_loss = compute_bayesian_loss(loglike_array)
    logger.info(f"Bayes train loss: {bayes_train_loss}")

    func_var = compute_functional_variance(loglike_array)
    logger.info(f"Functional Variance: {func_var}")

    waic = compute_waic(loglike_array)
    logger.info(f"WAIC: {waic}")

    # clean up memory intensive variable.
    del loglike_array
    gc.collect()

    # truth_entropy = -np.mean(
    #     toymodel_log_likelihood(
    #         forward_true.apply, true_param, X_test, Y_test, sigma=args.sigma_obs
    #     )
    # )
    # truth_entropy_train = -np.mean(
    #     log_likelihood(
    #         forward_true.apply, true_param, X, Y, sigma=args.sigma_obs
    #     )
    # )
    result = {
        "commandline_args": vars(args), 
        "rng_seed": rngseed,
        "n": n, 
        "m": m, 
        "rngseed": rngseed, 
        "num_unique_convexhull_config": num_unique_convexhull, 
        "convexhull_config_tally": convhull_vertices_tally, 
        "cluster_centers": cluster_centers.tolist(),
        "BL": float(bayes_loss),
        "BLt": float(bayes_train_loss),
        "GL": float(gibbs_loss),
        "GLt": float(gibbs_train_loss),
        "V_n": float(func_var), 
        "WAIC": float(waic),
    }
    logger.info(f"Result so far: \n{json.dumps(result, indent=2)}")

    ####################################################
    # Run RLCT estimation
    ####################################################
    do_rlct_estimation = (args.num_itemps is not None)
    if do_rlct_estimation:
        itemps = linspaced_itemps_by_n(
            args.num_training_data, num_itemps=args.num_itemps
        )
        enlls, stds = rlct_estimate_regression(
            itemps,
            rng_key,
            model,
            jax.jit(lambda W, b, X: log_likelihood_fn(W, b, X).sum()),
            X,
            mcmc_config,
            progress_bar=True
        )
        slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps, enlls)
        _map_float = lambda lst: list(
            map(float, lst)
        )  # just to make sure things are json serialisable.
        result["rlct_estimation"] = {
            "num_training_data": num_training_data,
            "itemps": _map_float(itemps),
            "enlls": _map_float(enlls),
            "chain_stds": _map_float(stds),
            "slope": float(slope),
            "intercept": float(intercept),
            "r^2": float(r_val**2),
        }

    logger.info(json.dumps(result, indent=2))


    ####################################################
    # Plots
    ####################################################
    logger.info(f"Plotting...")
    font = {"size": 10}
    matplotlib.rc("font", **font)

    max_polygon_plot_cols = max(5, num_unique_convexhull)
    nrows_cluster_plots = (n_cluster // max_polygon_plot_cols) + (n_cluster % max_polygon_plot_cols > 0)
    ncols_cluster_plots = min(max_polygon_plot_cols, n_cluster)

    nrows = 4 # for the top of the plots
    nrows += 2 * nrows_cluster_plots
    nrows += 2 # for convhull plots
    num_div = 3 if do_rlct_estimation else 2
    ncols = 2 * max(ncols_cluster_plots, num_div)
    logger.debug(f"Figure grid nrows:{nrows}, ncols:{ncols}.")
    gs = GridSpec(nrows, ncols)
    fig = plt.figure(figsize=(ncols * 3, nrows * 3))

    
    ax = fig.add_subplot(gs[0:2, : ncols // num_div])
    ax.hist(2 / frobenius_norms)
    ax.set_xlabel("$m / ||W||^2$")
    ax.set_title("Distribution of $m / ||W||^2$")
    
    ax = fig.add_subplot(gs[0:2, ncols // num_div:  2 * ncols // num_div])
    keys = sorted(convhull_vertices_tally.keys())
    vals = np.array([convhull_vertices_tally[k] for k in keys])
    assert len(convhull_vertices_nums) == len(Ws) == sum(vals)
    ax.bar(keys, vals / sum(vals))
    ax.set_xticks(keys)
    ax.set_xlabel("num convhull vertices")
    ax.set_title("Frequency of number of vertices in convex hull")

    if do_rlct_estimation:
        # add the RLCT regression to the figure
        ax = fig.add_subplot(gs[0:2:, 2 * ncols // num_div:])
        plot_rlct_regression(itemps, enlls, num_training_data, ax)

    # Plot the clusters
    ax = fig.add_subplot(gs[2:4, : ncols // 2])
    cmap = None
    p = ax.scatter(
        vecs_2d[:, 0], vecs_2d[:, 1], c=kmedoids.labels_, alpha=0.8, linewidth=0.0, cmap=cmap
    )
    fig.colorbar(p, ax=ax)
    ax.set_title(f"tSNE custom metric (color by {n_cluster}-Medoid clusters)")

    ax = fig.add_subplot(gs[2:4, ncols // 2 :])
    # color by number of convexhull vertices
    p = ax.scatter(
        vecs_2d[:, 0],
        vecs_2d[:, 1],
        c=convhull_vertices_nums,
        alpha=0.8,
        linewidth=0.0,
        cmap=cmap,
    )
    fig.colorbar(p, ax=ax)
    ax.set_title(f"tSNE custom metric (color by num vertices in convex hull)")

    share_ax = None
    c = 0
    for i in range(nrows_cluster_plots):
        for j in range(ncols_cluster_plots):
            if c >= len(cluster_centers):
                break
            W = cluster_centers[c].reshape(m, -1)
            c += 1
            ax = fig.add_subplot(gs[4 + 2 * i: 4 + 2 * i + 2, 2 * j : 2 * j + 2], projection=None if m == 2 else "3d")
            plot_polygon(normalise_func(W), ax)
            ax.set_title(f"{n_cluster}-Medoid center (k={count_convhull_vertices(W)})")
            if share_ax is None:
                share_ax = ax
            ax.sharex(share_ax)
            ax.sharey(share_ax)

    for i in range(num_unique_convexhull):
        num_vertex = sorted(set(convhull_vertices_nums))[i]
        idx = list(convhull_vertices_nums).index(num_vertex)
        W = Ws[idx]
        ax = fig.add_subplot(gs[nrows -2:, 2 * i : 2 * i + 2], projection=None if m == 2 else "3d")
        plot_polygon(W, ax, hull_alpha=1.0)
        ax.set_title(f"hull vertices={num_vertex}")
        ax.sharex(share_ax)
        ax.sharey(share_ax)
    # rescale all the polygon plots and set their aspect ratio to be "equal"
    share_ax.autoscale() 
    share_ax.set_aspect(aspect="equal")
    
    suptitle = (
        f"m={m}, n={n}, "
        f"num_train={num_training_data}, "
        f"num_convhull_config={num_unique_convexhull}, "
        f"num_mcmc={num_mcmc_samples}, "
        f"bias={not args.no_bias}, "
        f"prior_k={args.prior_center_k}, "
        f"prior_std={args.prior_std}, "
        f"rngseed={rngseed}"
    )
    fig.suptitle(suptitle, fontsize="x-large", fontweight="bold")
    fig.tight_layout(pad=3.0, w_pad=2.0, h_pad=2.0)

    if args.show_plot:
        plt.show()
    
    bias_string = "_nobias_" if args.no_bias else "_"
    prior_string = f"pstd{args.prior_std}" if prior_center is None else f"pmu{args.prior_center_k}_pstd{args.prior_std}"
    outfileprefix = f"toymcmc{bias_string}m{m}_n{n}_numtrain{num_training_data}_nummcmc{num_mcmc_samples}_ncluster{n_cluster}_{prior_string}_seed{rngseed}"

    if (args.outputdir is not None) and (not args.no_save_plot):
        outfilepath = os.path.join(args.outputdir, f"{outfileprefix}.png")
        logger.info(f"Saving plot image at: {outfilepath}")
        fig.savefig(outfilepath, bbox_inches="tight")

    if (args.outputdir is not None) and (not args.no_save_result):
        outfilepath = os.path.join(args.outputdir, f"{outfileprefix}.json")
        logger.info(f"Saving result JSON at: {outfilepath}")
        with open(outfilepath, "w") as outfile:
            json.dump(result, outfile, indent=2)
        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running MCMC on toy model of superposition.")
    parser.add_argument("--m", help="Number of hidden dimensions", type=int, default=2)
    parser.add_argument("--n", help="Number of input dimensions", type=int, default=5)
    parser.add_argument("--num_training_data", help="Training data set size.", type=int, default=100)
    parser.add_argument("--num_test_data", nargs="?", default=5000, type=int, help="Number of test data.")
    parser.add_argument("--num_posterior_samples", nargs="?", default=1000, type=int)
    parser.add_argument("--thinning", nargs="?", default=4, type=int)
    parser.add_argument("--num_warmup", nargs="?", default=500, type=int)
    parser.add_argument("--num_chains", nargs="?", default=4, type=int)
    parser.add_argument("--sigma_obs", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior_std", nargs="?", default=1.0, type=float)
    parser.add_argument("--prior_center_k", nargs="?", default=None, type=int, help="Chosing a k-gon to set as the center of the gaussian prior distribution. If not specified, prior is centered at the origin.")
    
    parser.add_argument("--host_device_count", nargs="?", default=4, type=int)
    parser.add_argument(
        "--no_bias",
        action="store_true",
        help="If set, use bias parameters.",
    )
    parser.add_argument(
        "--num_itemps",
        nargs="?",
        default=None,
        type=int,
        help="If this is specified, the RLCT estimation procedure will be run.",
    )
    parser.add_argument(
        "--num_cluster",
        nargs="?",
        default=None,
        type=int,
        help="Number of clusters used in k-Medoid clustering. If not specified, use the number of unique hull config detected in posterior samples.",
    )
    
    parser.add_argument(
        "--seeds",
        help="Experiments are repeated for each RNG seed given",
        nargs="+",
        type=int,
        default=[0],
    )
    parser.add_argument(
        "--outputdir",
        help="Path to output directory. If not specified, nothing is saved. Create if not exist.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--no_save_result",
        help="don't save experiment result if specified.",
        action="store_true",
    )
    parser.add_argument(
        "--no_save_plot", help="don't save plots to file if specified", action="store_true"
    )
    parser.add_argument(
        "--show_plot", help="if set, run plt.show()", action="store_true"
    )
    
    args = parser.parse_args()
    numpyro.set_host_device_count(args.host_device_count)

    if args.outputdir is not None and not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir, exist_ok=True)

    logger.info(f"Commandline arguments:\n {json.dumps(vars(args), indent=2)}")
    for rngseed in args.seeds:
        start_time = time.time()
        main(args, rngseed)
        logger.info(f"Finished rng_seed={rngseed}. Time taken: {time.time() - start_time:.1f}s")
