import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
from scipy.spatial import ConvexHull
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from collections import namedtuple
from scipy.special import logsumexp
import logging
logger = logging.getLogger("__main__")  
logging.basicConfig(level=logging.INFO)  # Lower the severity level to INFO


MCMCConfig = namedtuple(
    "MCMCConfig", ["num_posterior_samples", "num_warmup", "num_chains", "thinning"]
)


def zero_small_columns(W, threshold=1e-2):
    """
    Replace columns in W with zero if their magnitude is below a certain threshold.
    """
    col_magnitudes = np.linalg.norm(W, axis=0)  
    small_cols = col_magnitudes < threshold  
    W[:, small_cols] = 0
    return W



def count_convhull_vertices(W):
    hull = ConvexHull(W.T)
    return len(hull.vertices)

def linspaced_itemps_by_n(n, num_itemps):
    """
    Returns a 1D numpy array of length `num_itemps` that contains `num_itemps`
    evenly spaced inverse temperatures between the values calculated from the
    formula 1/log(n) * (1 - 1/sqrt(2log(n))) and 1/log(n) * (1 + 1/sqrt(2ln(n))).
    The formula is used in the context of simulating the behavior of a physical
    system at different temperatures using the Metropolis-Hastings algorithm.
    """
    return np.linspace(
        1 / np.log(n) * (1 - 1 / np.sqrt(2 * np.log(n))),
        1 / np.log(n) * (1 + 1 / np.sqrt(2 * np.log(n))),
        num_itemps,
    )


def expected_nll(log_likelihood_fn, Ws, bs, X):
    nlls = []
    for i in range(len(Ws)):
        nlls.append(-log_likelihood_fn(Ws[i], bs[i], X))
    return np.mean(nlls)

def chain_wise_enlls(mcmc, log_likelihood_fn, X):
    posterior_samples = mcmc.get_samples(group_by_chain=True)
    num_chains, num_mcmc_samples_per_chain = posterior_samples[
        list(posterior_samples.keys())[0]
    ].shape[:2]
    num_mcmc_samples = num_chains * num_mcmc_samples_per_chain
    logger.info(f"Total number of MCMC samples: {num_mcmc_samples}")
    logger.info(f"Number of MCMC chain: {num_chains}")
    logger.info(f"Number of MCMC samples per chain: {num_mcmc_samples_per_chain}")
    chain_enlls = []
    chain_sizes = []
    for chain_index in range(num_chains):
        Ws = posterior_samples["W"][chain_index]
        if "b" in posterior_samples:
            bs = posterior_samples["b"][chain_index]
        else:
            bs = np.zeros((Ws.shape[0], 1))
        chain_enll = expected_nll(log_likelihood_fn, Ws, bs, X)
        chain_size = len(Ws)
        chain_enlls.append(chain_enll)
        chain_sizes.append(chain_size)
        logger.info(
            f"Chain {chain_index} with size {chain_size} has enll {chain_enll}."
        )
    return chain_enlls, chain_sizes

def run_mcmc(model, data, rngkey, mcmc_config, init_params=None, itemp=1.0, progress_bar=True):
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel, 
        num_warmup=mcmc_config.num_warmup, 
        num_samples=mcmc_config.num_posterior_samples, 
        thinning=mcmc_config.thinning, 
        num_chains=mcmc_config.num_chains, 
        progress_bar=progress_bar
    )
    logger.info("Running MCMC")
    mcmc.run(rngkey, data, itemp=itemp, init_params=init_params)
    return mcmc

def rlct_estimate_regression(
    itemps,
    rng_key,
    model,
    log_likelihood_fn,
    X,
    mcmc_config: MCMCConfig,
    progress_bar=True,
):
    logger.info("Running RLCT estimation regression")
    logger.info(f"Sequence of itemps: {itemps}")
    n = len(X)
    enlls = []
    stds = []
    rngkeys = jax.random.split(rng_key, num=len(itemps))
    for i_itemp, itemp in enumerate(itemps):
        mcmc = run_mcmc(
            model,
            X,
            rngkeys[i_itemp],
            mcmc_config,
            itemp=itemp,
            progress_bar=progress_bar,
        )
        chain_enlls, chain_sizes = chain_wise_enlls(
            mcmc, log_likelihood_fn, X,
        )
        enll = np.sum(np.array(chain_enlls) * np.array(chain_sizes)) / np.sum(
            chain_sizes
        )
        chain_enlls_std = np.std(chain_enlls)
        logger.info(f"Finished {i_itemp} temp={1/itemp:.3f}. Expected NLL={enll:.3f}")
        logger.info(f"Across chain enll std: {chain_enlls_std}.")
        enlls.append(enll)
        stds.append(chain_enlls_std)
        if len(enlls) > 1:
            slope, intercept, r_val, _, _ = scipy.stats.linregress(
                1 / itemps[: len(enlls)], enlls
            )
            logger.info(
                f"est. RLCT={slope:.3f}, energy={intercept / n:.3f}, r2={r_val**2:.3f}"
            )
    return enlls, stds

@jax.jit
def frobenius_norm(W):
    return jnp.linalg.norm(W, ord="fro")


def plot_polygon(W, ax=None, hull_alpha=0.3):
    if ax is None:
        if W.shape[0] == 2: 
            fig, ax = plt.subplots(1, 1)
        elif W.shape[0] == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

    if W.shape[0] == 2: # 2D case
        # Compute the norms of the columns
        norms = np.linalg.norm(W, axis=0)

        # Normalize a copy of the vectors for angle calculations
        W_normalized = W / norms

        # Compute angles from the x-axis for each vector
        angles = np.arctan2(W_normalized[1, :], W_normalized[0, :])

        # Sort the columns of W by angles
        W_sorted = W[:, np.argsort(angles)]

        # Plot the origin
        ax.scatter(0, 0, color="red")

        # Plot the vectors
        for i in range(W_sorted.shape[1]):
            ax.quiver(
                0, 0, W_sorted[0, i], W_sorted[1, i], angles="xy", scale_units="xy", scale=1
            )

        # Connect the vectors to form a polygon
        polygon = np.column_stack((W_sorted, W_sorted[:, 0]))
        ax.plot(polygon[0, :], polygon[1, :], alpha=0.5)

        # Plot the convex hull
        hull = ConvexHull(W.T)
        vs = list(hull.vertices) + [hull.vertices[0]]
        ax.plot(W[0, vs], W[1, vs], "r--", alpha=hull_alpha)

        # Set the aspect ratio of the plot to equal to ensure that angles are displayed correctly
        ax.set_aspect("equal", adjustable="box")
        return ax
    
    elif W.shape[0] == 3:  # 3D case
        # Plot the origin
        ax.scatter([0], [0], [0], color="red")
        
        # Plot the vectors
        for i in range(W.shape[1]):
            ax.plot([0, W[0, i]], [0, W[1, i]], [0, W[2, i]], 'b-')

        # Plot the convex hull
        hull = ConvexHull(W.T)
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(W[0, s], W[1, s], W[2, s], "r--", alpha=hull_alpha)
        return ax

    else:
        raise ValueError("W must have either 2 or 3 rows")


def plot_rlct_regression(itemps, enlls, n, ax):
    slope, intercept, r_val, _, _ = scipy.stats.linregress(1 / itemps, enlls)
    ax.plot(1 / itemps, enlls, "kx")
    ax.plot(
        1 / itemps,
        1 / itemps * slope + intercept,
        label=f"$\lambda$={slope:.3f}, $nL_n(w_0)$={intercept:.3f}, $R^2$={r_val**2:.2f}",
    )
    ax.legend()
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Expected NLL")
    ax.set_title(f"n={n}, L_n={intercept / n:.3f}")
    return ax


def make_tally(lst):
    tally = {}
    for elem in lst:
        if elem in tally:
            tally[elem] += 1
        else:
            tally[elem] = 1
    return tally

def generate_2d_ngon_vertices(n, rot=0, pad_to=None):
    # Angles for the vertices
    theta = np.linspace(0, 2*np.pi, n, endpoint=False) + rot

    # Generate the vertices
    x = np.cos(theta)
    y = np.sin(theta)
    result = np.vstack((x, y))
    if pad_to is not None and n < pad_to:
        num_pad = pad_to - n
        result = np.hstack([result, np.zeros((2, num_pad))])
    return result


# dimension of `loglike_array` in all the following functions are 
# (num data samples X, num mcmc sampled parameters w)
def compute_bayesian_loss(loglike_array):
    num_mcmc_samples = loglike_array.shape[1]
    result = -np.mean(logsumexp(loglike_array, b=1 / num_mcmc_samples, axis=1))
    return result


def compute_gibbs_loss(loglike_array):
    gerrs = np.mean(loglike_array, axis=0)
    gg = np.mean(gerrs)
    return -gg


def compute_functional_variance(loglike_array):
    # variance over posterior samples and averaged over dataset.
    # V = 1/n \sum_{i=1}^n Var_w(\log p(X_i | w))
    result = np.mean(np.var(loglike_array, axis=1))
    return result


def compute_waic(loglike_array):
    func_var = compute_functional_variance(loglike_array)
    bayes_train_loss = compute_bayesian_loss(loglike_array)
    return bayes_train_loss + func_var


def compute_wbic(tempered_loglike_array):
    return -np.mean(np.sum(tempered_loglike_array, axis=0))
