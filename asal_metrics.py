import jax
import jax.numpy as jnp

from einops import repeat

def calc_supervised_target_score(z, z_txt):
    """
    Calculates the supervisted target score from ASAL.
    The returned score should be minimized, since we add a minus sign here.

    Parameters
    ----------
    z : jnp.ndarray of shape (T, D) or (D,)
        The latent representation of the images over time.
    z_txt : jnp.ndarray of shape (T2, D)
        The latent representation of the text prompts over time.
    """
    # Handle 1D case (single frame): just compare with first prompt
    if z.ndim == 1:
        # Single frame: compute dot product with all prompts and take best match
        scores = z @ z_txt.T  # (T2,)
        return -scores.max()  # maximize best match = minimize negative best match
    
    T = z.shape[0]
    T2 = z_txt.shape[0]
    
    # Use GCD to find compatible repeat factor
    from math import gcd
    g = gcd(T, T2)
    k = T // g
    t2_repeat = T2 // g
    
    # Repeat prompts to match temporal dimension
    z_txt_repeated = repeat(z_txt, "T2 D -> (k T2) D", k=k)  # (T, D)
    
    kernel = z_txt_repeated @ z.T  # (T, T)
    return -jnp.diag(kernel).mean()

def calc_supervised_target_softmax_score(z, z_txt, temperature_softmax=0.01):
    """
    Calculates the supervisted target score from ASAL with softmax.
    This isn't part of the original ASAL, but it's a useful extension.
    This score helps incentivize the simulation to find unique images for each prompt rather than one static image satisfying all prompts.
    The returned score should be minimized.

    Parameters
    ----------
    z : jnp.ndarray of shape (T, D) or (D,)
        The latent representation of the images over time.
    z_txt : jnp.ndarray of shape (T2, D)
        The latent representation of the text prompts over time.
    temperature_softmax : float
        The temperature for the softmax function. For CLIP, leave it at 0.01, since that is default CLIP softmax temperature.
    """
    # Handle 1D case (single frame): return 0 since softmax loss requires temporal data
    if z.ndim == 1:
        return jnp.array(0.0)
    
    T = z.shape[0]
    T2 = z_txt.shape[0]
    
    # Use GCD to find compatible repeat factor
    from math import gcd
    g = gcd(T, T2)
    k = T // g
    t2_repeat = T2 // g
    
    # Repeat prompts to match temporal dimension
    z_txt_repeated = repeat(z_txt, "T2 D -> (k T2) D", k=k)  # (T, D)

    kernel = z_txt_repeated @ z.T  # (T, T)
    loss_sm1 = jax.nn.softmax(kernel/temperature_softmax, axis=-1)
    loss_sm2 = jax.nn.softmax(kernel/temperature_softmax, axis=-2)
    loss_sm1 = -jnp.log(jnp.diag(loss_sm1))
    loss_sm2 = -jnp.log(jnp.diag(loss_sm2))
    return (loss_sm1.mean() + loss_sm2.mean())/2.

def calc_open_endedness_score(z):
    """
    Calculates the open-endedness score from ASAL.
    The returned score should be minimized.

    Parameters
    ----------
    z : jnp.ndarray of shape (T, D) or (D,)
        The latent representation of the images over time.
    """
    # Handle 1D case (single frame): return 0 since open-endedness is meaningless for single frame
    if z.ndim == 1:
        return jnp.array(0.0)
    
    kernel = (z @ z.T) # T, T
    kernel = jnp.tril(kernel, k=-1)
    return kernel.max(axis=-1).mean()

def calc_illumination_score(zs):
    """
    Calculates the illumination score from ASAL.
    The returned score should be minimized.

    Parameters
    ----------
    zs : jnp.ndarray of shape (N, D)
        The latent representation of the images from different simulation parameters.
    """
    N, D = zs.shape
    kernel = (zs @ zs.T) # N, N
    kernel = jnp.where(jnp.eye(N, dtype=bool), -jnp.inf, kernel) # set diagonal to -inf
    return kernel.max(axis=-1).mean()
