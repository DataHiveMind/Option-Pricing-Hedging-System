import jax
import jax.numpy as jnp
import gpjax as gpx
from gpjax.kernels import RBF
from gpjax.mean_functions import Zero
from gpjax.likelihoods import Gaussian

def create_gp_model(num_data, num_latent_gps=1):
    """
    Creates a Gaussian process model for option pricing.

    Args:
        num_data (int): The number of data points.
        num_latent_gps (int): Number of latent GPs.  Defaults to 1.

    Returns:
        gpjax.models.GP: A Gaussian process model.
    """
    # Define the kernel (e.g., RBF)
    kernel = RBF(lengthscale=jnp.ones(1), variance=jnp.ones(1))

    # Define the mean function (e.g., Zero)
    mean_function = Zero()

    # Define the likelihood
    likelihood = Gaussian(num_datapoints=num_data) # changed from num_data

    # Create the GP model
    model = gpx.models.GP(
        mean_function=mean_function,
        kernel=kernel,
        likelihood=likelihood,
        num_latent_gps=num_latent_gps
    )
    return model

def train_gp_model(model, train_data, num_epochs=100):
    """
    Trains the Gaussian process model.

    Args:
        model (gpjax.models.GP): The Gaussian process model to train.
        train_data (tuple): A tuple containing (X, Y), where X is the input
            data (e.g., time to expiration, strike price) and Y is the
            option price.
        num_epochs (int): The number of training epochs.

    Returns:
        gpjax.models.GP: The trained Gaussian process model.
    """
    import optax as opt
    from gpjax.losses import LogPosteriorDensity
    from jax import jit

    # Define the optimizer (e.g., Adam)
    optimizer = opt.adam(learning_rate=0.01)

    # Define the loss function (e.g., negative log marginal likelihood)
    loss_fn = LogPosteriorDensity(model)

    # Get the model parameters
    params = model.initialise_params()

    # Define the optimization loop
    @jit
    def update(params, opt_state, batch):
        """Single optimization step."""
        loss_value, grads = jax.value_and_grad(loss_fn)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    opt_state = optimizer.init(params)

    # Train the model
    for epoch in range(num_epochs):
        params, opt_state, loss_value = update(params, opt_state, train_data)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss_value:.4f}")

    return model, params  # Return the trained parameters

def predict_option_price(model, params, test_data):
    """
    Predicts option prices using the trained Gaussian process model.

    Args:
        model (gpjax.models.GP): The trained Gaussian process model.
        params: The trained model parameters
        test_data (jnp.ndarray): The input data for which to make predictions
            (e.g., time to expiration, strike price).

    Returns:
        jnp.ndarray: The predicted option prices.
    """
    posterior_dist = model.posterior(params, test_data)
    mean_prediction = posterior_dist.mean
    return mean_prediction
    # Note: The posterior distribution can also provide uncertainty estimates 