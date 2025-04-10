Usage
=====

Basic Usage
----------

Here's a basic example of using RiemannAX to perform operations on a Riemannian manifold:

.. code-block:: python

    import jax.numpy as jnp
    import riemannax as rieax

    # Initialize a manifold
    manifold = rieax.Sphere(dim=3)

    # Define a point
    x = jnp.array([0.0, 0.0, 1.0])

    # Define a tangent vector
    v = jnp.array([1.0, 0.0, 0.0])

    # Compute the exponential map
    y = manifold.exp(x, v)

    # Calculate the geodesic distance
    distance = manifold.dist(x, y)

    print(f"Distance: {distance}")
