Examples
========

Optimization on the Sphere
-------------------------

This example demonstrates solving an optimization problem on a sphere:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import riemannax as rieax

    # Define the objective function
    def objective(x):
        return jnp.sum((x - jnp.array([0.5, 0.5, 0.7])) ** 2)

    # Define the gradient
    grad_objective = jax.grad(objective)

    # Initialize the manifold
    manifold = rieax.Sphere(dim=3)

    # Initial point
    x0 = jnp.array([1.0, 0.0, 0.0])
    x0 = manifold.proj(x0)  # Project onto the manifold

    # Setup the optimizer
    optimizer = rieax.optimizers.RiemannianGradientDescent(manifold, step_size=0.1)

    # Optimization loop
    x = x0
    for i in range(100):
        grad = grad_objective(x)
        riemannian_grad = manifold.euclidean_to_riemannian_grad(x, grad)
        x = optimizer.step(x, riemannian_grad)

    print(f"Optimal solution: {x}")
    print(f"Objective value: {objective(x)}")

Embedding in Hyperbolic Space
---------------------------

This example shows how to embed graph data in hyperbolic space:

.. code-block:: python

    import jax.numpy as jnp
    import riemannax as rieax
    import networkx as nx

    # Create a graph
    G = nx.karate_club_graph()

    # Generate initial embeddings
    initial_embedding = jnp.random.normal(size=(G.number_of_nodes(), 2))

    # Initialize hyperbolic space
    manifold = rieax.Hyperbolic(dim=2)

    # Project embeddings onto the manifold
    embedding = manifold.proj(initial_embedding)

    print(f"Hyperbolic embeddings: {embedding}")
