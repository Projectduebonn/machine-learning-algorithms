import plotly.graph_objects as go
import numpy as np

def generate_manifold(n_samples):
    """
    Generates a random manifold with the specified number of samples.

    Parameters:
    - n_samples: int, the number of samples to generate

    Returns:
    - X: numpy array, shape (n_samples, n_features), the generated manifold
    """
    # Define the number of features (dimensions) of the manifold
    n_features = 10

    # Generate random points in a high-dimensional space
    X_high = np.random.normal(size=(n_samples, n_features))

    # Apply a non-linear transformation to create a curved manifold
    X = np.sin(X_high) + np.random.normal(scale=0.1, size=X_high.shape)

    return X

# Create the manifold object
n_samples = 1000
np.random.seed(42)
X = generate_manifold(n_samples)

# Project the manifold to 3D using Isomap
from sklearn.manifold import Isomap
embedding = Isomap(n_components=3)
X_3d = embedding.fit_transform(X)

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2], mode='markers', marker=dict(size=3))])

# Set the layout of the plot
fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

# Show the plot
fig.show()
