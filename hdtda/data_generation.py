import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

# Assuming ripser and persim are available in your environment
# from ripser import ripser
# from persim import plot_diagrams


class CircleEmbedding:
    """
    Generate noisy circles in 2D and embed them into high-dimensional spaces.

    This class provides methods for:
    - Linear embeddings (random projection)
    - Nonlinear embeddings (polynomial features)
    - Controlled noise in both low and high dimensions
    """

    def __init__(self, n_samples=200, seed=42):
        """
        Parameters
        ----------
        n_samples : int
            Number of points to sample from the circle
        seed : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_circle_2d(self, radius=1.0, noise_2d=0.05):
        """
        Generate a noisy circle in 2D.

        Parameters
        ----------
        radius : float
            Radius of the circle
        noise_2d : float
            Standard deviation of Gaussian noise added to circle points

        Returns
        -------
        X : ndarray of shape (n_samples, 2)
            Points sampled from noisy circle
        """
        # Sample angles uniformly
        theta = self.rng.uniform(0, 2 * np.pi, self.n_samples)

        # Generate circle points
        X = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])

        # Add noise in 2D
        if noise_2d > 0:
            X += self.rng.normal(0, noise_2d, X.shape)

        return X

    def get_orthonormal_basis(self, out_d=50, in_d=2, seed=0):
        """
        Creates an orthonormal basis for a subspace of dimension in_d in a space of dimension out_d.
        :param out_d: dimension of ambient space
        :param in_d: dimensions in which the data is given
        :param seed: random seed
        :return: basis matrix (in_d, out_d)
        """
        assert out_d >= in_d

        # create orthogonal normal basis by Gauss elemination procedure from random vectors
        np.random.seed(seed)
        basis = np.random.randn(in_d, out_d)
        for i, _ in enumerate(basis):
            basis[i] /= np.linalg.norm(basis[i])
            for j, _ in enumerate(basis):
                if j <= i:
                    continue
                basis[j] = basis[j] - np.dot(basis[i], basis[j]) * basis[i]
                assert np.allclose(
                    np.dot(basis[i], basis[j]), 0
                )  # check that the vectors are orthogonal
        return basis

    def linear_embedding(self, X, target_dim, noise_hd=0.01):
        """
        Embed 2D data into higher dimensions using random linear projection.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            2D input data
        target_dim : int
            Target embedding dimension (must be >= 2)
        noise_hd : float
            Standard deviation of Gaussian noise in additional dimensions

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, target_dim)
            Embedded data
        projection_matrix : ndarray of shape (target_dim, 2)
            The random projection matrix used
        """
        if target_dim < 2:
            raise ValueError("target_dim must be at least 2")

        # Generate random orthonormal basis using QR decomposition
        random_matrix = self.rng.randn(target_dim, target_dim)
        basis, _ = np.linalg.qr(random_matrix)

        # Embed: use first 2 basis vectors for the 2D data
        # X_embedded = X @ basis[:2, :].T preserves distances since basis is orthonormal
        X_embedded = X @ basis[:2, :]

        # proj = self.get_orthonormal_basis(out_d=target_dim, in_d=2)

        # # Project into high-dimensional space
        # X_embedded = X @ proj

        # Add high-dimensional noise to all dimensions
        if noise_hd > 0:
            X_embedded += self.rng.normal(0, noise_hd, X_embedded.shape)

        return X_embedded, basis[:2, :]

    def nonlinear_embedding(
        self, X, target_dim, noise_hd=0.01, embedding_type="polynomial"
    ):
        """
        Embed 2D data nonlinearly into higher dimensions while preserving distances.

        Uses the approach of embedding into a higher-dimensional feature space,
        then centering and normalizing to preserve relative distance structure.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 2)
            2D input data
        target_dim : int
            Target embedding dimension (must be >= 2)
        noise_hd : float
            Standard deviation of Gaussian noise in additional dimensions
        embedding_type : str
            Type of nonlinear embedding: 'polynomial', 'trigonometric', or 'mixed'

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, target_dim)
            Embedded data
        """
        n_samples = X.shape[0]

        # Compute pairwise distances in original space
        from scipy.spatial.distance import pdist, squareform

        original_distances = squareform(pdist(X))

        # Create initial nonlinear features
        features = [X]  # Start with original coordinates

        if embedding_type == "polynomial":
            # Add polynomial features up to degree 3
            features.append(X**2)
            features.append(X[:, [0]] * X[:, [1]])  # interaction term
            features.append(X**3)

        elif embedding_type == "trigonometric":
            # Add trigonometric features
            for freq in range(1, 4):
                features.append(np.sin(freq * X))
                features.append(np.cos(freq * X))

        elif embedding_type == "mixed":
            # Combine polynomial and trigonometric
            features.append(X**2)
            features.append(np.sin(X))
            features.append(np.cos(X))
            features.append(X[:, [0]] * X[:, [1]])

        # Concatenate all features
        X_features = np.hstack(features)

        # If we have more features than target_dim, use PCA to reduce
        # If we have fewer, pad with random projections
        if X_features.shape[1] > target_dim:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=target_dim)
            X_embedded = pca.fit_transform(X_features)
        else:
            X_embedded = np.zeros((n_samples, target_dim))
            X_embedded[:, : X_features.shape[1]] = X_features
            # Fill remaining with small random projections
            if target_dim > X_features.shape[1]:
                remaining = target_dim - X_features.shape[1]
                proj = self.rng.randn(2, remaining) * 0.1
                X_embedded[:, X_features.shape[1] :] = X @ proj

        # Scale to match original distance structure
        embedded_distances = squareform(pdist(X_embedded))
        # Use median distance ratio to scale (robust to outliers)
        orig_median = np.median(original_distances[original_distances > 0])
        embed_median = np.median(embedded_distances[embedded_distances > 0])

        if embed_median > 0:
            scale_factor = orig_median / embed_median
            X_embedded *= scale_factor

        # Add high-dimensional noise
        if noise_hd > 0:
            X_embedded += self.rng.normal(0, noise_hd, X_embedded.shape)

        return X_embedded

    def generate_dataset(
        self,
        target_dim,
        radius=1.0,
        noise_2d=0.05,
        noise_hd=0.01,
        embedding="linear",
        **kwargs,
    ):
        """
        Generate complete dataset: circle in 2D embedded in high dimensions.

        Parameters
        ----------
        target_dim : int
            Target embedding dimension
        radius : float
            Radius of the circle
        noise_2d : float
            Noise level in the original 2D circle
        noise_hd : float
            Noise level in the high-dimensional embedding
        embedding : str
            'linear' or 'nonlinear' (or specific type for nonlinear)
        **kwargs : dict
            Additional arguments passed to embedding functions

        Returns
        -------
        X_2d : ndarray of shape (n_samples, 2)
            Original 2D circle data
        X_embedded : ndarray of shape (n_samples, target_dim)
            Embedded high-dimensional data
        metadata : dict
            Dictionary containing embedding parameters and matrices
        """
        # Generate 2D circle
        X_2d = self.generate_circle_2d(radius=radius, noise_2d=noise_2d)

        # Embed into high dimensions
        metadata = {
            "radius": radius,
            "noise_2d": noise_2d,
            "noise_hd": noise_hd,
            "target_dim": target_dim,
            "embedding_type": embedding,
        }

        if embedding == "linear":
            X_embedded, proj = self.linear_embedding(X_2d, target_dim, noise_hd)
            metadata["projection_matrix"] = proj
        else:
            embedding_type = kwargs.get("embedding_type", "polynomial")
            X_embedded = self.nonlinear_embedding(
                X_2d, target_dim, noise_hd, embedding_type
            )
            metadata["nonlinear_type"] = embedding_type

        return X_2d, X_embedded, metadata

    def visualize(self, X_2d, X_embedded, metadata):
        """
        Visualize the original 2D data and PCA projection of embedded data.

        Parameters
        ----------
        X_2d : ndarray
            Original 2D data
        X_embedded : ndarray
            High-dimensional embedded data
        metadata : dict
            Embedding metadata
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original 2D circle
        axes[0].scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, s=20)
        axes[0].set_xlabel("X1")
        axes[0].set_ylabel("X2")
        axes[0].set_title("Original 2D Circle")
        axes[0].axis("equal")
        axes[0].grid(True, alpha=0.3)

        # PCA projection of embedded data (back to 2D for visualization)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_embedded)

        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=20)
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        axes[1].set_title(f'PCA of {metadata["target_dim"]}D Embedding')
        axes[1].axis("equal")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class TorusEmbedding:
    """
    Generate noisy 2-tori in 3D and embed them into high-dimensional spaces.

    This class provides methods for:
    - Generating a 3D torus point cloud
    - Linear embeddings (random projection)
    - Nonlinear embeddings (polynomial features)
    - Controlled noise in both low (3D) and high dimensions
    """

    def __init__(self, n_samples=1000, seed=42):
        """
        Parameters
        ----------
        n_samples : int
            Number of points to sample from the torus.
            (Note: More points are generally needed for a torus than a circle)
        seed : int
            Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_torus_3d(self, major_radius=2.0, minor_radius=1.0, noise_3d=0.05):
        """
        Generate a noisy torus in 3D.

        Parameters
        ----------
        major_radius : float
            Radius from the center of the hole to the center of the tube (R)
        minor_radius : float
            Radius of the tube (r)
        noise_3d : float
            Standard deviation of Gaussian noise added to torus points

        Returns
        -------
        X : ndarray of shape (n_samples, 3)
            Points sampled from noisy torus
        """
        # Sample two sets of angles uniformly
        theta = self.rng.uniform(0, 2 * np.pi, self.n_samples)
        phi = self.rng.uniform(0, 2 * np.pi, self.n_samples)

        # Generate torus points using parametric equations
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)

        X = np.column_stack([x, y, z])

        # Add noise in 3D
        if noise_3d > 0:
            X += self.rng.normal(0, noise_3d, X.shape)

        return X

    def linear_embedding(self, X, target_dim, noise_hd=0.01):
        """
        Embed 3D data into higher dimensions using random linear projection.

        This is an isometric embedding (preserves distances) up to noise.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 3)
            3D input data
        target_dim : int
            Target embedding dimension (must be >= 3)
        noise_hd : float
            Standard deviation of Gaussian noise in all high dimensions

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, target_dim)
            Embedded data
        projection_matrix : ndarray of shape (3, target_dim)
            The random projection matrix used
        """
        if target_dim < 3:
            raise ValueError("target_dim must be at least 3")

        # Generate random orthonormal basis using QR decomposition
        random_matrix = self.rng.randn(target_dim, target_dim)
        basis, _ = np.linalg.qr(random_matrix)

        # Use the first 3 basis vectors (as rows) for the 3D data
        # X is (n_samples, 3). basis[:3, :] is (3, target_dim).
        # X @ basis[:3, :] maps (x, y, z) to x*v_1 + y*v_2 + z*v_3
        projection_matrix = basis[:3, :]
        X_embedded = X @ projection_matrix

        # Add high-dimensional noise to all dimensions
        if noise_hd > 0:
            X_embedded += self.rng.normal(0, noise_hd, X_embedded.shape)

        return X_embedded, projection_matrix

    def nonlinear_embedding(
        self, X, target_dim, noise_hd=0.01, embedding_type="polynomial"
    ):
        """
        Embed 3D data nonlinearly into higher dimensions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, 3)
            3D input data
        target_dim : int
            Target embedding dimension
        noise_hd : float
            Standard deviation of Gaussian noise in additional dimensions
        embedding_type : str
            Type of nonlinear embedding: 'polynomial', 'trigonometric', or 'mixed'

        Returns
        -------
        X_embedded : ndarray of shape (n_samples, target_dim)
            Embedded data
        """
        n_samples = X.shape[0]

        # Compute pairwise distances in original space
        original_distances = squareform(pdist(X))

        # Create initial nonlinear features
        features = [X]  # Start with original coordinates (x, y, z)

        if embedding_type == "polynomial":
            # Add polynomial features up to degree 3
            features.append(X**2)  # x^2, y^2, z^2
            # Interaction terms
            features.append(X[:, [0]] * X[:, [1]])  # xy
            features.append(X[:, [0]] * X[:, [2]])  # xz
            features.append(X[:, [1]] * X[:, [2]])  # yz
            features.append(X**3)  # x^3, y^3, z^3

        elif embedding_type == "trigonometric":
            # Add trigonometric features
            for freq in range(1, 3):  # Keep feature count reasonable
                features.append(np.sin(freq * X))
                features.append(np.cos(freq * X))

        elif embedding_type == "mixed":
            # Combine polynomial and trigonometric
            features.append(X**2)
            features.append(np.sin(X))
            features.append(np.cos(X))
            features.append(X[:, [0]] * X[:, [1]])
            features.append(X[:, [0]] * X[:, [2]])

        # Concatenate all features
        X_features = np.hstack(features)

        # If we have more features than target_dim, use PCA to reduce
        # If we have fewer, pad with random projections
        if X_features.shape[1] > target_dim:
            pca = PCA(n_components=target_dim, random_state=self.seed)
            X_embedded = pca.fit_transform(X_features)
        else:
            X_embedded = np.zeros((n_samples, target_dim))
            X_embedded[:, : X_features.shape[1]] = X_features
            # Fill remaining with small random projections
            if target_dim > X_features.shape[1]:
                remaining = target_dim - X_features.shape[1]
                proj = self.rng.randn(3, remaining) * 0.1  # Project from 3D
                X_embedded[:, X_features.shape[1] :] = X @ proj

        # Scale to match original distance structure
        embedded_distances = squareform(pdist(X_embedded))
        orig_median = np.median(original_distances[original_distances > 0])
        embed_median = np.median(embedded_distances[embedded_distances > 0])

        if embed_median > 0 and orig_median > 0:
            scale_factor = orig_median / embed_median
            X_embedded *= scale_factor

        # Add high-dimensional noise
        if noise_hd > 0:
            X_embedded += self.rng.normal(0, noise_hd, X_embedded.shape)

        return X_embedded

    def generate_dataset(
        self,
        target_dim,
        major_radius=2.0,
        minor_radius=1.0,
        noise_3d=0.05,
        noise_hd=0.01,
        embedding="linear",
        **kwargs,
    ):
        """
        Generate complete dataset: torus in 3D embedded in high dimensions.

        Parameters
        ----------
        target_dim : int
            Target embedding dimension
        major_radius : float
            Major radius of the torus
        minor_radius : float
            Minor radius of the torus
        noise_3d : float
            Noise level in the original 3D torus
        noise_hd : float
            Noise level in the high-dimensional embedding
        embedding : str
            'linear' or 'nonlinear'
        **kwargs : dict
            Additional arguments passed to embedding functions

        Returns
        -------
        X_3d : ndarray of shape (n_samples, 3)
            Original 3D torus data
        X_embedded : ndarray of shape (n_samples, target_dim)
            Embedded high-dimensional data
        metadata : dict
            Dictionary containing embedding parameters
        """
        # Generate 3D torus
        X_3d = self.generate_torus_3d(
            major_radius=major_radius, minor_radius=minor_radius, noise_3d=noise_3d
        )

        # Embed into high dimensions
        metadata = {
            "major_radius": major_radius,
            "minor_radius": minor_radius,
            "noise_3d": noise_3d,
            "noise_hd": noise_hd,
            "target_dim": target_dim,
            "embedding_type": embedding,
        }

        if embedding == "linear":
            X_embedded, proj = self.linear_embedding(X_3d, target_dim, noise_hd)
            metadata["projection_matrix"] = proj
        else:
            embedding_type = kwargs.get("embedding_type", "polynomial")
            X_embedded = self.nonlinear_embedding(
                X_3d, target_dim, noise_hd, embedding_type
            )
            metadata["nonlinear_type"] = embedding_type

        return X_3d, X_embedded, metadata
