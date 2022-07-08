import torch
from scipy.spatial.transform import Rotation


class Kearsley:
    """
    A Kearsley transformation.
    Algorithm to minimize the sum of the squared distances between two sets of points.
    Author: Simon K. Kearsley
    Paper: "On the orthogonal transformation used for structural comparisons"
    https://doi.org/10.1107/S0108767388010128

    Args:
        rot: Rotation
             A scipy.spatial.transform Rotation in 3 dimensions.
        trans: ndarray, shape (3,)
             The 3D translation.

    Examples:

    Given two sets of 3D points u and v:
    >>> u
    array([[0, 0, 0],
           [0, 0, 1],
           [0, 0, 2],
           ...,
           [9, 9, 7],
           [9, 9, 8],
           [9, 9, 9]])
    >>> v
    array([[ 30.50347534, -20.16089091,  -7.42752623],
           [ 30.77704903, -21.02339348,  -7.27823201],
           [ 31.3215374 , -21.99452332,  -7.15703548],
           ...,
           [ 42.05988643, -23.50924264, -15.59516355],
           [ 42.27217891, -24.36478643, -15.59064995],
           [ 42.66080502, -25.27318759, -15.386241  ]])
    It is possible to calculate the rotation and translation that minimize the root mean squared deviation:
    >>> from kearsley import Kearsley
    >>> k = Kearsley()
    >>> rmsd = k.fit(u, v)
    >>> rmsd
    0.10003430497284149
    The rotation and translation are the attributes of the class:
    >>> k.rot.as_matrix()
    array([[ 0.05552838, -0.04405506, -0.99748471],
           [ 0.91956342,  0.39147652,  0.03390061],
           [ 0.38899835, -0.9191329 ,  0.06224948]])
    >>> k.trans
    array([ 30.46560753, -20.15086287,  -7.34422276])

    Once fitted you can apply the transformation:
    >>> v_transform = k.transform(v)
    >>> v_transform
    array([[ 0.08563846,  0.02807207,  0.01876202],
           [-0.01009153, -0.0529479 ,  0.92722971],
           [-0.05796549,  0.07167779,  2.03917659],
           ...,
           [ 9.0219524 ,  9.067236  ,  7.08333594],
           [ 9.06692944,  8.9276801 ,  7.95255679],
           [ 8.92463409,  8.93635832,  8.95139744]])
    It is also possible to fit and transform with one command:
    >>> v_transform, rmsd = k.fit_transform(u, v)
    >>> rmsd
    0.10003430497284149
    >>> v_transform
    array([[ 0.08563846,  0.02807207,  0.01876202],
           [-0.01009153, -0.0529479 ,  0.92722971],
           [-0.05796549,  0.07167779,  2.03917659],
           ...,
           [ 9.0219524 ,  9.067236  ,  7.08333594],
           [ 9.06692944,  8.9276801 ,  7.95255679],
           [ 8.92463409,  8.93635832,  8.95139744]])
    """

    def __init__(self, requires_grad: bool = True):
        """The attributes are filled with a rotation of zero degrees and a translation of zero."""
        self.requires_grad = requires_grad
        self.rot = Rotation.from_quat([0, 0, 0, 1])
        self.trans = torch.zeros(3)

    def _kearsley_matrix(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        """
        Calculates the Kearsley matrix.

        Args:
            x: shape (N, 3), Input array with 3D points, centered at zero.
            y: shape (N, 3), Input array with 3D points, centered at zero.

        Returns:
            K: shape (4, 4), The Kearsley matrix.
        """
        # diff and sum quantities
        d, s = x - y, x + y

        # extract columns to simplify notation
        d0, d1, d2 = d[:, 0], d[:, 1], d[:, 2]
        s0, s1, s2 = s[:, 0], s[:, 1], s[:, 2]

        # fill kearsley matrix
        K = torch.zeros((4, 4))
        K[0, 0] = torch.matmul(d0, d0) + torch.matmul(d1, d1) + torch.matmul(d2, d2)
        K[1, 0] = torch.matmul(s1, d2) - torch.matmul(d1, s2)
        K[2, 0] = torch.matmul(d0, s2) - torch.matmul(s0, d2)
        K[3, 0] = torch.matmul(s0, d1) - torch.matmul(d0, s1)
        K[1, 1] = torch.matmul(s1, s1) + torch.matmul(s2, s2) + torch.matmul(d0, d0)
        K[2, 1] = torch.matmul(d0, d1) - torch.matmul(s0, s1)
        K[3, 1] = torch.matmul(d0, d2) - torch.matmul(s0, s2)
        K[2, 2] = torch.matmul(s0, s0) + torch.matmul(s2, s2) + torch.matmul(d1, d1)
        K[3, 2] = torch.matmul(d1, d2) - torch.matmul(s1, s2)
        K[3, 3] = torch.matmul(s0, s0) + torch.matmul(s1, s1) + torch.matmul(d2, d2)

        return K

    def fit(
        self, u: torch.tensor, v: torch.tensor, indices: list = None
    ) -> torch.tensor:
        """
        Calculates the rotation and translation that best fits both sets of points.

        Args:
            u: shape (3*N,), Input array with 3D points.
            v: shape (3*N,), Input array with 3D points.
            indices: list of indices that are included in calculation of rmsd

        Returns:
            rmsd: The root mean squared deviation.
        """
        u_r = torch.reshape(u, (int(len(u) / 3), 3))
        v_r = torch.reshape(v, (int(len(v) / 3), 3))

        if indices != None:
            u_r = u_r[indices]
            v_r = v_r[indices]

        # centroids
        centroid_u = u_r.mean(axis=0)
        centroid_v = v_r.mean(axis=0)

        # center both sets of points
        x, y = u_r - centroid_u, v_r - centroid_v

        # calculate Kearsley matrix
        K = self._kearsley_matrix(x, y)

        # diagonalize K
        eig_vals, eig_vecs = torch.linalg.eigh(K)

        # calculate rotation and translation
        if self.requires_grad == False:
            q = eig_vecs[:, 0]
            q = torch.roll(q, 3)

            # this would break autograd!
            self.rot = Rotation.from_quat(q).inv()
            self.trans = centroid_v - self.rot.inv().apply(centroid_u)

        # calculate rmsd
        eig_val = torch.abs(eig_vals[0])
        rmsd = torch.sqrt(eig_val / u_r.size(dim=0))

        return rmsd

    def transform(self, u: torch.matmul) -> torch.matmul:
        """
        Transforms a list of 3D points with a rotation and a translation.

        Args:
            u: shape (N, 3), Input array with 3D points.

        Returns:
            t: shape (N, 3), Input points transformed.

        Raises:
            ValueError: If the input points don't have the correct shape.
        """
        print(u)
        if len(u.size(dim=0)) != 2:
            raise ValueError("Input array must have 2 dimensions")

        if u.sizre(dim=1) != 3:
            raise ValueError("Input array must have 3 columns")

        return self.rot.apply(u - self.trans)

    def fit_transform(
        self, u: torch.tensor, v: torch.tensor, indices: list = None
    ) -> tuple:
        """
        Calculates the rotation and translation that best fits both sets of points and
        applies the transformation to the second set.

        Args:
            u: shape (N, 3), Input array with 3D points.
            v: shape (N, 3), Input array with 3D points.

        Returns:
            array: shape (N, 3), Input points transformed.
            rmsd: The root mean squared deviation.
        """
        self.requires_grad = False
        rmsd = self.fit(u, v, indices=indices)
        return self.transform(v), rmsd
