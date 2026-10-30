"""
Icosahedron geometry for the Dymaxion map (numpy only; used by both
world_map_prep.py and world_projections.py).

Fuller's orientation of the icosahedron is taken from Robert W. Gray's
reference Dymaxion implementation (dymax.c): the 12 vertices below, as
unit vectors with x = cos(lat) cos(lon), y = cos(lat) sin(lon), z = sin(lat).

The map is made by projecting each point gnomonically onto the face it falls
in and unfolding the faces into a net. Which edges are cut and which stay as
hinges decides where the map tears; choosing hinges along every edge that
crosses land keeps the continents whole (see ocean_cut_hinges).
"""
import numpy as np

GRAY_VERTICES = np.array([
    [0.420152426708710003, 0.078145249402782959, 0.904082550615019298],
    [0.995009439436241649, -0.091347795276427931, 0.040147175877166645],
    [0.518836730327364437, 0.835420380378235850, 0.181331837557262454],
    [-0.414682225320335218, 0.655962405434800777, 0.630675807891475371],
    [-0.515455959944041808, -0.381716898287133011, 0.767200992517747538],
    [0.355781402532944713, -0.843580002466178147, 0.402234226602925571],
    [0.414682225320335218, -0.655962405434800777, -0.630675807891475371],
    [0.515455959944041808, 0.381716898287133011, -0.767200992517747538],
    [-0.355781402532944713, 0.843580002466178147, -0.402234226602925571],
    [-0.995009439436241649, 0.091347795276427931, -0.040147175877166645],
    [-0.518836730327364437, -0.835420380378235850, -0.181331837557262454],
    [-0.420152426708710003, -0.078145249402782959, -0.904082550615019298],
])

HINGE_ANGLE = np.pi - np.arccos(-np.sqrt(5) / 3)   # 41.81 degrees


def rot(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def faces_of(V):
    """The 20 faces (outward-oriented vertex triples) of an icosahedron."""
    from scipy.spatial import ConvexHull
    faces = ConvexHull(V).simplices
    return np.array([
        f if np.dot(np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]),
                    V[f].mean(0)) > 0 else f[::-1]
        for f in faces])


def edge_faces(faces):
    """{(a, b) vertex edge: [face, face]}"""
    out = {}
    for fi, f in enumerate(faces):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            out.setdefault((min(f[a], f[b]), max(f[a], f[b])), []).append(fi)
    return out


class Icosahedron:
    """Folded and unfolded positions of an icosahedral map.

    hinges: the 19 vertex-edges kept joined (the rest are cut); root: the
    face that stays put while the others unfold around it.
    """

    def __init__(self, V, faces, hinges, root):
        self.V, self.faces = np.asarray(V, float), np.asarray(faces)
        self.centres = self.V[self.faces].mean(1)
        self.normals = self.centres / np.linalg.norm(
            self.centres, axis=1, keepdims=True)
        self.plane_d = np.einsum("ij,ij->i", self.normals,
                                 self.V[self.faces[:, 0]])
        ef = edge_faces(self.faces)
        hinges = [tuple(sorted(map(int, e))) for e in hinges]
        adj = {fi: [] for fi in range(len(self.faces))}
        self.hinge_pairs = set()
        for e in hinges:
            f1, f2 = ef[e]
            adj[f1].append((f2, e))
            adj[f2].append((f1, e))
            self.hinge_pairs.add((min(f1, f2), max(f1, f2)))
        # tree from the root over hinge edges only
        self.parent, self.hinge = {root: None}, {root: None}
        self.order, queue = [root], [root]
        while queue:
            fi = queue.pop(0)
            for nb, e in adj[fi]:
                if nb not in self.parent:
                    self.parent[nb], self.hinge[nb] = fi, e
                    self.order.append(nb)
                    queue.append(nb)
        if len(self.order) != len(self.faces):
            raise ValueError("hinges do not form a spanning tree")
        # rotation direction of each hinge: the one that brings the face's
        # normal onto its parent's
        self.sign = {root: 0.0}
        for fi in self.order[1:]:
            a, b = self.hinge[fi]
            axis = self.V[b] - self.V[a]
            n_par = self.normals[self.parent[fi]]
            errs = [np.linalg.norm(rot(axis, s * HINGE_ANGLE) @ self.normals[fi]
                                   - n_par) for s in (1.0, -1.0)]
            self.sign[fi] = 1.0 if errs[0] < errs[1] else -1.0

    def face_of(self, unit_points):
        return np.argmax(unit_points @ self.normals.T, axis=1)

    def gnomonic(self, unit_points, face):
        """Project directions out onto the given faces' planes."""
        n, d = self.normals[face], self.plane_d[face]
        return unit_points * (d / np.einsum("ij,ij->i", unit_points, n))[:, None]

    def transforms(self, u):
        """4x4 transform of each face at unfold fraction u (0 = folded)."""
        out = {}
        for fi in self.order:
            if self.parent[fi] is None:
                out[fi] = np.eye(4)
                continue
            a, b = self.hinge[fi]
            A = self.V[a]
            R = rot(self.V[b] - A, self.sign[fi] * HINGE_ANGLE * u)
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = A - R @ A                       # about the hinge line
            out[fi] = out[self.parent[fi]] @ M
        return out

    def flat_triangles(self):
        """Unfolded face triangles in 2D (coordinates in the root's plane)."""
        T = self.transforms(1.0)
        n0 = self.normals[self.order[0]]
        ax = np.cross(n0, [0.0, 0.0, 1.0])
        R = (np.eye(3) if np.linalg.norm(ax) < 1e-9
             else rot(ax, np.arccos(np.clip(n0[2], -1, 1))))
        return [((self.V[self.faces[fi]] @ T[fi][:3, :3].T + T[fi][:3, 3])
                 @ R.T)[:, :2] for fi in range(len(self.faces))]

    def overlaps(self):
        """Do any two faces overlap once unfolded? (separating axis test,
        triangles shrunk slightly so shared edges don't count)"""
        tris = [t + 1e-3 * (t.mean(0) - t) for t in self.flat_triangles()]
        for i in range(len(tris)):
            for j in range(i + 1, len(tris)):
                A, B = tris[i], tris[j]
                separated = False
                for T in (A, B):
                    for k in range(3):
                        e = T[(k + 1) % 3] - T[k]
                        n = np.array([-e[1], e[0]])
                        a, b = A @ n, B @ n
                        if a.min() >= b.max() - 1e-12 or b.min() >= a.max() - 1e-12:
                            separated = True
                            break
                    if separated:
                        break
                if not separated:
                    return True
        return False


def ocean_cut_hinges(V, faces, land_weight, tries=3000, seed=0):
    """Choose the 19 hinge edges so the 11 cuts cross as little land as
    possible, subject to the net unfolding without overlap.

    land_weight: {vertex edge: amount of land along it}. A maximum spanning
    tree on these weights keeps every land-crossing edge joined where
    possible; random tie-breaking explores the many trees of equal weight.
    """
    ef = edge_faces(faces)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(tries):
        parent = list(range(len(faces)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        noisy = {e: land_weight[e] + rng.uniform(0, 1) for e in ef}
        hinges, cut_land = [], 0
        for e in sorted(ef, key=lambda e: -noisy[e]):
            f1, f2 = ef[e]
            if find(f1) != find(f2):
                parent[find(f1)] = find(f2)
                hinges.append(e)
            else:
                cut_land += land_weight[e]
        root = int(rng.integers(len(faces)))
        if best is not None and cut_land >= best[0]:
            continue
        if not Icosahedron(V, faces, hinges, root).overlaps():
            best = (cut_land, hinges, root)
            if cut_land == 0:
                break
    return best
