import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import trimesh


def export_level_sets(sdf, folder):
    mesh_scale = 0.8
    for level in (-0.02, 0.0, 0.02):
        vtx, faces, _, _ = skimage.measure.marching_cubes(sdf, level)

        vtx = vtx * (mesh_scale * 2.0 / (sdf.shape[0] - 1)) - 1.0
        mesh = trimesh.Trimesh(vtx, faces)
        mesh.export(os.path.join(folder, f"l{level:.2f}.obj"))


def render_slices(sdf, folder):
    num_levels = 6
    levels_pos = np.logspace(-2, 0, num=num_levels)
    levels = np.concatenate((-1.0 * levels_pos[::-1], levels_pos))
    colors = plt.get_cmap("Spectral")(np.linspace(0.0, 1.0, num=num_levels * 2 + 1))

    for i in range(sdf.shape[0]):
        fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

        ax.contourf(sdf[:, :, i], levels=levels, colors=colors)
        ax.contour(sdf[:, :, i], levels=levels, colors="k", linewidths=0.1)
        ax.contour(sdf[:, :, i], levels=[0], colors="k", linewidths=0.3)
        ax.axis("off")

        plt.savefig(os.path.join(folder, f"{i:03d}.png"))
        plt.close(fig)


def main():
    filename = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(os.path.dirname(__file__), "data", "plane.npy")
    )

    folder = filename[:-4]
    os.makedirs(folder, exist_ok=True)

    sdf = np.load(filename)
    print(sdf.max(), sdf.min())

    export_level_sets(sdf, folder)
    render_slices(sdf, folder)


if __name__ == "__main__":
    main()
