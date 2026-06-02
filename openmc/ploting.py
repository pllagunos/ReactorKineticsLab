from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.colors import LogNorm

SERPENT_COLORS = {
    'Light Water': (0, 0, 255),
    'Light Water Lower Density': (0, 0, 255),
    'Heavy water with 0.2% of H1': (0, 0, 180),
    'AlMg3': (112, 128, 144),
    'Al pure': (205, 205, 205),
    'Zircaloy2': (78, 78, 78),
    'Beryllium pure': (0, 238, 118),
    'Hafnium': (255, 255, 0),
    'Hafnium for shutdown rods': (255, 255, 0),
    'Boron (ring)': (255, 165, 0),
    'Liquid Deuterium': (0, 245, 255),
    'Liquid Deuterium void': (0, 245, 255),
    'Graphite (cold)': (139, 58, 58),
    'Helium': (135, 206, 250),
    'Magnesium (cold source)': (255, 128, 64),
    'Graphite hard felt': (205, 85, 85),
    'Graphite soft felt': (255, 106, 106),
    'Carbon Dioxide': (255, 255, 204),
    'Concrete (serpentine)': (205, 179, 139),
    'U3Si2 (rho 1.5g/cm3)': (238, 130, 238),
    'U3Si2 (rho 3.0g/cm3)': (199, 21, 133),
    'U-10Mo (19.75 enrich)': (238, 130, 238),
    'AlMg2': (97, 97, 97),
    'AlFeNi': (48, 48, 48),
    'U fuel of Mo99': (255, 30, 30),
    'AlMg3 and Lw': (23, 77, 94),
    'Cd nat': (255, 215, 0),
    'Cd nat (cold source)': (255, 215, 0),
    'Cf-252': (153, 51, 0),
    'LW_aboveCore': (0, 0, 245),
    'LW_Core': (0, 0, 250),
    'LW_belowCore': (0, 0, 255),
    'Natural uranium metal': (210, 35, 35),
    'Heavy water moderator': (0, 90, 200),
    'Light water tank': (80, 170, 255),
    'B4C control rod': (35, 35, 35),
}


def _configured_plot(
    *,
    name: str,
    basis: str,
    origin: tuple[float, float, float],
    width: tuple[float, float],
    pixels: tuple[int, int],
    colors: dict[openmc.Material, tuple[int, int, int]],
) -> openmc.Plot:
    plot = openmc.Plot(name=name)
    plot.basis = basis
    plot.origin = origin
    plot.width = width
    plot.pixels = pixels
    plot.color_by = "material"
    plot.colors = colors
    plot.background = (255, 255, 255)
    return plot


def get_material_colors(
    model: openmc.Universe | openmc.Geometry | openmc.Materials,
) -> dict[openmc.Material, tuple[int, int, int]]:
    if isinstance(model, openmc.Universe):
        materials = model.get_all_materials().values()
    elif isinstance(model, openmc.Geometry):
        materials = model.get_all_materials().values()
    elif isinstance(model, openmc.Materials):
        materials = model
    else:
        raise TypeError("model must be Universe, Geometry, or Materials")

    colors: dict[openmc.Material, tuple[int, int, int]] = {}
    for material in materials:
        rgb = SERPENT_COLORS.get(material.name)
        if rgb is not None:
            colors[material] = rgb
    return colors


def fuel_element_plots(
    model: openmc.Universe | openmc.Geometry,
    outer_radius_cm: float,
    total_height_cm: float,
) -> openmc.Plots:
    colors = get_material_colors(model)

    xy_overview = _configured_plot(
        name="fuel_element_xy_overview",
        basis="xy",
        origin=(0.0, 0.0, 0.0),
        width=(2.2 * outer_radius_cm, 2.2 * outer_radius_cm),
        pixels=(1800, 1800),
        colors=colors,
    )

    xy_detail = _configured_plot(
        name="fuel_element_xy_detail",
        basis="xy",
        origin=(0.0, 0.0, 0.0),
        width=(1.3 * outer_radius_cm, 1.3 * outer_radius_cm),
        pixels=(2400, 2400),
        colors=colors,
    )

    xz_view = _configured_plot(
        name="fuel_element_xz",
        basis="xz",
        origin=(0.0, 0.0, 0.0),
        width=(2.2 * outer_radius_cm, total_height_cm),
        pixels=(1800, 1400),
        colors=colors,
    )

    return openmc.Plots([xy_overview, xy_detail, xz_view])


def reactor_plots(
    model: openmc.Universe | openmc.Geometry,
    fuel_element_outer_radius_cm: float,
    d2o_tank_radius_cm: float,
    h2o_tank_radius_cm: float,
    d2o_tank_height_cm: float,
    total_height_cm: float,
) -> openmc.Plots:
    colors = get_material_colors(model)

    xy_full = _configured_plot(
        name="reactor_xy_full",
        basis="xy",
        origin=(0.0, 0.0, 0.0),
        width=(2.2 * h2o_tank_radius_cm, 2.2 * h2o_tank_radius_cm),
        pixels=(2000, 2000),
        colors=colors,
    )

    xy_fuel_zoom = _configured_plot(
        name="reactor_xy_fuel_zoom",
        basis="xy",
        origin=(0.0, 0.0, 0.0),
        width=(2.2 * fuel_element_outer_radius_cm, 2.2 * fuel_element_outer_radius_cm),
        pixels=(2400, 2400),
        colors=colors,
    )

    xz_full = _configured_plot(
        name="reactor_xz_full",
        basis="xz",
        origin=(0.0, 0.0, 0.0),
        width=(2.2 * h2o_tank_radius_cm, total_height_cm),
        pixels=(2000, 1600),
        colors=colors,
    )

    xz_d2o = _configured_plot(
        name="reactor_xz_d2o_tank",
        basis="xz",
        origin=(0.0, 0.0, 0.0),
        width=(2.2 * d2o_tank_radius_cm, d2o_tank_height_cm),
        pixels=(2200, 1800),
        colors=colors,
    )

    return openmc.Plots([xy_full, xy_fuel_zoom, xz_full, xz_d2o])


def resolve_openmc_exec(openmc_exec: str | None = None) -> str:
    if openmc_exec is not None:
        return openmc_exec

    candidate = Path(sys.executable).with_name("openmc")
    if candidate.exists():
        return str(candidate)

    resolved = shutil.which("openmc")
    if resolved is not None:
        return resolved

    raise FileNotFoundError("Could not find the OpenMC executable. Pass openmc_exec explicitly.")


def export_and_render_plots(
    model: openmc.Model,
    plots: openmc.Plots,
    output_dir: Path,
    openmc_exec: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots.export_to_xml(path=output_dir / "plots.xml")
    model.plots = plots
    model.plot_geometry(
        cwd=output_dir,
        openmc_exec=resolve_openmc_exec(openmc_exec),
        export_model_xml=True,
    )


def plot_reactor_preview(
    model: openmc.Model,
    preview_metadata: dict,
    fuel_parameters,
    reactor_tank_parameters,
    *,
    quick_layout_plot: Callable[..., None],
    layout_title: str,
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    model.geometry.root_universe.plot(
        origin=(0.0, 0.0, 0.0),
        width=(
            2.0 * reactor_tank_parameters.h2o_tank_radius_cm,
            2.0 * reactor_tank_parameters.h2o_tank_radius_cm,
        ),
        basis="xy",
        color_by="material",
        pixels=(1200, 1200),
        axes=axes[0, 0],
    )
    axes[0, 0].set_title("Full reactor midplane XY slice")

    model.geometry.root_universe.plot(
        origin=(0.0, 0.0, 0.0),
        width=(
            2.2 * fuel_parameters.outer_radius_cm,
            2.2 * fuel_parameters.outer_radius_cm,
        ),
        basis="xy",
        color_by="material",
        pixels=(2200, 2200),
        axes=axes[0, 1],
    )
    axes[0, 1].set_title("Fuel-element XY zoom")

    quick_layout_plot(fuel_parameters, ax=axes[1, 0])
    axes[1, 0].set_title(layout_title)

    model.geometry.root_universe.plot(
        origin=(0.0, 0.0, 0.0),
        width=(
            2.0 * reactor_tank_parameters.d2o_tank_radius_cm,
            reactor_tank_parameters.h_d2o_tank_cm,
        ),
        basis="xz",
        color_by="material",
        pixels=(1800, 1600),
        axes=axes[1, 1],
    )
    axes[1, 1].set_title(
        f"D2O tank XZ slice, rod tip z = {preview_metadata['fuel_element']['rod_tip_z_cm']:.1f} cm"
    )
    return fig, axes


def _plot_entropy_history(entropy: np.ndarray, inactive_batches: int) -> None:
    entropy_batches = np.arange(1, entropy.size + 1)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(entropy_batches, entropy, color="tab:blue", linewidth=1.2, alpha=0.7, label="entropy")

    if entropy.size >= 10:
        kernel = np.ones(10, dtype=float) / 10.0
        smoothed = np.convolve(entropy, kernel, mode="valid")
        smoothed_batches = np.arange(10, entropy.size + 1)
        ax.plot(
            smoothed_batches,
            smoothed,
            color="tab:orange",
            linewidth=2.0,
            label="10-batch rolling mean",
        )

    ax.axvline(
        inactive_batches,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=f"inactive cutoff ({inactive_batches})",
    )
    ax.set_xlabel("batch")
    ax.set_ylabel("Shannon entropy")
    ax.set_title("Fission source convergence")
    ax.legend()
    plt.show()


def _plot_mesh_tallies(
    statepoint: openmc.StatePoint,
    reference_metadata: dict,
    fuel_parameters,
    reactor_tank_parameters,
) -> None:
    tally_names = reference_metadata.get("tally_names", [])
    if not tally_names:
        print(
            "This run was executed without mesh tallies for speed. Rebuild with global_mesh_shape or fuel_mesh_shape to plot maps."
        )
        return

    tally_name = "global-mesh" if "global-mesh" in tally_names else tally_names[0]
    mesh_tally = statepoint.get_tally(name=tally_name)
    flux_tally = mesh_tally.get_slice(scores=["flux"])
    fission_tally = mesh_tally.get_slice(scores=["fission"])

    mesh_dimension = tuple(mesh_tally.filters[0].mesh.dimension)
    flux = flux_tally.get_reshaped_data(value="mean").reshape(mesh_dimension).squeeze()
    fission = fission_tally.get_reshaped_data(value="mean").reshape(mesh_dimension).squeeze()
    flux_std = flux_tally.get_reshaped_data(value="std_dev").reshape(mesh_dimension).squeeze()
    fission_std = fission_tally.get_reshaped_data(value="std_dev").reshape(mesh_dimension).squeeze()

    flux_masked = np.ma.masked_less_equal(flux, 0.0)
    fission_masked = np.ma.masked_less_equal(fission, 0.0)
    flux_rel_err = np.ma.masked_where((flux <= 0.0) | ~np.isfinite(flux_std), flux_std / flux)
    fission_rel_err = np.ma.masked_where((fission <= 0.0) | ~np.isfinite(fission_std), fission_std / fission)

    positive_flux = np.asarray(flux[flux > 0.0])
    positive_fission = np.asarray(fission[fission > 0.0])
    if positive_flux.size == 0 or positive_fission.size == 0:
        print(f"{tally_name} contains no positive flux or fission scores to plot.")
        return

    if tally_name == "fuel-mesh":
        extent = [
            -fuel_parameters.outer_radius_cm,
            fuel_parameters.outer_radius_cm,
            -0.5 * fuel_parameters.h_active_cm,
            0.5 * fuel_parameters.h_active_cm,
        ]
        histories_per_bin = reference_metadata.get("fuel_mesh_histories_per_bin")
    else:
        extent = [
            -reactor_tank_parameters.h2o_tank_radius_cm,
            reactor_tank_parameters.h2o_tank_radius_cm,
            -0.5 * reactor_tank_parameters.h_h2o_tank_cm,
            0.5 * reactor_tank_parameters.h_h2o_tank_cm,
        ]
        histories_per_bin = reference_metadata.get("global_mesh_histories_per_bin")

    if histories_per_bin is not None:
        print(f"{tally_name} histories per bin: {histories_per_bin:.3f}")
    print(f"flux nonzero fraction: {(flux > 0.0).mean():.3f}")
    print(f"fission nonzero fraction: {(fission > 0.0).mean():.3f}")
    if histories_per_bin is not None and histories_per_bin < 10.0:
        print(
            "This mesh is strongly under-sampled. Increase active batches and particles, or reduce the mesh size for a readable map."
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    flux_norm = LogNorm(
        vmin=max(np.percentile(positive_flux, 5), positive_flux.min()),
        vmax=positive_flux.max(),
    )
    fission_norm = LogNorm(
        vmin=max(np.percentile(positive_fission, 5), positive_fission.min()),
        vmax=positive_fission.max(),
    )

    flux_image = axes[0, 0].imshow(
        flux_masked,
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=flux_norm,
        cmap="viridis",
    )
    axes[0, 0].set_xlabel("x [cm]")
    axes[0, 0].set_ylabel("z [cm]")
    axes[0, 0].set_title(f"Flux score ({tally_name}, log scale)")
    plt.colorbar(flux_image, ax=axes[0, 0], label="flux")

    fission_image = axes[0, 1].imshow(
        fission_masked,
        origin="lower",
        aspect="auto",
        extent=extent,
        norm=fission_norm,
        cmap="magma",
    )
    axes[0, 1].set_xlabel("x [cm]")
    axes[0, 1].set_ylabel("z [cm]")
    axes[0, 1].set_title(f"Fission score ({tally_name}, log scale)")
    plt.colorbar(fission_image, ax=axes[0, 1], label="fission")

    flux_rel_err_finite = np.asarray(flux_rel_err.filled(np.nan))
    flux_rel_err_finite = flux_rel_err_finite[np.isfinite(flux_rel_err_finite)]
    fission_rel_err_finite = np.asarray(fission_rel_err.filled(np.nan))
    fission_rel_err_finite = fission_rel_err_finite[np.isfinite(fission_rel_err_finite)]

    if flux_rel_err_finite.size == 0 or fission_rel_err_finite.size == 0:
        axes[1, 0].axis("off")
        axes[1, 1].axis("off")
        axes[1, 0].text(
            0.5,
            0.5,
            "Relative error is unavailable with only one active batch.",
            ha="center",
            va="center",
            wrap=True,
        )
        axes[1, 1].text(
            0.5,
            0.5,
            "Run more active batches to estimate tally uncertainty.",
            ha="center",
            va="center",
            wrap=True,
        )
    else:
        flux_err_image = axes[1, 0].imshow(
            flux_rel_err,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=0.0,
            vmax=min(2.0, float(np.percentile(flux_rel_err_finite, 95))),
            cmap="cividis",
        )
        axes[1, 0].set_xlabel("x [cm]")
        axes[1, 0].set_ylabel("z [cm]")
        axes[1, 0].set_title("Flux relative error")
        plt.colorbar(flux_err_image, ax=axes[1, 0], label="std. dev. / mean")

        fission_err_image = axes[1, 1].imshow(
            fission_rel_err,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=0.0,
            vmax=min(2.0, float(np.percentile(fission_rel_err_finite, 95))),
            cmap="cividis",
        )
        axes[1, 1].set_xlabel("x [cm]")
        axes[1, 1].set_ylabel("z [cm]")
        axes[1, 1].set_title("Fission relative error")
        plt.colorbar(fission_err_image, ax=axes[1, 1], label="std. dev. / mean")

    plt.show()


def show_statepoint_diagnostics(
    statepoint_path: str | Path | None,
    reference_metadata: dict | None,
    fuel_parameters,
    reactor_tank_parameters,
) -> None:
    if statepoint_path is None or reference_metadata is None:
        print("Run the previous cell after configuring OpenMC nuclear data to get k_eff and optional flux or fission maps.")
        return

    with openmc.StatePoint(statepoint_path) as statepoint:
        keff = statepoint.keff
        rho_pcm = (keff.n - 1.0) / keff.n * 1.0e5
        print(f"k_eff = {keff.n:.6f} +/- {keff.s:.6f}")
        print(f"reactivity = {rho_pcm:.1f} pcm")

        entropy = np.asarray(getattr(statepoint, "entropy", []), dtype=float)
        if entropy.size:
            inactive_batches = reference_metadata["inactive"]
            entropy_active = entropy[inactive_batches:]
            print(f"entropy mesh shape = {reference_metadata['entropy_mesh_shape']}")
            print(f"final Shannon entropy = {entropy[-1]:.6f}")
            if entropy_active.size >= 10:
                entropy_recent_spread = float(np.ptp(entropy_active[-10:]))
                print(f"entropy spread over last 10 active batches = {entropy_recent_spread:.6f}")
            _plot_entropy_history(entropy, inactive_batches)
        else:
            print("This statepoint does not contain Shannon entropy history.")

        _plot_mesh_tallies(
            statepoint,
            reference_metadata,
            fuel_parameters,
            reactor_tank_parameters,
        )

if __name__ == "__main__":
    print("Import this module from fuel_element.py or reactor_geometry.py to create OpenMC plot sets.")