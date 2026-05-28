import openmc
import matplotlib.pyplot as plt
import shutil
import sys
from pathlib import Path

# how to actually make ploting useful?

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


def create_plot(materials: openmc.Materials,id:int) -> openmc.Plot:
    mat_by_name = {mat.name: mat for mat in materials}

    # Create dictionary material -> color normalized
    colors = {
        mat_by_name[name]: rgb
        for name, rgb in SERPENT_COLORS.items()
        if name in mat_by_name
    }

    # Create a basic plot XY
    plot = openmc.Plot(plot_id=id)
    plot.color_by = 'material'
    plot.colors = colors

    plot2 = openmc.Plot(plot_id=id)
    plot2.color_by = "material"
    plot2.colors = colors
    plot2.basis = "xz"

    # Parametri dalla carta MCNP
    x_low, x_high = -30.0, 5.6
    y_low, y_high = -5.6, 5.6

    # Calcolo centro e larghezza
    center_x = 0.5*(x_low + x_high)
    center_y = 0.5*(y_low + y_high)
    width_x  = x_high  - x_low
    width_y  = y_high  - y_low

    # Definizione del plot
    plot33 = openmc.Plot(plot_id=33)
    plot33.basis  = 'xy'                           # piano XY
    plot33.origin = (center_x, center_y, 0.0)      # z=0
    plot33.width  = (width_x, width_y)             # larghezza in X e Y
    plot33.pixels = (4000, 4000)  
    plot33.color_by = "material"
    plot33.colors = colors               

    #return openmc.Plots([plot,plot2])
    return plot33




def get_material_colors(model):
    """
    Ritorna un dict {Material: (r,g,b)} pronto per Plot.colors,
    usando direttamente valori 0–255 senza normalizzazione.
    
    model può essere un openmc.Universe o un openmc.Geometry.
    """
    # Estrae tutti i materiali dal modello
    if isinstance(model, openmc.Universe):
        mats = model.get_all_materials().values()
    elif isinstance(model, openmc.Geometry):
        mats = model.get_all_materials().values()
    elif isinstance(model, openmc.Materials):
        mats = model
    else:
        raise TypeError("model deve essere Universe, Geometry o Materials")
    
    # Costruisce il mapping solo per i materiali presenti in serpent_colors
    colors = {}
    for m in mats:
        rgb = SERPENT_COLORS.get(m.name)#type:ignore
        if rgb is not None:
            colors[m] = rgb  # valori 0–255, openmc li accetta direttamente
    
    return colors

def region_of(univ):
    """Returns the union of the regions of all cells in a Universe."""
    cells = univ.cells.values()
    # Costruisco l'unione delle loro regioni
    regs = [c.region for c in cells]
    if not regs:
        raise ValueError(f"Il universe {univ.name} non contiene celle!")
    
    reg_union = regs[0]
    for r in regs[1:]:
        reg_union &= r
    return reg_union

def verify_material_colors(materials):
    """
    Genera un grafico orizzontale per verificare i colori
    associati a ciascun materiale usando create_plot.
    """
    # Crea il plot temporaneo
    plot = create_plot(materials, id=1)
    mapping = plot.colors

    # Prepara nomi e colori normalizzati [0–1]
    names  = [mat.name for mat in mapping]
    rgbs   = [rgb for rgb in mapping.values()]
    colors = [(r/255, g/255, b/255) for r, g, b in rgbs]

    # Disegna barre orizzontali
    fig, ax = plt.subplots(figsize=(6, len(names)*0.5))
    ax.barh(range(len(names)), [1]*len(names), color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xticks([])
    ax.set_title("Verifica colori materiali", fontsize=12)
    plt.tight_layout()
    plt.show()

def basic_4_plots(Root:openmc.Universe)->tuple[openmc.Plot,openmc.Plot,openmc.Plot,openmc.Plot]:
    # Create the four basic plot used for the plot of each component
    # use this to have the basic option already set, and then adjust the origin and width to each component

    colors=get_material_colors(Root)

    first_plot = openmc.Plot(name="XY overview")
    first_plot.basis = 'xy'
    first_plot.pixels = (4000, 4000)
    first_plot.color_by = 'material'
    first_plot.colors = colors
    first_plot.background= (0,0,0)
    first_plot.show_overlaps = True
    first_plot.overlap_color = 'red'

    second_plot = openmc.Plot(name="XY detail")
    second_plot.basis = 'xy'
    second_plot.pixels = (4000, 4000)
    second_plot.color_by = 'material'
    second_plot.colors = colors
    second_plot.background= (0,0,0)
    second_plot.show_overlaps = True
    second_plot.overlap_color = 'red'

    third_plot = openmc.Plot(name="YZ overview")
    third_plot.basis = 'yz'
    third_plot.pixels = (4000, 4000)
    third_plot.color_by = 'material'
    third_plot.colors = colors
    third_plot.background= (0,0,0)
    third_plot.show_overlaps = True
    third_plot.overlap_color = 'red'

    fourth_plot = openmc.Plot(name="YZ detail")
    fourth_plot.basis = 'yz'
    fourth_plot.pixels = (4000, 4000)
    fourth_plot.color_by = 'material'
    fourth_plot.colors = colors
    fourth_plot.background= (0,0,0)
    fourth_plot.show_overlaps = True
    fourth_plot.overlap_color = 'red'

    return first_plot,second_plot,third_plot,fourth_plot


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

if __name__ == "__main__":
    print("Import this module from fuel_element.py or reactor_geometry.py to create OpenMC plot sets.")