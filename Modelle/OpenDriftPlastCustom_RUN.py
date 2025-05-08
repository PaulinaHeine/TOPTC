from datetime import datetime, timedelta
import numpy as np
import logging; logger = logging.getLogger(__name__)
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from data.Patches.patch_composition import generate_random_patch, generate_test_patch
import math
from datetime import datetime

import random

import cartopy.crs as ccrs
import matplotlib
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['legend.scatterpoints'] = 1
matplotlib.rcParams['figure.autolayout'] = True
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.path import Path

import math

def offset_point(lat, lon, offset_m=100):
    """Verschiebt (lat, lon) um festen Abstand offset_m in zufälliger Richtung."""
    R = 6371000.0
    delta_deg = offset_m / 111000.0
    bearing = random.uniform(0, 2 * math.pi)
    new_lat = lat + delta_deg * math.cos(bearing)
    new_lon = lon + delta_deg * math.sin(bearing) / math.cos(math.radians(lat))
    return new_lat, new_lon


class Lagrangian3DArray(LagrangianArray):
    variables = LagrangianArray.add_variables([
        ('current_drift_factor', {'dtype': np.float32, 'units': '1', 'description': 'Drift factor for currents', 'default': 1.}),
        ('density', {'dtype': np.float32, 'units': 'kg/m^3', 'description': 'Density'}),
        ('weight', {'dtype': np.float32, 'units': 'kg', 'description': 'Total weight of patch'}),
        ('area', {'dtype': np.float32, 'units': 'm^2', 'description': 'Total area of patch'}),
        ('drag_coefficient', {'dtype': np.float32, 'units': '1', 'description': 'Hydrodynamic drag coefficient', 'default': 1.0}),
        ('surface_area_ratio', {'dtype': np.float32, 'units': '1', 'description': 'Surface area to volume ratio', 'default': 1.0}),
        ('markersize', {'dtype': np.float32, 'units': '1', 'description': 'Size for plotting', 'default': 20.0}),
        ('value', {'dtype': np.float32, 'units': '1', 'description': 'Reward value of patch', 'default': 1.0}),


    ])

class OpenDriftPlastCustom(OpenDriftSimulation):
    ElementType = Lagrangian3DArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0}
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._add_config({
            'drift:stokes_drift': {'type': 'bool', 'default': False,
                                   'description': 'Use Stokes drift',
                                   'level': 3},
            'drift:current_drift_factor': {'type': 'float', 'default': 1.0,
                                           'min': 0.0, 'max': 5.0,
                                           'units': '1',
                                           'description': 'Factor to scale ocean current drift',
                                           'level': 3},
        })

    def seed_plastic_patch(self, lon, lat, time, number=1, radius_km=5, z = 1):
        for _ in range(number):
            patch = generate_test_patch() #test
            props = patch['properties']

            logger.info(f"Seeding patch with properties: {props}")

            km_to_deg = 1.0 / 111.0
            rand_lat = lat + (random.uniform(-1, 1) * radius_km * km_to_deg)
            rand_lon = lon + (random.uniform(-1, 1) * radius_km * km_to_deg / np.cos(np.radians(lat)))

            density_factor = max(0.01, (1.025 - props['patch_density']) / 1.025)
            area_factor = props['patch_area'] / 10.0
            weight_factor = 1.0 / (1.0 + props['patch_weight'])

            drift_factor = np.sqrt(density_factor * area_factor * weight_factor)
            drift_factor = np.clip(drift_factor, 0.01, 0.2)

            drag_coefficient = 0.47 * (1.0 + 0.5 * props['patch_density'])
            drag_coefficient = np.clip(drag_coefficient, 0.1, 2.0)

            surface_area_ratio = props['patch_area'] / max(0.01, props['patch_weight'])
            markersize = np.clip(props['patch_area'] * 100, 10, 300)

            value =(props['patch_area'] * props['patch_density'] * props['patch_weight'])

            self.seed_elements(
                lon=rand_lon, lat=rand_lat, time=time, number=1,
                current_drift_factor=drift_factor,
                density=props['patch_density'],
                weight=props['patch_weight'],
                area=props['patch_area'],
                z=z, #immer mit depth gleichsetzen
                value= value,
                drag_coefficient=drag_coefficient,
                surface_area_ratio=surface_area_ratio,
                markersize=markersize
            )

            # Nur bei stepwise aktivieren
            # self.release_elements()



    def update(self):

        super().update()

        self.environment = self.get_environment(
                                variables=['x_sea_water_velocity', 'y_sea_water_velocity'],
                                time=self.time,           # aktueller Zeitpunkt
                                lon=self.elements.lon,    # aktuelle Längengrade
                                lat=self.elements.lat,      # aktuelle Breitengrade
                                z = self.elements.z,
                                profiles=None
                                )[0]
        self.advect_ocean_current()
        self.merge_close_patches()

    def advect_ocean_current(self):
        u_rel = self.environment.x_sea_water_velocity * self.elements.current_drift_factor
        v_rel = self.environment.y_sea_water_velocity * self.elements.current_drift_factor
        speed = np.sqrt(u_rel**2 + v_rel**2)
        max_speed = 0.05
        scale = np.clip(max_speed / (speed + 1e-8), 0, 1.0)
        u = u_rel * scale
        v = v_rel * scale
        if len(u_rel) == 0:
            logger.warning("⚠️ Keine Strömungsdaten vorhanden – Bewegung wird übersprungen.")
            return
        self.update_positions(u, v)


    def merge_close_patches(self, threshold_km=0.1):# TODO soll abhängig von der größe sein
        threshold_deg = threshold_km / 111.0
        positions = np.vstack([self.elements.lat, self.elements.lon]).T
        merged = set()

        for i in range(len(positions)):
            if i in merged:
                continue
            for j in range(i + 1, len(positions)):
                if j in merged:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold_deg:
                    self.elements.weight[i] += self.elements.weight[j]
                    self.elements.area[i] += self.elements.area[j]
                    self.elements.current_drift_factor[i] = (self.elements.current_drift_factor[i] + self.elements.current_drift_factor[j]) / 2
                    self.elements.density[i] = (self.elements.density[i] + self.elements.density[j]) / 2
                    self.elements.drag_coefficient[i] = (self.elements.drag_coefficient[i] + self.elements.drag_coefficient[j]) / 2
                    self.elements.surface_area_ratio[i] = (self.elements.surface_area_ratio[i] + self.elements.surface_area_ratio[j]) / 2
                    self.elements.lat[j] = np.nan
                    self.elements.lon[j] = np.nan
                    merged.add(j)

        if merged:
            self.deactivate_elements(np.isnan(self.elements.lat))

    def remove_large_patches_randomly(self):
        """
        Löscht zufällig große/schwere Patches ohne Ersatz.
        Kein Split, kein Seeding, nur Entfernung.
        """
        if self.num_elements_active() == 0:
            return

        base_removal_probability = 0.001
        area_threshold = 100.0
        weight_threshold = 300.0

        removed = 0

        for i in range(self.num_elements_active()):
            weight = self.elements.weight[i]
            area = self.elements.area[i]

            if area < area_threshold and weight < weight_threshold:
                continue

            if np.random.rand() > base_removal_probability:
                continue

            # Markiere Patch zum Entfernen
            self.elements.lat[i] = np.nan
            self.elements.lon[i] = np.nan
            removed += 1

            logger.info(f"Patch {i} entfernt (area={area:.1f}, weight={weight:.1f})")

        if removed > 0:
            self.deactivate_elements(np.isnan(self.elements.lat))
            logger.info(f"{removed} große Patches wurden zufällig entfernt.")


    def animation_custom(self,
                  buffer=.2,
                  corners=None,
                  filename=None,
                  compare=None,
                  compare_marker='o',
                  background=None,
                  bgalpha=.5,
                  vmin=None,
                  vmax=None,
                  drifter=None,
                  skip=None,
                  scale=None,
                  color=False,
                  clabel=None,
                  colorbar=True,
                  cmap=None,
                  density=False,
                  show_elements=True,
                  show_trajectories=False,
                  trajectory_alpha=.1,
                  hide_landmask=False,
                  density_pixelsize_m=1000,
                  unitfactor=1,
                  lcs=None,
                  surface_only=False,
                  markersize=20,
                  origin_marker=None,
                  legend=None,
                  legend_loc='best',
                  title='auto',
                  fps=8,
                  lscale=None,
                  fast=False,
                  blit=False,
                  **kwargs):
        """Animate last run."""

        filename = str(filename) if filename is not None else None

        if self.history is not None and self.num_elements_total(
        ) == 0 and not hasattr(self, 'ds'):
            raise ValueError('Please run simulation before animating')

        if compare is not None:
            compare_list, compare_args = self._get_comparison_xy_for_plots(
                compare)
            kwargs.update(compare_args)

        markersizebymass = False
        if isinstance(markersize, str):
            if markersize.startswith('mass'):
                markersizebymass = True
                if markersize[len('mass'):] == '':
                    # default initial size if not specified
                    markersize = 100
                else:
                    markersize = int(markersize[len('mass'):])

        start_time = datetime.now()
        if cmap is None:
            cmap = 'jet'
        if isinstance(cmap, str):
            cmap = matplotlib.cm.get_cmap(cmap)

        if color is False and background is None and lcs is None and density is False:
            colorbar = False

        markercolor = self.plot_comparison_colors[0]

        if isinstance(density, str):
            # Density field is weighted by this variable
            # TODO: not yet implemented!
            density_weight = density
            density = True
        else:
            if density is True:
                density_weight = None
            elif density is not False:
                density_weight = density
                density = True
        if density is True:  # Get density arrays
            if hasattr(self, 'ds'):  # opened with Xarray
                if origin_marker is None:
                    origin_marker = 0
                    per_origin_marker = False
                else:
                    per_origin_marker = True
                H, H_om, lon_array, lat_array = self.get_density_xarray(
                    pixelsize_m=density_pixelsize_m, weights=density_weight)
                if per_origin_marker is True:
                    H = H_om[:, :, :, origin_marker]
            else:
                if origin_marker is not None:
                    raise ValueError(
                        'Separation by origin_marker is only active when imported from file with '
                        'open_xarray: https://opendrift.github.io/gallery/example_huge_output.html'
                    )
                H, H_submerged, H_stranded, lon_array, lat_array = \
                    self.get_density_array(pixelsize_m=density_pixelsize_m,
                                           weight=density_weight)
                H = H + H_submerged + H_stranded

        # Find map coordinates and plot points with empty data
        fig, ax, crs, x, y, index_of_first, index_of_last = \
            self.set_up_map(buffer=buffer, corners=corners, lscale=lscale,
                            fast=fast, hide_landmask=hide_landmask, **kwargs)

        gcrs = ccrs.PlateCarree(globe=crs.globe)

        def plot_timestep(i):
            """Sub function needed for matplotlib animation."""

            ret = [points, points_deactivated]  # list of elements to return for blitting
            if title == 'auto':
                ax.set_title('%s\n%s UTC' % (self._figure_title(), times[i]))
            else:
                ax.set_title('%s\n%s UTC' % (title, times[i]))
            if background is not None:
                ret.append(bg)
                if isinstance(background, xr.DataArray):
                    scalar = background[i, :, :].values
                else:
                    map_x, map_y, scalar, u_component, v_component = \
                        self.get_map_background(ax, background, crs,
                                                time=times[i])
                # https://stackoverflow.com/questions/18797175/animation-with-pcolormesh-routine-in-matplotlib-how-do-i-initialize-the-data
                bg.set_array(scalar.ravel())
                if type(background) is list:
                    ret.append(bg_quiv)
                    bg_quiv.set_UVC(u_component[::skip, ::skip],
                                    v_component[::skip, ::skip])

            if lcs is not None:
                ax.pcolormesh(lcs['lon'],
                              lcs['lat'],
                              lcs['ALCS'][i, :, :],
                              alpha=bgalpha,
                              vmin=vmin,
                              vmax=vmax,
                              cmap=cmap,
                              transform=gcrs)

            if density is True:
                # Update density plot
                pm.set_array(H[i, :, :].ravel())
                ret.append(pm)

            # Move points
            if show_elements is True:
                points.set_offsets(np.c_[x[i, range(x.shape[1])],
                                         y[i, range(x.shape[1])]])
                points_deactivated.set_offsets(
                    np.c_[x_deactive[index_of_last_deactivated < i],
                          y_deactive[index_of_last_deactivated < i]])

                if markersizebymass:
                    if 'chemicaldrift' in self.__module__:
                        points.set_sizes(
                            markersize * (self.history['mass'][:, i] /
                                          (self.history['mass'][:, i] +
                                           self.history['mass_degraded'][:, i] +
                                           self.history['mass_volatilized'][:, i])))
                    elif 'openoil' in self.__module__:
                        points.set_sizes(
                            markersize * (self.history['mass_oil'][:, i] /
                                          (self.history['mass_oil'][:, i] +
                                           self.history['mass_biodegraded'][:, i] +
                                           self.history['mass_dispersed'][:, i] +
                                           self.history['mass_evaporated'][:, i])))

                if color is not False:  # Update colors
                    points.set_array(colorarray[:, i])
                    if compare is not None:
                        for cd in compare_list:
                            cd['points_other'].set_array(colorarray[:, i])
                    if isinstance(color, str) or hasattr(color, '__len__'):
                        points_deactivated.set_array(colorarray_deactivated[
                            index_of_last_deactivated < i])

            if drifter is not None:
                for drnum, dr in enumerate(drifter):
                    drifter_pos[drnum].set_offsets(np.c_[dr['x'][i],
                                                         dr['y'][i]])
                    drifter_line[drnum].set_data(dr['x'][0:i], dr['y'][0:i])
                    ret.append(drifter_line[drnum])
                    ret.append(drifter_pos[drnum])

            if show_elements is True:
                if compare is not None:
                    for cd in compare_list:
                        cd['points_other'].set_offsets(
                            np.c_[cd['x_other'][range(cd['x_other'].shape[0]),
                                                i],
                                  cd['y_other'][range(cd['x_other'].shape[0]),
                                                i]])
                        cd['points_other_deactivated'].set_offsets(np.c_[
                            cd['x_other_deactive'][
                                cd['index_of_last_deactivated_other'] < i],
                            cd['y_other_deactive'][
                                cd['index_of_last_deactivated_other'] < i]])
                        ret.append(cd['points_other'])
                        ret.append(cd['points_other_deactivated'])

            return ret

        if surface_only is True:
            z = self.get_property('z')[0]
            x[z < 0] = np.nan
            y[z < 0] = np.nan

        if show_trajectories is True:
            ax.plot(x, y, color='gray', alpha=trajectory_alpha, transform=gcrs)

        if color is not False and show_elements is True:
            if isinstance(color, str):
                colorarray = self.get_property(color)[0].T
                colorarray = colorarray * unitfactor
                colorarray_deactivated = \
                    self.get_property(color)[0][
                        index_of_last[self.elements_deactivated.ID-1],
                                      self.elements_deactivated.ID-1].T
            elif hasattr(color,
                         '__len__'):  # E.g. array/list of ensemble numbers
                colorarray_deactivated = color[self.elements_deactivated.ID -
                                               1]
                colorarray = np.tile(color, (self.steps_output, 1)).T
            else:
                colorarray = color
            if vmin is None:
                vmin = colorarray.min()
                vmax = colorarray.max()

        if background is not None:
            if isinstance(background, xr.DataArray):
                map_x = background.coords['lon_bin']
                map_y = background.coords['lat_bin']
                scalar = background[0, :, :]
                map_y, map_x = np.meshgrid(map_y, map_x)
            else:
                map_x, map_y, scalar, u_component, v_component = \
                    self.get_map_background(ax, background, crs,
                                            time=self.start_time)
            bg = ax.pcolormesh(map_x,
                               map_y,
                               scalar,
                               alpha=bgalpha,
                               zorder=1,
                               antialiased=True,
                               linewidth=0.0,
                               rasterized=True,
                               vmin=vmin,
                               vmax=vmax,
                               cmap=cmap,
                               transform=gcrs)
            if type(background) is list:
                bg_quiv = ax.quiver(map_x[::skip, ::skip],
                                    map_y[::skip, ::skip],
                                    u_component[::skip, ::skip],
                                    v_component[::skip, ::skip],
                                    scale=scale,
                                    zorder=1,
                                    transform=gcrs)

        if lcs is not None:
            if vmin is None:
                vmin = lcs['ALCS'].min()
                vmax = lcs['ALCS'].max()
            lcsh = ax.pcolormesh(lcs['lon'],
                                 lcs['lat'],
                                 lcs['ALCS'][0, :, :],
                                 vmin=vmin,
                                 vmax=vmax,
                                 cmap=cmap,
                                 transform=gcrs)

        times = self.get_time_array()[0]
        if show_elements is True:
            index_of_last_deactivated = \
                index_of_last[self.elements_deactivated.ID-1]
        if legend is None:
            legend = ['']

        if color is False:
            c = markercolor
        else:
            c = []

        if markersizebymass:
            points = ax.scatter([], [],
                                c=c,
                                zorder=10,
                                edgecolor=[],
                                cmap=cmap,
                                alpha=.4,
                                vmin=vmin,
                                vmax=vmax,
                                label=legend[0],
                                transform=gcrs)
        else:
            points = ax.scatter([], [],
                                c=c,
                                zorder=10,
                                edgecolor=[],
                                cmap=cmap,
                                s=markersize,
                                vmin=vmin,
                                vmax=vmax,
                                label=legend[0],
                                transform=gcrs)

        if (compare is None) and (legend != ['']):
            markers = []
            for legend_index in np.arange(len(legend)):
                if legend[legend_index] != '':
                    markers.append(
                        matplotlib.lines.Line2D(
                            [0], [0],
                            marker='o',
                            color='w',
                            linewidth=0,
                            markeredgewidth=0,
                            markerfacecolor=cmap(legend_index / (len(legend) - 1)),
                            markersize=10,
                            label=legend[legend_index]))
            legend=list(filter(None, legend))
            ax.legend(markers, legend, loc=legend_loc)

        # Plot deactivated elements, with transparency
        if markersizebymass:
            points_deactivated = ax.scatter([], [],
                                            c=c,
                                            zorder=9,
                                            vmin=vmin,
                                            vmax=vmax,
                                            s=markersize,
                                            cmap=cmap,
                                            edgecolor=[],
                                            alpha=0,
                                            transform=gcrs)
        else:
            points_deactivated = ax.scatter([], [],
                                            c=c,
                                            zorder=9,
                                            vmin=vmin,
                                            vmax=vmax,
                                            s=markersize,
                                            cmap=cmap,
                                            edgecolor=[],
                                            alpha=.3,
                                            transform=gcrs)

        x_deactive, y_deactive = (self.elements_deactivated.lon,
                                  self.elements_deactivated.lat)

        if compare is not None:
            for cn, cd in enumerate(compare_list):
                if legend != ['']:
                    legstr = legend[cn + 1]
                else:
                    legstr = None
                if color is False:
                    c = self.plot_comparison_colors[cn+1]
                else:
                    c = []
                cd['points_other'] = \
                    ax.scatter([], [], c=c, marker=compare_marker, cmap=cmap,
                               s=markersize, label=legstr, zorder=10, transform = gcrs)
                # Plot deactivated elements, with transparency
                cd['points_other_deactivated'] = \
                    ax.scatter([], [], alpha=.3, zorder=9, marker=compare_marker, cmap=cmap,
                               c=c, s=markersize, transform = gcrs)

            if legend != ['', '']:
                plt.legend(markerscale=2, loc=legend_loc)

        if density is True:
            cmap.set_under('w')
            H = np.ma.masked_where(H == 0, H)
            lat_array, lon_array = np.meshgrid(lat_array, lon_array)
            if vmax is None:
                vmax = H.max()
            pm = ax.pcolormesh(lon_array,
                               lat_array,
                               H[0, :, :],
                               vmin=0.1,
                               vmax=vmax,
                               cmap=cmap,
                               transform=gcrs)

        if drifter is not None:
            if not isinstance(drifter, list):
                drifter = [drifter]
            drifter_pos = [None]*len(drifter)
            drifter_line = [None]*len(drifter)
            for drnum, dr in enumerate(drifter):
                # Interpolate drifter time series onto simulation times
                sts = np.array(
                    [t.total_seconds() for t in np.array(times) - times[0]])
                dts = np.array([
                    t.total_seconds() for t in np.array(dr['time']) - times[0]
                ])
                dr['x'] = np.interp(sts, dts, dr['lon'])
                dr['y'] = np.interp(sts, dts, dr['lat'])
                dr['x'][sts < dts[0]] = np.nan
                dr['x'][sts > dts[-1]] = np.nan
                dr['y'][sts < dts[0]] = np.nan
                dr['y'][sts > dts[-1]] = np.nan
                dlabel = dr['label'] if 'label' in dr else 'Drifter'
                dcolor = dr['color'] if 'color' in dr else 'r'
                dlinewidth = dr['linewidth'] if 'linewidth' in dr else 2
                dzorder = dr['zorder'] if 'zorder' in dr else 10
                dmarkersize = dr[
                    'markersize'] if 'markersize' in dr else 20
                drifter_pos[drnum] = ax.scatter([], [],
                                                c=dcolor,
                                                zorder=dzorder+1,
                                                s=dmarkersize,
                                                label=dlabel,
                                                transform=gcrs)
                drifter_line[drnum] = ax.plot([], [],
                    color=dcolor, linewidth=dlinewidth,
                    zorder=dzorder, transform=gcrs)[0]
                #ax.plot(dr['x'],
                #        dr['y'],
                #        color=dcolor,
                #        linewidth=dlinewidth,
                #        zorder=dzorder,
                #        transform=gcrs)
            plt.legend()

        fig.canvas.draw()
        fig.set_layout_engine('tight')
        if colorbar is True:
            if color is not False:
                if isinstance(color, str) or clabel is not None:
                    if clabel is None:
                        clabel = color
                item = points
            elif density is not False:
                item = pm
                if clabel is None:
                    clabel = 'density'
            elif lcs is not None:
                item = lcsh
                if clabel is None:
                    clabel = 'LCS'
            elif background is not None:
                item = bg
                if clabel is None:
                    if isinstance(background, xr.DataArray):
                        clabel = background.name
                    else:
                        clabel = background

            cb = fig.colorbar(item,
                              orientation='horizontal',
                              pad=.05,
                              aspect=30,
                              shrink=.8,
                              drawedges=False)
            cb.set_label(clabel)

        frames = x.shape[0]

        if compare is not None:
            frames = min(x.shape[0], cd['x_other'].shape[1])

        # blit is now provided to animation()
        #blit = sys.platform != 'darwin'  # blitting does not work on mac os

        self.__save_or_plot_animation__(plt.gcf(),
                                        plot_timestep,
                                        filename,
                                        frames,
                                        fps,
                                        interval=50,
                                        blit = blit)

        logger.info('Time to make animation: %s' %
                    (datetime.now() - start_time))

