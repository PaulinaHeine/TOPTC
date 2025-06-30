from datetime import datetime, timedelta
import numpy as np
import logging;


logger = logging.getLogger(__name__)
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from data.Patches.patch_composition import generate_random_patch, generate_test_patch
import math
from datetime import datetime
import random
import math


import cartopy.crs as ccrs
import matplotlib
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['legend.scatterpoints'] = 1
matplotlib.rcParams['figure.autolayout'] = True
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.path import Path


# compare geht aber nicht große punkte
def animation_custom(
              model=None,
              buffer=.2,
              corners=None,
              filename=None,
              compare=None,
              compare_marker='v',
              background=None,
              bgalpha=.5,
              vmin=None,
              vmax=None,
              drifter=None,
              skip=None,
              scale=None,
              color=False,
              size=False,
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

    if model.history is not None and model.num_elements_total(
    ) == 0 and not hasattr(model, 'ds'):
        raise ValueError('Please run simulation before animating')

    if compare.history is not None and compare.num_elements_total(
    ) == 0 and not hasattr(compare, 'ds'):
        raise ValueError('Please run simulation before animating')



    if compare is not None:
        compare_list, compare_args = model._get_comparison_xy_for_plots(
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

    markercolor = model.plot_comparison_colors[0]

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
        if hasattr(model, 'ds'):  # opened with Xarray
            if origin_marker is None:
                origin_marker = 0
                per_origin_marker = False
            else:
                per_origin_marker = True
            H, H_om, lon_array, lat_array = model.get_density_xarray(
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
                model.get_density_array(pixelsize_m=density_pixelsize_m,
                                       weight=density_weight)
            H = H + H_submerged + H_stranded

    # Find map coordinates and plot points with empty data
    fig, ax, crs, x, y, index_of_first, index_of_last = \
        model.set_up_map(buffer=buffer, corners=corners, lscale=lscale,
                        fast=fast, hide_landmask=hide_landmask, **kwargs)

    # Anzeigeelemente initialisieren (Zeit + Bootswerte)
    text = ax.text(1.18, 1.0, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')



    gcrs = ccrs.PlateCarree(globe=crs.globe)



    from datetime import timedelta
    times = [
        model.start_time + timedelta(seconds=float(e['age']))
        for e in model.get_structured_history()[0]
    ]

    if show_elements is True:
        index_of_last_deactivated = \
            index_of_last[model.elements_deactivated.ID - 1]

    if (color is not False or size is not False) and show_elements is True:
        records = model.get_structured_history()

        if color is not False:
            colorarray = np.stack([patch[color] for patch in records], axis=0) * unitfactor
            colorarray_deactivated = np.full_like(colorarray, np.nan)
            if colorarray.size == 0:
                raise ValueError("colorarray ist leer – keine Daten zum Visualisieren.")
            if vmin is None:
                vmin = np.nanmin(colorarray)
            if vmax is None:
                vmax = np.nanmax(colorarray)
                vmax = np.nanmax(colorarray)

        if size is not False:
            if size not in records[0].dtype.names:
                raise ValueError(f"Größenattribut '{size}' nicht im Record gefunden.")
            sizearray = np.stack([patch[size] for patch in records], axis=0) * unitfactor
            if vmin is None:
                vmin_size = np.nanmin(sizearray)
            else:
                vmin_size = vmin
            if vmax is None:
                vmax_size = np.nanmax(sizearray)
            else:
                vmax_size = vmax

    if surface_only is True:
        z = model.get_property('z')[0]
        x[z < 0] = np.nan
        y[z < 0] = np.nan

    if show_trajectories is True:
        ax.plot(x, y, color='gray', alpha=trajectory_alpha, transform=gcrs)

    c = markercolor if color is False else []

    points = ax.scatter([], [],
                        c=c,
                        zorder=10,
                        edgecolor=[],
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        s=[],
                        label=legend[0] if legend else '',
                        transform=gcrs)

    points_deactivated = ax.scatter([], [],
                                    c=c,
                                    zorder=9,
                                    vmin=vmin,
                                    vmax=vmax,
                                    s=[],
                                    cmap=cmap,
                                    edgecolor=[],
                                    alpha=.3,
                                    transform=gcrs)

    x_deactive, y_deactive = (model.elements_deactivated.lon,
                              model.elements_deactivated.lat)

    if compare is not None:

        for cd in compare_list:
            cd['points_other'] = ax.scatter([], [],
                                            c=model.plot_comparison_colors[1],
                                            s=[],
                                            alpha=0.7,
                                            marker=compare_marker,
                                            transform=gcrs,
                                            label='Boats')

    def plot_timestep(i):
        """Sub function needed for matplotlib animation."""

        ret = [points, points_deactivated]  # list of elements to return for blitting

        if title == 'auto':
            ax.set_title('%s\n%s UTC' % (model._figure_title(), times[i]))
        else:
            ax.set_title('%s\n%s UTC' % (title, times[i]))

        # Zeit- und Value-Anzeige stündlich aktualisieren
        #if i % 60 == 0:
        elapsed = model.start_time + timedelta(seconds=i * model.time_step.total_seconds()) - model.start_time
        hours = int(elapsed.total_seconds() // 3600)
        #minutes = int((elapsed.total_seconds() % 3600) // 60)
        days = int(elapsed.total_seconds() // 86400)


        # Gesamtwert aller Boote anzeigen (falls in compare enthalten)
        if compare is not None and len(compare_list) > 0:

            collected = compare.history['collected_value'] # _total
            #collected_currrent = compare.history['collected_value_current']

            #print(collected)
            # Prüfe Zeitindex
            if isinstance(collected, np.ndarray) and i < collected.shape[1]:
                # Pro Boot und gesamt summieren
                per_boat = collected[:, i]
                #per_boat_current = collected_currrent[:, i]
                total_value = float(np.sum(per_boat))

                # Zusammenbauen des Anzeigetexts

                summary_lines = [f"Boot {idx + 1}: {v:.1f}" for idx, v in enumerate(per_boat)]
                #summary_lines.extend(f"Boot current {idx + 1}: {v:.1f}" for idx, v in enumerate(per_boat_current))
                summary_lines.append(f"Gesamt: {total_value:.1f}")
                summary_text = "\n".join(summary_lines)

                text.set_text(f"Hours since start {hours:02d}\nDays since start {days:02d} \n{summary_text}")

        if background is not None:
            ret.append(bg)
            if isinstance(background, xr.DataArray):
                scalar = background[i, :, :].values
            else:
                map_x, map_y, scalar, u_component, v_component = \
                    model.get_map_background(ax, background, crs,
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

            if size is not False:
                values_size = sizearray[:, i]
                min_size, max_size = 10, 100
                normed_sizes = (values_size - vmin_size+ 1e-6) / (vmax_size - vmin_size)
                size_scaled = min_size + normed_sizes * (max_size - min_size)
                points.set_sizes(size_scaled)

            if color is not False:
                values_color = colorarray[:, i]
                points.set_array(values_color)
                if isinstance(color, str) or hasattr(color, '__len__'):
                    points_deactivated.set_array(colorarray_deactivated[
                                                     index_of_last_deactivated < i])



            if markersizebymass:
                if 'chemicaldrift' in model.__module__:
                    points.set_sizes(
                        markersize * (model.history['mass'][:, i] /
                                      (model.history['mass'][:, i] +
                                       model.history['mass_degraded'][:, i] +
                                       model.history['mass_volatilized'][:, i])))
                elif 'openoil' in model.__module__:
                    points.set_sizes(
                        markersize * (model.history['mass_oil'][:, i] /
                                      (model.history['mass_oil'][:, i] +
                                       model.history['mass_biodegraded'][:, i] +
                                       model.history['mass_dispersed'][:, i] +
                                       model.history['mass_evaporated'][:, i])))

            if color is not False:  # Update colors
                points.set_array(colorarray[:,i])  # shape (particles,) == rank 1


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
        z = model.get_property('z')[0]
        x[z < 0] = np.nan
        y[z < 0] = np.nan

    if show_trajectories is True:
        ax.plot(x, y, color='gray', alpha=trajectory_alpha, transform=gcrs)

    if color is not False and show_elements is True:
        records = model.get_structured_history()


        colorarray = np.stack([patch[color] for patch in records], axis=0) * unitfactor
        colorarray_deactivated = np.full_like(colorarray, np.nan)

        if colorarray.size == 0:
            raise ValueError("colorarray ist leer – keine Daten zum Visualisieren.")

        if vmin is None:
            vmin = np.nanmin(colorarray)
        if vmax is None:
            vmax = np.nanmax(colorarray)


    if background is not None:
        if isinstance(background, xr.DataArray):
            map_x = background.coords['lon_bin']
            map_y = background.coords['lat_bin']
            scalar = background[0, :, :]
            map_y, map_x = np.meshgrid(map_y, map_x)
        else:
            map_x, map_y, scalar, u_component, v_component = \
                model.get_map_background(ax, background, crs,
                                        time=model.start_time)
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

    from datetime import timedelta
    times = [
        model.start_time + timedelta(seconds=float(e['age']))
        for e in model.get_structured_history()[0]
    ]
    if show_elements is True:
        index_of_last_deactivated = \
            index_of_last[model.elements_deactivated.ID-1]
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

    x_deactive, y_deactive = (model.elements_deactivated.lon,
                              model.elements_deactivated.lat)

    if compare is not None:
        for cn, cd in enumerate(compare_list):
            if legend != ['']:
                legstr = legend[cn + 1]
            else:
                legstr = None
            if color is False:
                c = model.plot_comparison_colors[cn+1]
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

    model.__save_or_plot_animation__(plt.gcf(),
                                    plot_timestep,
                                    filename,
                                    frames,
                                    fps,
                                    interval=50,
                                    blit = blit)



    logger.info('Time to make animation: %s' %
                (datetime.now() - start_time))




