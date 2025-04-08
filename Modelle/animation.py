def animation(self,
              buffer=.2,
              corners=None,
              filename=None,
              compare=None,
              compare_marker='o',
              background=None,
              alpha=1,
              bgalpha=.5,
              vmin=None,
              vmax=None,
              drifter=None,
              shapefiles=None,
              skip=None,
              scale=None,
              color=False,
              clabel=None,
              colorbar=True,
              cmap=None,
              density=False,
              show_elements=True,
              show_trajectories=False,
              linewidth=1,
              trajectory_alpha=.1,
              hide_landmask=False,
              density_pixelsize_m=1000,
              unitfactor=1,
              lcs=None,
              surface_only=False,
              markersize=20,
              markersize_scaling=None,
              origin_marker=None,
              legend=None,
              legend_loc='best',
              title='auto',
              fps=8,
              lscale=None,
              fast=False,
              blit=False,
              frames=None,
              xlocs=None,
              ylocs=None,
              **kwargs):
    """Animate last run."""

    filename = str(filename) if filename is not None else None

    if self.result is not None and self.num_elements_total(
    ) == 0 and not hasattr(self, 'ds'):
        raise ValueError('Please run simulation before animating')

    if compare is not None:
        compare_list, compare_args = self._get_comparison_xy_for_plots(
            compare)
        kwargs.update(compare_args)

    start_time = datetime.now()
    if cmap is None:
        cmap = 'jet'
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]

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
                        fast=fast, hide_landmask=hide_landmask, xlocs=xlocs, ylocs=ylocs, **kwargs)
