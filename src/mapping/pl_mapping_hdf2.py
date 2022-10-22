import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets, interactive, GridspecLayout, Layout
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import h5py
from Cosmic_Filter import find_cosmics
from scipy import sparse
from scipy.sparse.linalg import spsolve


class Mapper:
    mask = []
    _is_interp = False

    def __init__(self, fname, sx=None, dsx=None, levels=10, interpolate=False, custom_wavelength=None):
        self.specx = sx
        self.dspecx = dsx

        self._read_hdf5_file(fname, interpolate, custom_wavelength)
        self.df = self.df[~self.df.index.duplicated(keep='first')]

        self.df_maximum = pd.DataFrame(index=self._midx,
                                       columns=['Intensity'])
        self.df_integrated = pd.DataFrame(index=self._midx,
                                          columns=['Intensity'])
        self.df_wno = pd.DataFrame(index=self._midx,
                                   columns=['Intensity'])

        self.selected_point = self.df.index.drop_duplicates()[0]

        # Set slider values if not provided
        if sx is None:
            self.specx = self.df.columns[0]
        if dsx is None:
            self.dspecx = 50 * abs(self.df.columns[1] - self.df.columns[0])

        self.i_max = self.df.max().max()
        self.i_min = self.df.min().min()

        self.levels = levels
        self.energy_axis = self.df.columns
        self.x = self.df.index.get_level_values(0).drop_duplicates().sort_values()
        self.y = self.df.index.get_level_values(1).drop_duplicates().sort_values()
        self.selected_point = self.df.index.drop_duplicates()[0]

    def _read_hdf5_file(self, fname, interpolate=False, custom_wavelength=None):
        f = h5py.File(fname, 'r')

        measurement_index = list(f.keys())[0]
        x = list(f[measurement_index].keys())
        if 'Background' in x: 
            x.remove('Background')
        x.remove('Wavelength')
        y = list(f['{:}/{:}'.format(measurement_index, x[0])].keys())
        xm, ym = np.meshgrid(x, y, indexing='ij')

        if custom_wavelength is None:
            data = pd.DataFrame(index=pd.MultiIndex.from_arrays([xm.flatten(), ym.flatten()], names=['x', 'y']),
                                columns=f['{:}/Wavelength'.format(measurement_index)])
        else:
            data = pd.DataFrame(index=pd.MultiIndex.from_arrays([xm.flatten(), ym.flatten()], names=['x', 'y']),
                                columns=custom_wavelength)
        data.columns.name = "Wavelength"

        sensX = []
        sensY = []
        for x, y in data.index:
            dataset = f['{:}/{:}/{:}/Spectrum'.format(measurement_index, x, y)]
            sensX.append(dataset.attrs['Sensor X'])
            sensY.append(dataset.attrs['Sensor Y'])

            data.loc[x, y] = np.array(dataset)

        # Convert x, y to float
        data.index = data.index.set_levels([data.index.levels[0].astype(float), data.index.levels[1].astype(float)])
        self._midx = data.index.copy()

        # Sort Wavelengths
        data = data.reindex(sorted(data.columns), axis=1)

        if interpolate:
            data.index = pd.MultiIndex.from_arrays([sensX, sensY], names=['x', 'y'])
            self._is_interp = True

        f.close()

        self.df = data

    def plot(self):
        self.specx_slider = widgets.FloatSlider(value=self.specx,
                                                min=self.energy_axis.min(),
                                                max=self.energy_axis.max(),
                                                step=abs(self.energy_axis[1] - self.energy_axis[0]),
                                                description='λ',
                                                continous_update=False)

        self.dspecx_slider = widgets.FloatSlider(value=self.dspecx,
                                                 min=0.2,
                                                 max=(self.energy_axis.max() - self.energy_axis.min()) / 2,
                                                 step=abs(self.energy_axis[1] - self.energy_axis[0]),
                                                 description='Δλ',
                                                 continous_update=False)

        self.recalc_button = widgets.Button(description="Recalculate",
                                            disabled=False,
                                            layout=Layout(width='75%'),
                                            tooltip='Recalculate 2D Graph')

        self.recalc_checkbox = widgets.Checkbox(value=False,
                                                description='Auto',
                                                disabled=False)

        self.autoscale_button = widgets.Button(description="Rescale z-Axis",
                                               disabled=False,
                                               layout=Layout(width='75%'),
                                               tooltip='Autoscale 2D Graph z-Axis')

        self.autoscale_checkbox = widgets.Checkbox(value=True,
                                                   description='Auto',
                                                   disabled=False)


        self.background_button = widgets.Button(description="Select background point",
                                                disabled=False,
                                                layout=Layout(width='75%'),
                                                tooltip='Use current data point for background correction')

        self.mask_button = widgets.Button(description="Un/mask",
                                          disabled=False,
                                          layout=Layout(width='75%'),
                                          tooltip='Mask or unmask current data point')

        self.cosmic_select = widgets.Dropdown(
            options=['None', 'standard', 'non constant background', 'strong'],
            value='None',
            description='cosmic_filter',
            disabled=False)

        self.level_slider = widgets.IntSlider(value=self.levels,
                                              min=3,
                                              max=50,
                                              description='Lvls',
                                              disabled=False)

        self.contour_select = widgets.Dropdown(
            options=['Maximum', 'Integrated', 'Position'],
            value='Maximum',
            description='Display',
            disabled=False)

        self.background_select = widgets.Dropdown(
            options=['None', 'Mean', 'Point', 'y-Mean', 'als_baseline'],
            value='None',
            description='Background',
            disabled=False)

        self._calculate_2d()
        df_roi = self.get_roi()

        self.active_df = self.df_maximum

        self.i_max = self.active_df.max()
        self.i_min = self.active_df.min()

        self.i_max_slider = widgets.FloatSlider(value=self.i_max,
                                                min=self.i_min,
                                                max=self.i_max,
                                                step=(self.i_max-self.i_min)/50,
                                                readout_format='.3f',
                                                description='I_max',
                                                continous_update=False)

        self.i_min_slider = widgets.FloatSlider(value=self.i_min,
                                                min=self.i_min,
                                                max=self.i_max,
                                                step=(self.i_max-self.i_min)/50,
                                                readout_format='.3f',
                                                description='I_min',
                                                continous_update=False)

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)
        self.c = go.Contour(z=Mapper.reshape_df(self.active_df),
                            x=self.x,
                            y=self.y,
                            colorscale='Viridis',
                            colorbar=dict(x=-.2, len=1),
                            contours=dict(start=float(self.i_min),
                                          end=float(self.i_max),
                                          size=float(self.i_max - self.i_min)/self.levels))

        self.s = go.Scatter(x=df_roi.index, y=df_roi.values)
        self.s_corr = go.Scatter(x=df_roi.index, y=df_roi.values)
        self.s_peak = go.Scatter()

        fig.add_trace(self.c, 1, 1)
        fig.add_trace(self.s, 1, 2)
        fig.add_trace(self.s_peak, 1, 2)
        fig.update_xaxes(title_text="X pos (µm)", row=1, col=1)
        fig.update_yaxes(title_text="Y pos (µm)", row=1, col=1)
        fig.update_xaxes(title_text="Wavelength (nm)", row=1, col=2)
        fig.update_yaxes(title_text="Intensity (arb. units)", row=1, col=2)
        fig.update_layout(width=1000, height=500, autosize=True)

        self.g = go.FigureWidget(fig, layout=go.Layout(barmode='overlay'))
        self.g.data[0].name = 'λ={:.2f},Δλ={:.2f}'.format(self.specx, self.dspecx)
        self.g.data[1].name = 'x={:.2f},y={:.2f}'.format(self.selected_point[0], self.selected_point[1])
        self.g.data[2].name = 'Detected Peak'
        self.g.data[2].marker = dict(size=8, color='red', symbol='cross')
        self.g.add_shape(type='line', x0=self.specx - self.dspecx, xref='x2', yref='paper', y0=0,
                         x1=self.specx - self.dspecx, y1=1, line_width=3, name='l1')
        self.g.add_shape(type='line', x0=self.specx + self.dspecx, xref='x2', yref='paper', y0=0,
                         x1=self.specx + self.dspecx, y1=1, line_width=3, name='l2')

        self.specx_slider.observe(self._change_roi_slider, names="value")
        self.dspecx_slider.observe(self._change_roi_slider, names="value")
        self.i_max_slider.observe(self._change_contour_presentation, names="value")
        self.i_min_slider.observe(self._change_contour_presentation, names="value")
        self.contour_select.observe(self._redraw_contour, names="value")
        self.recalc_button.on_click(self._redraw_contour)
        self.autoscale_button.on_click(self._autoscale_contour)
        self.level_slider.observe(self._change_contour_presentation, names='value')
        self.g.data[0].on_click(self._change_xy)
        self.mask_button.on_click(self._mask_point)
        self.background_button.on_click(self._set_background_point)
        self.background_select.observe(self._redraw_contour, names='value')
        self.cosmic_select.observe(self.remove_cosmics, names='value')

        self.grid = GridspecLayout(5, 4)
        self.grid[0, 0] = self.contour_select
        self.grid[1, 0] = self.background_select
        self.grid[2, 0] = self.background_button
        self.grid[3, 0] = widgets.HBox([self.recalc_button, self.recalc_checkbox])
        self.grid[0, 1] = self.i_max_slider
        self.grid[1, 1] = self.i_min_slider
        self.grid[2, 1] = self.level_slider
        self.grid[3, 1] = widgets.HBox([self.autoscale_button, self.autoscale_checkbox])
        self.grid[0, 2] = self.specx_slider
        self.grid[1, 2] = self.dspecx_slider
        self.grid[2, 2] = self.cosmic_select
        self.grid[3, 2] = self.mask_button

        return widgets.VBox([self.grid, self.g])

    def update_graphics(self, change=None):
        df_roi = self.get_roi()
        xpos = Mapper.get_pos_of_max(pd.to_numeric(df_roi.loc[self.specx - self.dspecx:self.specx + self.dspecx]))
        ypos = Mapper.get_intensity(df_roi.loc[self.specx - self.dspecx:self.specx + self.dspecx])

        with self.g.batch_update():
            self.g.data[0].x = self.x
            self.g.data[0].y = self.y
            self.g.data[0].z = Mapper.reshape_df(self._apply_mask(self.active_df))
            self.g.data[0].name = 'ν={:.2f},Δν={:.2f}'.format(self.specx, self.dspecx)
            self.g.data[0].contours = dict(start=float(self.i_min_slider.value),
                                           end=float(self.i_max_slider.value),
                                           size=float(self.i_max_slider.value - self.i_min_slider.value)/self.levels)
            self.g.data[1].x = df_roi.index
            self.g.data[1].y = df_roi.values
            self.g.data[2].x = [xpos]
            self.g.data[2].y = [ypos]
            self.g.data[1].name = 'x={:.2f},y={:.2f}'.format(self.selected_point[0], self.selected_point[1])
            self.g.update_shapes(patch=dict(x0=self.specx - self.dspecx,
                                            x1=self.specx - self.dspecx),
                                 selector=dict(name="l1"))
            self.g.update_shapes(patch=dict(x0=self.specx + self.dspecx,
                                            x1=self.specx + self.dspecx),
                                 selector=dict(name="l2"))

    def _calculate_2d(self):
        if self.background_select.value == 'Mean':
            self.corrected_df = self.df - self.df.mean()
        elif self.background_select.value == 'Point' and self.bg_point != None:
            self.corrected_df = self.df - self.df.loc[self.bg_point[0], self.bg_point[1]]
        elif self.background_select.value == 'y-Mean':
            self.corrected_df = self.df.copy()
            for i in self.x:
                self.corrected_df.loc[(pd.IndexSlice[i, :]), :] = self.df.loc[(pd.IndexSlice[i, :]), :] - self.df.loc[(pd.IndexSlice[i, :]), :].mean()
        elif self.background_select.value == 'als_baseline':
            self.corrected_df = self.als_remove_baseline()
        else:
            self.corrected_df = self.df.copy()

        roi = self.corrected_df.loc[:, self.specx - self.dspecx:self.specx + self.dspecx]

        x, y = roi.index.get_level_values(0), roi.index.get_level_values(1)


        if self._is_interp:
            xi, yi = self._midx.get_level_values(0), self._midx.get_level_values(1)

            self.df_maximum = pd.DataFrame(griddata((x, y),
                                           roi.apply(lambda x: x.max(), axis=1).values,
                                           (xi, yi),
                                           method='linear'),
                                           index=self._midx).iloc[:, 0].sort_index()

            self.df_integrated = pd.DataFrame(griddata((x, y),
                                              roi.apply(lambda x: x.sum(), axis=1).values,
                                              (xi, yi),
                                              method='linear'),
                                              index=self._midx).iloc[:, 0].sort_index()

            self.df_wno = pd.DataFrame(griddata((x, y),
                                       roi.apply(lambda x: pd.to_numeric(x).idxmax(), axis=1).values,
                                       (xi, yi),
                                       method='linear'),
                                       index=self._midx).iloc[:, 0].sort_index()
        else:
            self.df_maximum = roi.apply(lambda x: x.max(), axis=1).sort_index()
            self.df_integrated = roi.apply(lambda x: x.sum(), axis=1).sort_index()
            self.df_wno = roi.apply(lambda x: pd.to_numeric(x).idxmax(), axis=1).sort_index()

        if self.contour_select.value == 'Maximum':
            self.active_df = self.df_maximum
        elif self.contour_select.value == 'Integrated':
            self.active_df = self.df_integrated
        elif self.contour_select.value == 'Position':
            self.active_df = self.df_wno

        self.i_max = self.active_df.max()
        self.i_min = self.active_df.min()

    def _change_xy(self, trace, points, selector):
        if len(points.xs) > 0 and len(points.ys) > 0:
            self.selected_point = (points.xs[0], points.ys[0])
            self.update_graphics()

    def _change_roi_slider(self, change=None):
        self.specx = self.specx_slider.value
        self.dspecx = self.dspecx_slider.value

        if self.recalc_checkbox.value:
            self._calculate_2d()
            self._update_min_max_sliders()
        self.update_graphics()

    def _change_contour_presentation(self, change=None):
        self.levels = self.level_slider.value
        self.update_graphics()

    def _redraw_contour(self, change=None):
        self._calculate_2d()
        self._update_min_max_sliders()
        self.update_graphics()

    def remove_cosmics(self, change=None, method_i = "everything", mask = "area", filter_length = 8, filter_size = 1, peak_to_noise = 1.5, signal_to_noise = 1.3):
        """
        find_cosmics finds and smoothens Cosmics from data.

        The function takes a data frame (pandas package) with 4 dimensions as standard input.
        The amount of data checked and edited by the function can be adjusted with wv_index.
        method_i and mask allow to adjust the essential way this filter works. filter_length,
        filter size, signal_to_noise and peak_to_noise are parameters to optimize the filter
        for different kinds of data.

        Parameters
        ----------
        method : str
            Chooses filter method, e.g. "spectral" or "area".
        mask : str
            Method of flatten, spacial or spectral.
        filter_length : int
            Dividable by 2! Defines size of spectral width, which is checked for signals.
        filter_size : int
            Defines size of spacial area checked.
        peak_to_noise : float
            What do I define as a Cosmic? Cosmic > (mean) * signal_to_noise
        signal_to_noise: float
            What do I define as a signal? signal > (mean) * signal_to_noise

        Returns
        -------
        None
        """
        if self.cosmic_select.value == "non constant background":
            method_i = "opt1"
        elif self.cosmic_select.value == "strong":
            method_i = "opt3"

        if self.cosmic_select.value != "None":
            self.df = find_cosmics(self.df, list(range(0, len(self.df.columns.values),)), method_i, mask, filter_length, filter_size, peak_to_noise, signal_to_noise)
            self._redraw_contour()

    def _update_min_max_sliders(self, change=None):
        buffer_max = self.i_max_slider.value
        buffer_min = self.i_min_slider.value

        self.i_max_slider = widgets.FloatSlider(value=buffer_max,
                                                min=self.i_min,
                                                max=self.i_max,
                                                step=(self.i_max-self.i_min)/50,
                                                readout_format='.3f',
                                                description='I_max',
                                                continous_update=False)

        self.i_min_slider = widgets.FloatSlider(value=buffer_min,
                                                min=self.i_min,
                                                max=self.i_max,
                                                step=(self.i_max-self.i_min)/50,
                                                readout_format='.3f',
                                                description='I_min',
                                                continous_update=False)

        if self.autoscale_checkbox.value:
            self.i_max_slider.value = self.i_max
            self.i_min_slider.value = self.i_min

        self.grid[0, 1] = self.i_max_slider
        self.grid[1, 1] = self.i_min_slider

        self.i_max_slider.observe(self._change_contour_presentation, names="value")
        self.i_min_slider.observe(self._change_contour_presentation, names="value")

    def _autoscale_contour(self, change=None):
        self._update_min_max_sliders()
        self.i_max_slider.value = self.i_max
        self.i_min_slider.value = self.i_min
        self.update_graphics()

    def _apply_mask(self, df):
        df_display = df.copy()
        for xdel, ydel in self.mask:
            df_display.loc[xdel, ydel] = np.nan

        return df_display

    def _mask_point(self, change=None):
        if self.selected_point in self.mask:
            self.mask.remove(self.selected_point)
        else:
            self.mask.append(self.selected_point)

        self._update_min_max_sliders()
        self.update_graphics()

    def _set_background_point(self, change=None):
        self.bg_point = self.selected_point
        self._redraw_contour()

    def get_roi(self):
        if self._is_interp:
            multidx = self.df.index.drop_duplicates()
            idx_pos = np.array(list(map(lambda c: np.sqrt((c[0] - self.selected_point[0]) ** 2 + (c[1] - self.selected_point[1]) ** 2),
                               multidx.to_numpy()))).argmin()

            self.selected_point = multidx[idx_pos]

        df_roi = self.corrected_df.loc[self.selected_point[0], self.selected_point[1]]
        return df_roi

    def get_dataframes(self):
        return self.df, self.df_maximum, self.df_integrated, self.df_wno

    def save_to_files(self, base_filename, append_info=False, unstacked=False):
        info_string = '_cwave={:.2f}_dcwave={:.2f}'.format(
            self.specx, self.dspecx)
        if append_info:
            base_filename += info_string

        if unstacked:
            self._apply_mask(self.df_maximum).unstack().to_csv(base_filename + '_maximum.csv')
            self._apply_mask(self.df_integrated).unstack().to_csv(base_filename + '_integrated.csv')
            self._apply_mask(self.df_wno).unstack().to_csv(base_filename + '_wavelength.csv')
        else:
            self._apply_mask(self.df_maximum).to_csv(base_filename + '_maximum.csv')
            self._apply_mask(self.df_integrated).to_csv(base_filename + '_integrated.csv')
            self._apply_mask(self.df_wno).to_csv(base_filename + '_wavelength.csv')

    def save_current_spectrum_to_file(self, filename):
        df_roi = self.get_roi()
        pos = ' (x={:},y={:})'.format(self.selected_point[0], self.selected_point[1])

        pd.DataFrame({'Raw' + pos: df_roi.values})\
            .set_index(df_roi.index)\
            .to_csv(filename)

    def reverse(self, revx, revy):
        self.df_integrated.index = pd.MultiIndex.from_arrays([Mapper.get_xindex(self.df_integrated)[::1 - 2 * int(revx)],
                                                              Mapper.get_yindex(self.df_integrated)[::1 - 2 * int(revy)]])

        self.df_wno.index = pd.MultiIndex.from_arrays([Mapper.get_xindex(self.df_wno)[::1 - 2 * int(revx)],
                                                       Mapper.get_yindex(self.df_wno)[::1 - 2 * int(revy)]])

    def revx(self):
        self.reverse(True, False)

    def revy(self):
        self.reverse(False, True)

    def als_remove_baseline(self, change=None):
        baseline_i = self.df.apply(lambda x: Mapper.baseline_als_optimized(list(x)), axis=1)
        baseline_i.set_axis(list(self.df.columns.values), axis=1, inplace=True)
        self.df = self.df - baseline_i
        return self.df

    @staticmethod
    def baseline_als_optimized(y, lam=10**8, p=0.1, niter=10):
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
        D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        for i in range(niter):
            W.setdiag(w)  # Do not create a new matrix, just update diagonal values
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return pd.Series(z)

    @staticmethod
    def reshape_df(df):
        x = df.index.get_level_values(0).nunique()
        y = df.index.get_level_values(1).nunique()

        if df.index.size != (x * y):
            fillarr = np.ones(abs(df.index.size - y * x)) * np.nan
            return np.append(df.values, fillarr).reshape(x, y, order='C').T
        else:
            return df.sort_index().values.reshape(x, y, order='C').T

    @staticmethod
    def get_xindex(df):
        return df.index.get_level_values(0)

    @staticmethod
    def get_yindex(df):
        return df.index.get_level_values(1)

    @staticmethod
    def get_intensity(df):
        return df.max()

    @staticmethod
    def get_pos_of_max(df):
        return df.idxmax()

    @staticmethod
    def get_pos_of_max_gb(df):
        return df.idxmax().map(lambda x: x[2])

    @staticmethod
    def get_bg_mean(df):
        return df.mean()

    @staticmethod
    def file_helper(basepath):
        bp = Path(basepath)
        files = bp.glob('*.h*5')

        def choose_file(file):
            print('Filesize: {:.2f} MB'.format((bp / file).stat().st_size / 1024 ** 2))
            return bp / file

        datafile = interactive(choose_file, file=files)
        return datafile