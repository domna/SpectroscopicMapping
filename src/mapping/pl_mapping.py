import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets, interactive
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.interpolate import interp2d
from scipy.sparse.linalg import spsolve
from pathlib import Path

class Mapper:
    x, y = 0, 0
    spec_bgx, dspec_bgx = 0, 0
    mask = []
    _is_interp = False

    def __init__(self, fname, sx=None, dsx=None, interpolate_to=None):
        self.specx = sx
        self.dspecx = dsx

        self.df = pd.read_csv(fname,
                              sep=r'\s+',
                              names=['x', 'y', 'Wavelength', 'Intensity'],
                              index_col=[0, 1, 2])

        self._midx = self.df.index.droplevel(level=2).drop_duplicates()
        if interpolate_to is not None:
            #self._midx = pd.MultiIndex.from_arrays(
            #    (np.around(np.array([*self._midx.to_flat_index().to_numpy()]) / interpolate_to) * interpolate_to ).T)
            self._midx = interpolate_to
            self._is_interp = True

        self._calculated = False

        self.integrated = pd.DataFrame(index=self._midx,
                                       columns=['Intensity'])
        self.wno = pd.DataFrame(index=self._midx,
                                columns=['Intensity'])

        self.x, self.y = self.df.index.droplevel(level=2).drop_duplicates()[0]
        wnumber = self.df.index.get_level_values(2).drop_duplicates()

        # Set slider values if not provided
        if sx is None:
            self.specx = wnumber[0]
        if dsx is None:
            self.dspecx = 50 * abs(wnumber[1] - wnumber[0])

        self.specx_slider = widgets.FloatSlider(value=wnumber[0],
                                                min=wnumber.min(),
                                                max=wnumber.max(),
                                                step=abs(wnumber[1] - wnumber[0]),
                                                description='ν',
                                                continous_update=False)
        self.dspecx_slider = widgets.FloatSlider(value=50*abs(wnumber[1] - wnumber[0]),
                                                 min=1,
                                                 max=wnumber.max() / 2,
                                                 step=abs(wnumber[1] - wnumber[0]),
                                                 description='Δν',
                                                 continous_update=False)

        self.recalc_button = widgets.Button(description="Recalculate",
                                            disabled=False,
                                            button_style='',
                                            tooltip='Recalculate 2D Graph')

        self.mask_button = widgets.Button(description="Un/mask",
                                          disabled=False,
                                          button_style='',
                                          tooltip='Mask or unmask current data point')

        self.level_slider = widgets.IntSlider(value=10,
                                              min=3,
                                              max=50,
                                              description='Lvls',
                                              disabled=False)

        self.contour_select = widgets.Dropdown(
            options=['Signal', 'Position'],
            value='Signal',
            description='Display',
            disabled=False
        )

        self.levels = self.level_slider.value

        self.g = go.FigureWidget()
        self.s = go.Scatter()
        self.s_peak = go.Scatter()
        self.c = go.Contour()

        self.calculate_2d()

    def get_dataframes(self):
        return self.df, self.integrated, self.wno

    def save_to_files(self, base_filename, append_info=False):
        info_string = '_cwave={:.2f}_dcwave={:.2f}'.format(
            self.specx, self.dspecx)
        if append_info:
            base_filename += info_string

        self.apply_mask(self.integrated).to_csv(base_filename + '_integrated.csv')
        self.apply_mask(self.wno).to_csv(base_filename + '_wavelength.csv')

    def save_current_spectrum_to_file(self, filename):
        df_roi = self.get_roi()
        pos = ' (x={:},y={:})'.format(self.x, self.y)

        pd.DataFrame({'Raw' + pos: df_roi.values[:, 0]})\
            .set_index(df_roi.index)\
            .to_csv(filename)

    def reverse(self, revx, revy):
        self.integrated.index = pd.MultiIndex.from_arrays([Mapper.get_xindex(self.integrated)[::1 - 2 * int(revx)],
                                                           Mapper.get_yindex(self.integrated)[::1 - 2 * int(revy)]])

        self.wno.index = pd.MultiIndex.from_arrays([Mapper.get_xindex(self.wno)[::1 - 2 * int(revx)],
                                                    Mapper.get_yindex(self.wno)[::1 - 2 * int(revy)]])

    def revx(self):
        self.reverse(True, False)

    def revy(self):
        self.reverse(False, True)

    def set_initial_params(self, central_wavelen, delta_central_wavelen):
        self.specx = central_wavelen
        self.dspecx = delta_central_wavelen

    def calculate_2d(self):
        idx = pd.IndexSlice
        roi = self.df.unstack(level=-1).loc[:,idx[:,self.specx - self.dspecx:self.specx + self.dspecx]]

        x = roi.index.get_level_values(0)
        y = roi.index.get_level_values(1)

        if self._is_interp:
            int_interpol = interp2d(x, y, roi.apply(lambda x: x.max(), axis=1).values)
            self.integrated = pd.DataFrame(int_interpol(self._midx.get_level_values(0).unique(),
                                                        self._midx.get_level_values(1).unique()).flatten(),
                                           index=self._midx.sortlevel()[0],
                                           columns=['Intensity'])

            wno_interpol = interp2d(x, y, roi.apply(lambda x: x.idxmax()[1], axis=1).values, fill_value=np.nan)
            self.wno = pd.DataFrame(wno_interpol(self._midx.get_level_values(0).unique(),
                                                 self._midx.get_level_values(1).unique()).flatten(),
                                           index=self._midx.sortlevel()[0],
                                           columns=['Intensity'])
        else:
            self.integrated = roi.apply(lambda x: x.max(), axis=1)
            self.wno = roi.apply(lambda x: x.idxmax()[1], axis=1)

        self.x, self.y = self.df.index.droplevel(level=2).drop_duplicates()[0]

        self._calculated = True

    def get_roi(self):
        if self._is_interp:
            multidx = self.df.index.droplevel(level=2).drop_duplicates()
            idx_pos = np.array(list(map(lambda c: np.sqrt((c[0] - self.x) ** 2 + (c[1] - self.y) ** 2),
                              multidx.to_numpy()))).argmin()

            self.x, self.y = multidx[idx_pos]

        df_roi = self.df.loc[pd.IndexSlice[self.x, self.y, :]]
        return df_roi

    def construct_scatter(self):
        df_roi = self.get_roi()

        xpos = self.wno.loc[self.x, self.y]
        ypos = self.integrated.loc[self.x, self.y]

        self.s = go.Scatter(x=df_roi.index, y=df_roi.values[:, 0])
        self.s_corr = go.Scatter(x=df_roi.index, y=df_roi.values[:, 0])
        self.s_peak = go.Scatter(x=[xpos], y=[ypos])

    def update_spectrum(self):
        df_roi = self.get_roi()
        xpos = Mapper.get_pos_of_max(df_roi.loc[self.specx - self.dspecx:self.specx + self.dspecx].iloc[:, 0])
        ypos = Mapper.get_intensity(df_roi.loc[self.specx - self.dspecx:self.specx + self.dspecx].iloc[:, 0])

        with self.g.batch_update():
            self.g.data[1].x = df_roi.index
            self.g.data[1].y = df_roi.values[:, 0]
            self.g.data[2].x = [xpos]
            self.g.data[2].y = [ypos]
            self.g.data[1].name = 'x={:.2f},y={:.2f}'.format(self.x, self.y)
            self.g.update_shapes(patch=dict(x0=self.specx - self.dspecx,
                                            x1=self.specx - self.dspecx),
                                 selector=dict(name="l1"))
            self.g.update_shapes(patch=dict(x0=self.specx + self.dspecx,
                                            x1=self.specx + self.dspecx),
                                 selector=dict(name="l2"))

    def change_xy(self, trace, points, selector):
        if len(points.xs) > 0 and len(points.ys) > 0:
            self.x = points.xs[0]
            self.y = points.ys[0]

            self.update_spectrum()

    def change_slider(self, change):
        self.specx = self.specx_slider.value
        self.dspecx = self.dspecx_slider.value

        self.update_spectrum()

    def set_contour(self, df):
        y = df.index.get_level_values(1)
        x = df.index.get_level_values(0)

        df_display = self.apply_mask(df)

        with self.g.batch_update():
            self.g.data[0].x = x.drop_duplicates()
            self.g.data[0].y = y.drop_duplicates()
            self.g.data[0].z = Mapper.reshape_df(df_display)
            self.g.data[0].name = 'ν={:.2f},Δν={:.2f}'.format(self.specx, self.dspecx)
            self.g.data[0].contours = dict(start=float(df_display.min()),
                                           end=float(df_display.max()),
                                           size=float(df_display.max() - df_display.min())/self.levels)

    def change_contour(self, change):
        if self.contour_select.value == 'Signal':
            self.set_contour(self.integrated)

        if self.contour_select.value == 'Position':
            self.set_contour(self.wno)

    def change_levels(self, change):
        self.levels = self.level_slider.value

        self.change_contour(None)

    def update_contour(self, change):
        self.calculate_2d()
        self.change_contour(None)

    def apply_mask(self, df):
        df_display = df.copy()
        for xdel, ydel in self.mask:
            df_display.loc[xdel, ydel] = np.nan

        return df_display

    def mask_point(self, change):
        if (self.x, self.y) in self.mask:
            self.mask.remove((self.x, self.y))
        else:
            self.mask.append((self.x, self.y))

        self.change_contour(None)

    @staticmethod
    def reshape_df(df):
        x = df.index.get_level_values(0).nunique()
        y = df.index.get_level_values(1).nunique()

        if df.index.size != (x * y):
            fillarr = np.ones(abs(df.index.size - y * x)) * np.nan
            return np.append(df.values, fillarr).reshape(x, y, order='F').T
        else:
            return df.sort_index().values.reshape(x, y, order='C').T

    def generate_interactive_plot(self):
        if not self._calculated:
            return None

        y = self.integrated.index.get_level_values(1)
        x = self.integrated.index.get_level_values(0)

        fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)
        self.c = go.Contour(z=Mapper.reshape_df(self.integrated),
                            x=x.drop_duplicates(),
                            y=y.drop_duplicates(),
                            colorscale='Viridis',
                            colorbar=dict(x=-.2, len=1),
                            contours=dict(start=float(self.integrated.min()),
                                          end=float(self.integrated.max()),
                                          size=float(self.integrated.max() - self.integrated.min())/self.levels))

        self.construct_scatter()
        self.specx_slider.value = self.specx
        self.dspecx_slider.value = self.dspecx


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
        self.g.data[1].name = 'x={:.2f},y={:.2f}'.format(self.x, self.y)
        self.g.data[2].name = 'Detected Peak'
        self.g.data[2].marker = dict(size=8, color='red', symbol='cross')
        self.g.add_shape(type='line', x0=self.specx - self.dspecx, xref='x2', yref='paper', y0=0,
                         x1=self.specx - self.dspecx, y1=1, line_width=3, name='l1')
        self.g.add_shape(type='line', x0=self.specx + self.dspecx, xref='x2', yref='paper', y0=0,
                         x1=self.specx + self.dspecx, y1=1, line_width=3, name='l2')

        self.specx_slider.observe(self.change_slider, names="value")
        self.dspecx_slider.observe(self.change_slider, names="value")
        self.contour_select.observe(self.change_contour, names="value")
        self.recalc_button.on_click(self.update_contour)
        self.level_slider.observe(self.change_levels, names='value')
        self.g.data[0].on_click(self.change_xy)
        self.mask_button.on_click(self.mask_point)

        return widgets.VBox([widgets.HBox([widgets.VBox([self.specx_slider,
                                                         self.dspecx_slider,
                                                         self.recalc_button]),
                                           widgets.VBox([widgets.HBox([self.contour_select, self.mask_button]),
                                                         self.level_slider])]),
                             self.g])

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
        files = bp.glob('*.txt')

        def choose_file(file):
            print('Filesize: {:.2f} MB'.format((bp / file).stat().st_size / 1024 ** 2))
            return bp / file

        datafile = interactive(choose_file, file=files)
        return datafile

    @staticmethod
    def generate_eqi_grid(points_per_axis, points_overall, stepsize):
        x = np.arange(points_overall)
        y = np.arange(points_overall)
        for i in range(points_per_axis):
            x[points_per_axis * i:points_per_axis * (i + 1)] = (np.arange(points_per_axis) * stepsize)[::(-1)**i]
            y[points_per_axis * i:points_per_axis * (i + 1)] = np.ones(points_per_axis) * i * stepsize

        return pd.MultiIndex.from_arrays([x, y], names=['x', 'y'])
