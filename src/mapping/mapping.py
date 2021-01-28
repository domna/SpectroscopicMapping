import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets, interactive
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from pathlib import Path


class Mapper:
    x, y = 0, 0
    # https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    # Useful parameter range: 0.001 <= p <= 0.1 and 10^2 <= lam <= 10^9
    lam, p = 1e4, 1e-3
    spec_bgx, dspec_bgx = 0, 0
    mask = []

    def __init__(self, fname, sx=None, dsx=None, bgx=None, dbgx=None):
        self.specx = sx
        self.dspecx = dsx
        self.spec_bgx = bgx
        self.dspec_bgx = dbgx

        self.df = pd.read_csv(fname,
                              sep='\t',
                              names=['x', 'y', 'Wavenumber', 'Intensity'],
                              index_col=[0, 1, 2])

        self._calculated = False

        self.integrated = pd.DataFrame(index=self.df.index.droplevel(level=2).drop_duplicates(),
                                       columns=['Intensity'])
        self.wno = pd.DataFrame(index=self.df.index.droplevel(level=2).drop_duplicates(),
                                columns=['Intensity'])
        self.d_bg = pd.DataFrame(index=self.df.index.droplevel(level=2).drop_duplicates(),
                                 columns=['Intensity'])

        self.x, self.y = self.df.index.droplevel(level=2).drop_duplicates()[0]
        wnumber = self.df.loc[self.x, self.y].index

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

        self.specx_bg_slider = widgets.FloatSlider(value=self.specx,
                                                   min=self.specx - self.dspecx + 1,
                                                   max=self.specx + self.dspecx - 1,
                                                   step=abs(wnumber[1] - wnumber[0]),
                                                   description='ν_bg',
                                                   continous_update=False)
        self.dspecx_bg_slider = widgets.FloatSlider(value=self.specx,
                                                    min=1,
                                                    max=self.dspecx,
                                                    step=abs(wnumber[1] - wnumber[0]),
                                                    description='Δν_bg',
                                                    continous_update=False)

        self.recalc_button = widgets.Button(description="Recalculate",
                                            disabled=False,
                                            button_style='',
                                            tooltip='Recalculate 2D Graph')

        self.mask_button = widgets.Button(description="Un/mask",
                                   disabled=False,
                                   button_style='',
                                   tooltip='Mask or unmask current data point')

        self.recalc_progress = widgets.IntProgress(value=0, min=0, max=1, step=1, description="",
                                                   bar_style='', orientation='horizontal')

        self.lam_text = widgets.BoundedFloatText(value=self.lam,
                                                 min=1e-20,
                                                 max=1e20,
                                                 description='λ',
                                                 disabled=False)
        self.p_text = widgets.BoundedFloatText(value=self.p,
                                               min=1e-20,
                                               max=1e20,
                                               description='p',
                                               disbaled=False)

        self.level_slider = widgets.IntSlider(value=10,
                                              min=3,
                                              max=50,
                                              description='Lvls',
                                              disabled=False)

        self.contour_select = widgets.Dropdown(
            options=['Signal', 'BG', 'Position'],
            value='Signal',
            description='Display',
            disabled=False
        )

        if self.spec_bgx is None:
            self.spec_bgx = self.specx_bg_slider.value
        if self.dspec_bgx is None:
            self.dspec_bgx = self.dspecx_bg_slider.value

        self.levels = self.level_slider.value

        self.g = go.FigureWidget()
        self.s = go.Scatter()
        self.s_als = go.Scatter()
        self.s_corr = go.Scatter()
        self.s_peak = go.Scatter()
        self.c = go.Contour()

        self.calculate_2d()

    def get_dataframes(self):
        return self.df, self.integrated, self.d_bg, self.wno

    def save_to_files(self, base_filename, append_info=False):
        info_string = '_cwave={:.2f}_dcwave={:.2f}_cbgwave={:.2f}_dcbgwave={:.2f}_lam={:.6f}_p={:.6f}'.format(
            self.specx, self.dspecx, self.spec_bgx, self.dspec_bgx, self.lam, self.p)
        if append_info:
            base_filename += info_string

        self.apply_mask(self.integrated).to_csv(base_filename + '_integrated.csv')
        self.apply_mask(self.d_bg).to_csv(base_filename + '_bg.csv')
        self.apply_mask(self.wno).to_csv(base_filename + '_wavenumber.csv')

    def save_current_spectrum_to_file(self, filename):
        df_roi, baseline = self.get_roi_and_baseline()
        pos = ' (x={:},y={:})'.format(self.x, self.y)

        pd.DataFrame({'Raw' + pos: df_roi.values[:, 0],
                      'Baseline' + pos: baseline,
                      'Corrected' + pos: df_roi.values[:, 0] - baseline})\
            .set_index(df_roi.index)\
            .to_csv(filename)

    def reverse(self, revx, revy):
        self.integrated.index = pd.MultiIndex.from_arrays([Mapper.get_xindex(self.integrated)[::1 - 2 * int(revx)],
                                                           Mapper.get_yindex(self.integrated)[::1 - 2 * int(revy)]])

        self.wno.index = pd.MultiIndex.from_arrays([Mapper.get_xindex(self.wno)[::1 - 2 * int(revx)],
                                                    Mapper.get_yindex(self.wno)[::1 - 2 * int(revy)]])

        self.d_bg.index = pd.MultiIndex.from_arrays([Mapper.get_xindex(self.d_bg)[::1 - 2 * int(revx)],
                                                     Mapper.get_yindex(self.d_bg)[::1 - 2 * int(revy)]])

    def revx(self):
        self.reverse(True, False)

    def revy(self):
        self.reverse(False, True)

    def set_initial_params(self, central_wavelen, delta_central_wavelen,
                           bg_central_wavelen, delta_bg_central_wlen):
        self.specx = central_wavelen
        self.dspecx = delta_central_wavelen
        self.spec_bgx = bg_central_wavelen
        self.dspec_bgx = delta_bg_central_wlen

    def calculate_2d(self):
        wlen_axis = self.df.index.get_level_values(2).drop_duplicates()
        idx_upper = np.argmax(wlen_axis < (self.specx + self.dspecx))
        idx_lower = np.argmin(wlen_axis > (self.specx - self.dspecx))
        idx_min = min(idx_upper, idx_lower)
        idx_max = max(idx_upper, idx_lower)

        bgx_upper = np.argmax(wlen_axis < (self.spec_bgx + self.dspec_bgx))
        bgx_lower = np.argmin(wlen_axis > (self.spec_bgx - self.dspec_bgx))
        bgx_min = min(bgx_upper, bgx_lower)
        bgx_max = max(bgx_upper, bgx_lower)
        no_wlen_points = wlen_axis.shape[0]
        arr = self.df.values[:, 0]
        no_points = len(arr) // no_wlen_points

        self.recalc_progress.max = no_points
        self.recalc_progress.value = 0

        for i in range(no_points):
            d = arr[no_wlen_points * i + idx_min:no_wlen_points * i + idx_max]
            bg = Mapper.baseline_als(d, self.lam, self.p)

            corr = d - bg
            idx = corr.argmax()
            self.integrated.iloc[i] = corr[idx]
            self.wno.iloc[i] = wlen_axis[idx_min + idx]
            self.d_bg.iloc[i] = corr[bgx_min - idx_min:bgx_max - idx_min].mean()

            self.recalc_progress.value += 1

        self.x, self.y = self.df.index.droplevel(level=2).drop_duplicates()[0]

        self._calculated = True

    def get_roi_and_baseline(self):
        df_roi = self.df.loc[self.x, self.y].loc[self.specx + self.dspecx:self.specx - self.dspecx]
        baseline = Mapper.baseline_als(df_roi.values[:, 0], self.lam, self.p)

        return df_roi, baseline

    def construct_scatter(self):
        df_roi, baseline = self.get_roi_and_baseline()

        xpos = self.wno.loc[self.x, self.y].values[0]
        ypos = self.integrated.loc[self.x, self.y].values[0]

        self.s = go.Scatter(x=df_roi.index, y=df_roi.values[:, 0])
        self.s_als = go.Scatter(x=df_roi.index, y=baseline)
        self.s_corr = go.Scatter(x=df_roi.index, y=df_roi.values[:, 0] - baseline)
        self.s_peak = go.Scatter(x=[xpos], y=[ypos])

    def update_spectrum(self):
        df_roi, baseline = self.get_roi_and_baseline()
        xpos = Mapper.get_pos_of_max(df_roi.iloc[:, 0] - baseline)
        ypos = Mapper.get_intensity((df_roi.iloc[:, 0] - baseline))

        with self.g.batch_update():
            self.g.data[1].x = df_roi.index
            self.g.data[1].y = df_roi.values[:, 0]
            self.g.data[2].x = df_roi.index
            self.g.data[2].y = baseline
            self.g.data[3].x = df_roi.index
            self.g.data[3].y = df_roi.values[:, 0] - baseline
            self.g.data[4].x = [xpos]
            self.g.data[4].y = [ypos]
            self.g.data[1].name = 'x={:.2f},y={:.2f}'.format(self.x, self.y)
            self.g.update_shapes(patch=dict(x0=self.spec_bgx + self.dspec_bgx,
                                            x1=self.spec_bgx + self.dspec_bgx),
                                 selector=dict(name="l1"))
            self.g.update_shapes(patch=dict(x0=self.spec_bgx - self.dspec_bgx,
                                            x1=self.spec_bgx - self.dspec_bgx),
                                 selector=dict(name="l2"))

    def change_xy(self, trace, points, selector):
        if len(points.xs) > 0 and len(points.ys) > 0:
            self.x = points.xs[0]
            self.y = points.ys[0]

            self.update_spectrum()

    def change_slider(self, change):
        self.specx = self.specx_slider.value
        self.dspecx = self.dspecx_slider.value

        newmin = self.specx - self.dspecx + 1
        newmax = self.specx + self.dspecx - 1
        if newmin > self.specx_bg_slider.max and newmax > self.specx_bg_slider.min:
            self.specx_bg_slider.max = newmax
            self.specx_bg_slider.min = newmin
        else:
            self.specx_bg_slider.min = newmin
            self.specx_bg_slider.max = newmax

        self.dspecx_bg_slider.max = min(self.specx + self.dspecx - self.specx_bg_slider.value,
                                        self.specx_bg_slider.value - self.specx + self.dspecx)

        if self.specx_bg_slider.value < self.specx_bg_slider.min or \
                self.specx_bg_slider.value > self.specx_bg_slider.max:
            self.specx_bg_slider.value = self.specx

        self.spec_bgx = self.specx_bg_slider.value
        self.dspec_bgx = self.dspecx_bg_slider.value

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

        if self.contour_select.value == 'BG':
            self.set_contour(self.d_bg)

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

    def update_bs_params(self, change):
        self.lam = self.lam_text.value
        self.p = self.p_text.value

        self.update_spectrum()

    @staticmethod
    def reshape_df(df):
        x = df.index.get_level_values(0).nunique()
        y = df.index.get_level_values(1).nunique()

        if df.index.size != (x * y):
            fillarr = np.ones(abs(df.index.size - y * x)) * np.nan
            return np.append(df.values[:, 0], fillarr).reshape(x, y, order='C').T
        else:
            return df.values[:, 0].reshape(x, y, order='C').T

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

        self.specx_bg_slider.min = self.specx - self.dspecx
        self.specx_bg_slider.max = self.specx + self.dspecx

        fig.add_trace(self.c, 1, 1)
        fig.add_trace(self.s, 1, 2)
        fig.add_trace(self.s_als, 1, 2)
        fig.add_trace(self.s_corr, 1, 2)
        fig.add_trace(self.s_peak, 1, 2)
        fig.update_xaxes(title_text="X pos (µm)", row=1, col=1)
        fig.update_yaxes(title_text="Y pos (µm)", row=1, col=1)
        fig.update_xaxes(title_text="Wavenumber (cm-1)", row=1, col=2)
        fig.update_yaxes(title_text="Intensity (arb. units)", row=1, col=2)
        fig.update_layout(width=1000, height=500, autosize=True)

        self.g = go.FigureWidget(fig, layout=go.Layout(barmode='overlay'))
        self.g.data[0].name = 'ν={:.2f},Δν={:.2f}'.format(self.specx, self.dspecx)
        self.g.data[1].name = 'x={:.2f},y={:.2f}'.format(self.x, self.y)
        self.g.data[2].name = 'Baseline'
        self.g.data[3].name = 'Corr'
        self.g.data[4].name = 'Detected Peak'
        self.g.data[4].marker = dict(size=8, color='red', symbol='cross')
        self.g.add_shape(type='line', x0=self.spec_bgx - self.dspec_bgx, xref='x2', yref='paper', y0=0,
                         x1=self.spec_bgx - self.dspec_bgx, y1=1, line_width=3, name='l1')
        self.g.add_shape(type='line', x0=self.spec_bgx + self.dspec_bgx, xref='x2', yref='paper', y0=0,
                         x1=self.spec_bgx + self.dspec_bgx, y1=1, line_width=3, name='l2')

        self.specx_slider.observe(self.change_slider, names="value")
        self.dspecx_slider.observe(self.change_slider, names="value")
        self.contour_select.observe(self.change_contour, names="value")
        self.dspecx_bg_slider.observe(self.change_slider, names="value")
        self.specx_bg_slider.observe(self.change_slider, names="value")
        self.recalc_button.on_click(self.update_contour)
        self.lam_text.observe(self.update_bs_params, names="value")
        self.p_text.observe(self.update_bs_params, names="value")
        self.level_slider.observe(self.change_levels, names='value')
        self.g.data[0].on_click(self.change_xy)
        self.mask_button.on_click(self.mask_point)

        return widgets.VBox([widgets.HBox([widgets.VBox([self.specx_slider,
                                                         self.dspecx_slider,
                                                         self.specx_bg_slider,
                                                         self.dspecx_bg_slider,
                                                         widgets.HBox([self.recalc_button, self.recalc_progress])]),
                                           widgets.VBox([widgets.HBox([self.contour_select, self.mask_button]),
                                                         self.level_slider, self.lam_text, self.p_text])]),
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
    def baseline_als(y, lam, p, niter=10):
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
        return z

    @staticmethod
    def file_helper(basepath):
        bp = Path(basepath)
        files = bp.glob('*.txt')

        def choose_file(file):
            print('Filesize: {:.2f} MB'.format((bp / file).stat().st_size / 1024 ** 2))
            return bp / file

        datafile = interactive(choose_file, file=files)
        return datafile
