import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import widgets, interactive
import pandas as pd
import numpy as np
from scipy import sparse
from scipy.interpolate import griddata
from scipy.sparse.linalg import spsolve
from scipy.optimize import curve_fit
from pathlib import Path
import h5py

class Mapper:
    x, y = 0, 0
    spec_bgx, dspec_bgx = 0, 0
    mask = []
    _is_interp = False

    def _read_hdf5_file(self, fname, interpolate=False):
        f = h5py.File(fname, 'r')

        measurement_index = list(f.keys())[0]
        x = list(filter(lambda x: x != 'Wavelength', f[measurement_index].keys()))
        y = list(f['{:}/{:}'.format(measurement_index, x[0])].keys())
        xm, ym = np.meshgrid(x, y, indexing='ij')

        data = pd.DataFrame(index=pd.MultiIndex.from_arrays([xm.flatten(), ym.flatten()], names=['x', 'y']), 
                            columns=f['{:}/Wavelength'.format(measurement_index)])
        data.columns.name = "Wavelength"

        sensX = []
        sensY = []
        for x, y in data.index:
            dataset = f['{:}/{:}/{:}/Spectrum'.format(measurement_index, x , y)]
            sensX.append(dataset.attrs['Sensor X'])
            sensY.append(dataset.attrs['Sensor Y'])
            
            data.loc[x, y] = np.array(dataset)
    
        # Convert x, y to float
        data.index = data.index.set_levels([data.index.levels[0].astype(float), data.index.levels[1].astype(float)])
        self._midx = data.index.copy()

        if interpolate:
            data.index = pd.MultiIndex.from_arrays([sensX, sensY], names=['x', 'y'])
            self._is_interp = True

        f.close()

        self.df = data

    def __init__(self, fname, sx=None, dsx=None, interpolate=False):
        self.specx = sx
        self.dspecx = dsx

        self._read_hdf5_file(fname, interpolate)
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        self.x, self.y = self.df.index[0]

        self._calculated = False

        self.integrated = pd.DataFrame(index=self._midx,
                                       columns=['Intensity'])
        self.wno = pd.DataFrame(index=self._midx,
                                columns=['Intensity'])
        self.peak_gauss = pd.DataFrame(index=self._midx,
                                       columns=['Intensity'])

        self.x, self.y = self.df.index.drop_duplicates()[0]
        wnumber = self.df.columns

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

        self.gauss_amp = widgets.FloatText(value=488,
                                           description="Gauss A",
                                           disabled=False)
        
        self.gauss_sigma = widgets.FloatText(value=6.1,
                                            description="Gauss Sigma",
                                            disabled=False)

        self.gauss_mu = widgets.FloatText(value=370,
                                          description="Gauss mu",
                                          disabled=False)

        self.lorentz_amp = widgets.FloatText(value=667,
                                            description="Lor A",
                                            disabled=False)
        
        self.lorentz_sigma = widgets.FloatText(value=7,
                                            description="Lor Sigma",
                                            disabled=False)

        self.lorentz_mu = widgets.FloatText(value=387,
                                          description="Lor mu",
                                          disabled=False)

        self.offset = widgets.FloatText(value=0,
                                        description="Offset",
                                        disabled=False)



        self.contour_select = widgets.Dropdown(
            options=['Signal', 'Position', 'Relative Intensity'],
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

        pd.DataFrame({'Raw' + pos: df_roi.values})\
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

    def fit_peaks(self, spectrum):
        params = [self.gauss_amp.value, 
                  self.gauss_sigma.value, 
                  self.gauss_mu.value, 
                  self.lorentz_amp.value, 
                  self.lorentz_sigma.value, 
                  self.lorentz_mu.value, 
                  self.offset.value]

        popt, _ = curve_fit(Mapper.lorentzgauss, spectrum.index.get_level_values(1), spectrum.values, p0=params)

        self.gauss_amp.value = popt[0]
        self.gauss_sigma.value = popt[1]
        self.gauss_mu.value = popt[2]
        self.lorentz_amp.value = popt[3]
        self.lorentz_sigma.value = popt[4]
        self.lorentz_mu.value = popt[5]
        self.offset.value = popt[6] 

        return popt

    def relative(self, spectrum):
        popt = self.fit_peaks(spectrum)

        return popt[0] / popt[3]


    def relative_max(self, spectrum):
        two_max = spectrum.nlargest(2).sort_index()

        return two_max[0] / two_max[1]


    def calculate_2d(self):
        roi = self.df.loc[:,self.specx - self.dspecx:self.specx + self.dspecx]

        x, y = roi.index.get_level_values(0), roi.index.get_level_values(1)

        if self._is_interp:
            xi, yi = self._midx.get_level_values(0), self._midx.get_level_values(1)
            
            self.integrated = pd.DataFrame(griddata((x, y),
                                                    roi.apply(lambda x: x.max(), axis=1).values,
                                                    (xi, yi),
                                                    method='linear'),
                                            index=self._midx).iloc[:,0].sort_index()

            self.wno = pd.DataFrame(griddata((x, y),
                                            roi.apply(lambda x: x.max(), axis=1).values,
                                            (xi, yi),
                                            method='linear'),
                                    index=self._midx).iloc[:,0].sort_index()
        else:
            self.integrated = roi.apply(lambda x: x.max(), axis=1).sort_index()
            self.wno = roi.apply(lambda x: pd.to_numeric(x).idxmax(), axis=1).sort_index()
            #self.peak_gauss = roi.apply(lambda x: self.relative_max(x), axis=1)
            #self.peak_gauss = roi.apply(lambda x: self.relative(x), axis=1)

        self.x, self.y = self.df.index.drop_duplicates()[0]

        self._calculated = True

    def get_roi(self):
        if self._is_interp:
            multidx = self.df.index.drop_duplicates()
            idx_pos = np.array(list(map(lambda c: np.sqrt((c[0] - self.x) ** 2 + (c[1] - self.y) ** 2),
                              multidx.to_numpy()))).argmin()

            self.x, self.y = multidx[idx_pos]

        df_roi = self.df.loc[self.x, self.y]
        return df_roi

    def construct_scatter(self):
        df_roi = self.get_roi()

        #xpos = self.wno.loc[self.x, self.y]
        #ypos = self.integrated.loc[self.x, self.y]

        self.s = go.Scatter(x=df_roi.index, y=df_roi.values)
        self.s_corr = go.Scatter(x=df_roi.index, y=df_roi.values)
        #self.s_peak = go.Scatter(x=[xpos], y=[ypos])

    def update_spectrum(self):
        df_roi = self.get_roi()
        xpos = Mapper.get_pos_of_max(pd.to_numeric(df_roi.loc[self.specx - self.dspecx:self.specx + self.dspecx]))
        ypos = Mapper.get_intensity(df_roi.loc[self.specx - self.dspecx:self.specx + self.dspecx])

        with self.g.batch_update():
            self.g.data[1].x = df_roi.index
            self.g.data[1].y = df_roi.values
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

        if self.contour_select.value == 'Relative Intensity':
            self.set_contour(self.peak_gauss)

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

    def generate_interactive_plot(self):
        if not self._calculated:
            return None

        x = self.integrated.index.get_level_values(0)
        y = self.integrated.index.get_level_values(1)

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
                                                         self.level_slider]),
                                           widgets.VBox([self.gauss_amp,
                                                        self.gauss_sigma,
                                                        self.gauss_mu,
                                                        self.lorentz_amp,
                                                        self.lorentz_sigma,
                                                        self.lorentz_mu,
                                                        self.offset])
                                            ]),
                             self.g])

    @staticmethod
    def lorentzgauss(x, A1, sigma1, mu1, A2, sigma2, mu2, offset):
        return A1 / sigma1 / np.sqrt(2 * np.pi) * np.exp(-(x - mu1)**2 / 2 / sigma1**2) + A2 / np.pi * sigma2 / ((x - mu2)**2 + sigma2**2) + offset

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

    @staticmethod
    def generate_eqi_grid(points_per_axis, points_overall, stepsize):
        x = np.arange(points_overall)
        y = np.arange(points_overall)
        for i in range(points_per_axis):
            x[points_per_axis * i:points_per_axis * (i + 1)] = (np.arange(points_per_axis) * stepsize)[::(-1)**i]
            y[points_per_axis * i:points_per_axis * (i + 1)] = np.ones(points_per_axis) * i * stepsize

        return pd.MultiIndex.from_arrays([x, y], names=['x', 'y'])
