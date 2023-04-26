import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .plotting import get_nyquist_limits

class LiveAxes:
    def __init__(self, ax=None, fixed_xlim=None, fixed_ylim=None, axis_extend_ratio={'x': 0.5, 'y': 0.25}):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 3))
        else:
            fig = ax.get_figure()

        self.fig, self.ax = fig, ax

        self.fixed_xlim = fixed_xlim
        self.fixed_ylim = fixed_ylim
        self.axis_extend_ratio = axis_extend_ratio

        self.line_artists = {}
        self.data_update_funcs = {}
        self.data_update_kwargs = {}

        self.text_artists = {}
        self.text_update_funcs = {}

    @property
    def all_artist_handles(self):
        return list(self.line_artists.values()) + list(self.text_artists.values())

    def add_line_artist(self, name, data_update_func, data_update_kwargs={}, **kw):
        """
        Add animated line artist to axis
        :param str name:
        :param data_update_func: function that will provide updated data at each frame.
        Call should be data_update_func(frame, **kwargs); must return tuple (x_data, y_data)
        :param dict data_update_kwargs: dict of keyword arguments that should be passed to data_update_func.
        Intended for use when the same data_update_func is used for multiple artists
        :param kw: kwargs passed to plt.plot
        :return:
        """
        # Initialize artist with empty dataset
        xdata = []
        ydata = []
        artist, = self.ax.plot(xdata, ydata, **kw)

        # Store artist and its data source
        self.line_artists[name] = artist
        self.data_update_funcs[name] = data_update_func
        self.data_update_kwargs[name] = data_update_kwargs

    def add_text_artist(self, name, x, y, text_update_func, **kw):
        """
        Add animated text artist to axis
        :param str name: artist name
        :param float x: x coordinate
        param float y: y coordinate
        :param text_update_func: function that will provide updated text at each frame.
        Call should be text_update_func(frame); must return str
        :param kw: kwargs passed to plt.text
        :return:
        """
        artist = self.ax.text(x, y, '', **kw)

        self.text_artists[name] = artist
        self.text_update_funcs[name] = text_update_func

    def anim_init(self):
        # Set fixed axis limits if specified
        for axis in ['x', 'y']:
            if getattr(self, f'fixed_{axis}lim') is not None:
                getattr(self.ax, f'set_{axis}lim')(getattr(self, f'fixed_{axis}lim'))

        if len(self.ax.get_legend_handles_labels()[0]) > 0:
            self.ax.legend()  # loc='upper right')

        return self.all_artist_handles

    def anim_update(self, frame):
        # Update line data
        for name, artist in self.line_artists.items():
            # Refresh data
            x_refreshed, y_refreshed = self.data_update_funcs[name](frame, **self.data_update_kwargs[name])
            artist.set_xdata(x_refreshed)
            artist.set_ydata(y_refreshed)

        # Update text
        for name, artist in self.text_artists.items():
            artist.set_text(self.text_update_funcs[name](frame))

        # Update non-fixed axis limits
        axis_lim_changed = False
        for axis in ['x', 'y']:
            if getattr(self, f'fixed_{axis}lim') is None:
                update_lim = self.update_axis_limits(axis)
                axis_lim_changed = max(axis_lim_changed, update_lim)
        if axis_lim_changed:
            self.fig.canvas.draw()

        return self.all_artist_handles

    def update_axis_limits(self, axis):
        # Get data limits for each artist
        datalims = np.empty((len(self.line_artists), 2))

        for i, artist in enumerate(self.line_artists.values()):
            data = getattr(artist, f'get_{axis}data')()
            datalims[i] = [np.nanmin(data), np.nanmax(data)]

        # Get data limits across all artists
        datalim = (np.nanmin(datalims[:, 0]), np.nanmax(datalims[:, 1]))
        data_range = datalim[1] - datalim[0]

        new_lim = list(getattr(self.ax, f'get_{axis}lim')())
        update_lim = False

        # Check if data limits exceed axis limits
        if datalim[0] < new_lim[0]:
            new_lim[0] = datalim[0] - data_range * self.axis_extend_ratio[axis]
            update_lim = True
        if datalim[1] > new_lim[1]:
            new_lim[1] = datalim[1] + data_range * self.axis_extend_ratio[axis]
            update_lim = True

        if update_lim:
            getattr(self.ax, f'set_{axis}lim')(new_lim)

        return update_lim

    def run(self, frames=100, interval=100, repeat=False):
        # If axis limits are fixed, we can use blit to increase rendering speed
        if self.fixed_xlim and self.fixed_ylim:
            blit = True
        else:
            blit = False

        ani = FuncAnimation(self.fig, self.anim_update, frames=frames, repeat=repeat,
                            init_func=self.anim_init, blit=blit, interval=interval)
        plt.show(block=False)

        return ani

    def plot_static(self):
        self.anim_init()
        self.anim_update(1)
        for axis in ['x', 'y']:
            self.update_axis_limits(axis)


class LiveFigure:
    """
    Container for mutiple LiveAxes instances. LiveAxes instances do not have to be subplots of the same figure.
    """
    def __init__(self, live_axes):

        self.live_axes = live_axes

        self.fig = self.live_axes[0].ax.get_figure()

        # self.axis_artists = {index: [] for index in range(len(self.axes))}

    def anim_init(self):
        # Initialize each LiveAxes
        artists = []
        for ax in self.live_axes:
            artists += list(ax.anim_init())

        # self.fig.tight_layout()

        return artists

    def anim_update(self, frame):
        # Update each LiveAxes
        artists = []
        for ax in self.live_axes:
            artists += list(ax.anim_update(frame))

        return artists

    def run(self, frames=100, interval=100, repeat=False):
        # If all axis limits are fixed, we can use blit to increase rendering speed
        axis_lims_fixed = [(ax.fixed_xlim is None and ax.fixed_ylim is None) for ax in self.live_axes]
        if np.min(axis_lims_fixed):
            blit = True
        else:
            blit = False

        ani = FuncAnimation(self.fig, self.anim_update, frames=frames, repeat=repeat,
                            init_func=self.anim_init, blit=blit, interval=interval)
        plt.show(block=False)

        return ani

    def plot_static(self):
        self.anim_init()
        self.anim_update(1)
        for lax in self.live_axes:
            for axis in ['x', 'y']:
                lax.update_axis_limits(axis)


# Nyquist subclass
class LiveNyquist(LiveAxes):
    """
    Subclass for animated Nyquist plots. Maintains appropriate aspect ratio as data is refreshed
    """
    def anim_update(self, frame):
        # Update line data
        for name, artist in self.line_artists.items():
            # Refresh data
            x_refreshed, y_refreshed = self.data_update_funcs[name](frame)
            artist.set_xdata(x_refreshed)
            artist.set_ydata(y_refreshed)

        # Update text
        for name, artist in self.text_artists.items():
            artist.set_text(self.text_update_funcs[name](frame))

        # Update axis limits using Nyquist rules
        update_lim, new_lim = self.update_axis_limits()
        if update_lim:
            for axis in ['x', 'y']:
                getattr(self.ax, f'set_{axis}lim')(new_lim[axis])
            self.fig.canvas.draw()

        return self.all_artist_handles

    def update_axis_limits(self, *ignored_args):
        # Get data limits for each artist
        datalims = np.empty((len(self.line_artists), 2))

        # Check if EITHER axis needs to be extended
        update_lim = False
        z_data = {'x': [], 'y': []}
        for axis in ['x', 'y']:
            for i, artist in enumerate(self.line_artists.values()):
                data = getattr(artist, f'get_{axis}data')()
                z_data[axis].append(data)
                datalims[i] = [np.nanmin(data), np.nanmax(data)]

            # Get data limits across all artists
            datalim = (np.nanmin(datalims[:, 0]), np.nanmax(datalims[:, 1]))

            # Get current axis limits
            current_lim = list(getattr(self.ax, f'get_{axis}lim')())

            # Check if data limits exceed axis limits
            if datalim[0] < current_lim[0] or datalim[1] > current_lim[1]:
                update_lim = True

        if update_lim:
            # Get new limits obeying Nyquist scaling rules
            z = np.concatenate(z_data['x']) - 1j * np.concatenate(z_data['y'])
            new_lim = get_nyquist_limits(self.ax, z)
        else:
            new_lim = None

        return update_lim, new_lim
