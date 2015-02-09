#!/usr/bin/env python

import os
import glob

import numpy as np
from astropy.io import fits

# --dir=obs4041 --label=case1


def get_options(args=None):
    from optparse import OptionParser
    parser = OptionParser()
    parser.set_defaults()
    parser.add_option("--dir",
                      help="Root directory (e.g. obs4041)")
    parser.add_option("--case",
                      help="Sub-dir within root for case (e.g. case1 or baseline)")
    parser.add_option("--revision",
                      default=1,
                      type="int",
                      help="integer revision label for output (e.g. 1 for out1)")

    opt, args = parser.parse_args(args)
    return opt


class NoiseCase(object):
    def __init__(self, name, add_noise_func, **kwargs):
        self.name = name
        self.add_noise_func = add_noise_func
        self.kwargs = kwargs

    def __call__(self, acen):
        self.add_noise_func(acen, **self.kwargs)

    def __str__(self):
        outs = []
        outs.append('Noise case "{}"'.format(self.name))
        outs.append('function: {}'.format(self.add_noise_func.__name__))
        for key in sorted(self.kwargs):
            outs.append(' {}: {}'.format(key, self.kwargs[key]))
        return '\n'.join(outs)


def mult_star_error(acen, slots=None, counts_ratios=5, ms_dys=8, ms_dzs=8,
                    dither_period_y=1000, dither_period_z=707, dither_ampl=8):
    """
    Error signal from a mult-star spoiler.
    """
    for slot, counts_ratio, ms_dy, ms_dz in zip(
            *np.broadcast_arrays(slots, counts_ratios, ms_dys, ms_dzs)):
        ok = (acen['alg'] == 8) & (acen['slot'] == slot)
        times = acen['time'][ok]
        dither_y = dither_ampl * np.sin((times - times[0]) * 2 * np.pi / dither_period_y)
        dither_z = dither_ampl * np.sin((times - times[0]) * 2 * np.pi / dither_period_z)
        ms_y = ms_dy + dither_y
        ms_z = ms_dz + dither_z
        cen_y = ms_y / (1 + counts_ratio) / 3600.  # (1 * ms_y + counts_ratio * 0) / (1 + counts_ratio)
        cen_z = ms_z / (1 + counts_ratio) / 3600. # (1 * ms_z + counts_ratio * 0) / (1 + counts_ratio)

        acen['ang_y'][ok] += cen_y
        acen['ang_y_sm'][ok] += cen_y
        acen['ang_z'][ok] += cen_z
        acen['ang_z_sm'][ok] += cen_z


multstar1 = NoiseCase('multstar1', mult_star_error,
                      slots=[7, 5], ms_dys=[8, -8], ms_dzs=[8, -8])


def sinusoidal_noise(acen, slots=None, y_ampls=None, y_periods=None, z_ampls=None, z_periods=None):
    for slot, y_ampl, y_period, z_ampl, z_period in zip(
            *np.broadcast_arrays(slots, y_ampls, y_periods, z_ampls, z_periods)):
        ok = (acen['alg'] == 8) & (acen['slot'] == slot)
        if not np.any(ok):
            continue
        times = acen['time'][ok]
        noise = y_ampl / 3600. * np.sin((times - times[0]) * 2 * np.pi / y_period)
        acen['ang_y'][ok] += noise
        acen['ang_y_sm'][ok] += noise
        noise = z_ampl / 3600. * np.sin((times - times[0]) * 2 * np.pi / z_period)
        acen['ang_z'][ok] += noise
        acen['ang_z_sm'][ok] += noise

case1 = NoiseCase('case1', sinusoidal_noise,
                  slots=[3, 4, 5, 6, 7], y_ampls=5.0, y_periods=450, z_ampls=0.0, z_periods=450)

case2 = NoiseCase('case2', sinusoidal_noise,
                  slots=[3, 4, 5, 6, 7], y_ampls=0.0001, y_periods=450, z_ampls=0.0, z_periods=450)

case6 = NoiseCase('case6', sinusoidal_noise,
                  slots=[3, 4, 5, 6, 7], y_ampls=1.0, y_periods=1000, z_ampls=0.0, z_periods=1000)

case3 = NoiseCase('case3', sinusoidal_noise,
                  slots=[7], y_ampls=1.0, y_periods=1000, z_ampls=1.0, z_periods=707)

case4 = NoiseCase('case4', sinusoidal_noise,
                  slots=[7, 5], y_ampls=[5, -5], y_periods=1000, z_ampls=[5, -5], z_periods=707)

case5 = NoiseCase('case5', sinusoidal_noise,
                  slots=[7, 5], y_ampls=[5, -5], y_periods=10000, z_ampls=[5, -5], z_periods=10000)

case7 = NoiseCase('case7', sinusoidal_noise,
                  slots=[7, 5, 3, 4, 6], y_ampls=[1.0, -1.0, -1.0, 0.3, 0.4], y_periods=10000,
                  z_ampls=[1.0, -1.0, -1.0, 0.3, 0.4], z_periods=10000)

case8 = NoiseCase('case8', sinusoidal_noise,
                  slots=[7, 5], y_ampls=[1.0, -1.0], y_periods=10000,
                  z_ampls=[1.0, -1.0], z_periods=10000)

opt = get_options()
outdir = os.path.join(opt.dir, opt.case, 'out{}'.format(opt.revision))

acens = glob.glob(os.path.join(outdir, 'pcadf*_acen1.fits'))
if len(acens) == 0:
    raise IOError('No ACEN files found')
if len(acens) > 1:
    raise IOError('Multiple ACEN files found {}'.format(acens))

acen_hdus = fits.open(acens[0], mode='update')
acen = acen_hdus[1].data
case_func = globals()[opt.case]
case_func(acen)
acen_hdus.flush()
acen_hdus.close()

with open(os.path.join(opt.dir, opt.case, 'info'), 'w') as fh:
    fh.write(str(case_func))
