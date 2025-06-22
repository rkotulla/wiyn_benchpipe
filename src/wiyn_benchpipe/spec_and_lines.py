
import numpy
import matplotlib.pyplot as plt
import scipy.interpolate

class SpecAndLines(object):

    def __init__(self, spec, filterwidth=50, steps=10):
        self.spec = spec
        self.wl = numpy.arange(spec.shape[0])

        self.filterwidth = filterwidth
        self.steps = steps

        self.fit_continuum()
        self.contsub = self.spec - self.continuum


    def fit_continuum(self, filterwidth=None, steps=None):

        if (filterwidth is None):
            filterwidth = self.filterwidth
        if (steps is None):
            steps = self.steps

        wl = self.wl
        spec = self.spec
        n_wl = self.spec.shape[0]

        padded = numpy.pad(spec, (filterwidth, filterwidth), mode='constant', constant_values=numpy.nan)
        wl_padded = numpy.arange(-filterwidth, n_wl + filterwidth)
        #         print("padded shape:", padded.shape)
        #         print("wl padded:", wl_padded.shape, n_wl)

        filtered_wl = []
        filtered_flux = []
        # fig,ax = plt.subplots()
        # ax.plot(wl_padded, padded, lw=1, alpha=0.2)
        _med = numpy.nan
        for c in range(0, n_wl, steps):
            left = c #- filterwidth
            right = c + 2*filterwidth + 1
            cutout = padded[left:right].copy()
            good = numpy.isfinite(cutout)
            _med = numpy.nan
            for iteration in range(3):
                if (numpy.sum(good) <= 0):
                    break
                _stats = numpy.nanpercentile(cutout, [16 ,50 ,84])
                _med = _stats[1]
                _sigma = 0.5 *(_stats[2] -_stats[0])
                good = good & (cutout > _med - 3 *_sigma) & (cutout < _med + 3 *_sigma)

            filtered_wl.append(c)
            filtered_flux.append(_med)

        # ax.plot(filtered_wl, filtered_flux)
        interp = scipy.interpolate.interp1d(x=filtered_wl, y=filtered_flux)
        fullres = interp(wl)
        # ax.plot(wl, fullres, lw=2, alpha=0.5)

        self.continuum_interp = interp
        self.continuum = fullres

        return fullres

    def match_amplitude(self, other, plot=False, plotname=None):

        gain = 0.2
        readnoise = 3

        fm_size = 3
        self_fm = scipy.ndimage.median_filter(self.contsub, size=fm_size)
        other_fm = scipy.ndimage.median_filter(other.contsub, size=fm_size)

        # work out best scaling factor
        valid = numpy.isfinite(other.contsub) & numpy.isfinite(self.contsub) & (self.contsub != 0)

        # require a minimum flux
        cs =  other.contsub
        use = numpy.isfinite(cs)
        for i in range(3):
            _stats = numpy.nanpercentile(cs[use], [16 ,50 ,84])
            _med = _stats[1]
            _sigma = 0.5 *(_stats[2 ] -_stats[0])
            use = use & (cs > _med - 3 *_sigma) & (cs < _med + 3 *_sigma)
        min_flux_other = _med + 2* _sigma

        cs = self.contsub
        use = numpy.isfinite(cs)
        _med, _sigma = 0, 0
        for i in range(3):
            _stats = numpy.nanpercentile(cs[use], [16, 50, 84])
            try:
                _med = _stats[1]
                _sigma = 0.5 * (_stats[2] - _stats[0])
                use = use & (cs > _med - 3 * _sigma) & (cs < _med + 3 * _sigma)
            except IndexError:
                break
        min_flux_self = _med + 2 * _sigma

        valid = valid & (self.contsub > min_flux_self) & (other.contsub > min_flux_other)

        # scale = spec_cs / skylines_only
        # weighted_scale = numpy.sum((spec_cs * skylines_only)[valid]) / numpy.sum(skylines_only[valid])
        # print(numpy.nanmean(scale), weighted_scale)
        # scale = other_fm / self_fm
        scale = self_fm / other_fm
        weighted_scale = numpy.sum((scale * self_fm)[valid]) / numpy.sum(self_fm[valid])

        if (plot):
            _xlim = (750, 2550)

            fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(24, 10))
            # fig.suptitle("spec id: %d" % (specid))

            # spec_cs = spec - spec_cont
            ax = axs[0, 0]
            ax.plot(self.wl, self.contsub, lw=1, label='self')
            ax.plot(self.wl, other.contsub, lw=1, alpha=0.5, label='other')
            #             ax.axhline(y=min_flux_self, alpha=0.5)
            #             ax.scatter(wl[valid], other.contsub[valid], alpha=0.5)
            ax.axhline(y=0)
            ax.set_xlim(_xlim)
            ax.set_ylim((-50, 800))
            ax.legend()

            ax = axs[0, 1]
            ax.scatter(other.contsub, self.contsub, s=1)
            _max = numpy.nanmax([numpy.nanmax(self.contsub), numpy.nanmax(other.contsub)])
            _min = numpy.nanmin([numpy.nanmin(self.contsub), numpy.nanmin(other.contsub)])
            ax.set_xlabel("other")
            ax.set_ylabel("self")
            ax.set_xlim((_min, _max))
            ax.set_ylim((_min, _max))
            _x = numpy.arange(_min, _max, 10)
            for s in [0.9, 1.1, 1.0, 0.8, 1.2, 0.7, 1.3]:
                ax.plot(_x, _x * s, alpha=0.12)
            ax.plot(_x, _x * weighted_scale)

            ax = axs[1, 0]
            ax.plot(self.wl, self.contsub, lw=1, alpha=0.5, label='self')
            ax.plot(self.wl, other.contsub * weighted_scale, lw=1, alpha=0.5, label='other scaled')
            ax.set_xlim(_xlim)
            ax.set_ylim((-50, 800))

            ax = axs[1, 1]
            noise2 = gain * self.spec + 5 * readnoise ** 2
            noise = numpy.sqrt(noise2) / gain
            ax.plot(self.wl, other.contsub * weighted_scale - self.contsub, lw=1)
            ax.plot(self.wl, noise, alpha=0.4, lw=2)
            ax.set_xlim(_xlim)
            ax.set_ylim((-200, 200))
            ax.axhline(y=0, lw=5, alpha=0.2)

            #             if (plotname is not None):
            #                 fig.savefig(plotname)
            #                 plt.close(fig)

            return weighted_scale, min_flux_self, min_flux_other, fig

        return weighted_scale, min_flux_self, min_flux_other


    def dump(self, fn):
        c = numpy.array([self.wl, self.spec, self.contsub, self.continuum]).T
        numpy.savetxt(fn, c)