import numpy
import pandas

def pick_locally_bright_lines(cat, n_blocks, n_per_block, xmin, xmax, col_pos='gauss_center', col_flux='gauss_amp'):
    blocksize = (xmax - xmin) / n_blocks
    block_pos = (numpy.arange(n_blocks) + 0.5) * blocksize

    selections = []
    for b in range(n_blocks):
        b1 = int(numpy.max([xmin, numpy.floor(b * blocksize + xmin)]))
        b2 = int(numpy.min([xmax, numpy.ceil((b + 1) * blocksize + xmin)]))
        # block_max[b] = numpy.nanmax(_comp[b1:b2])

        in_block = (cat[col_pos] > b1) & (cat[col_pos] <= b2)
        if (numpy.sum(in_block) <= 0):
            continue

        block_cat = cat[in_block]

        fluxes = block_cat[col_flux].to_numpy()
        flux_sort = numpy.argsort(fluxes)[::-1][:n_per_block]

        selected = block_cat.iloc[flux_sort]
        print(b, b1, b2, len(selected.index))
        selections.append(selected)

    final_cat = pandas.concat(selections, axis='index')
    return final_cat


def find_best_offset(comp, ref, bins, conv=None, return_hist=False):
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    diffs = comp.reshape((-1, 1)) - ref.reshape((1, -1))
    hist, _bins = numpy.histogram(diffs.ravel(), bins=bins)

    if (conv is not None):
        hist_added = numpy.convolve(hist, conv, mode='same')
    else:
        hist_added = hist

    peak_pos = numpy.argmax(hist_added)
    max_line_matches = hist_added[peak_pos]
    best_offset = bin_centers[peak_pos]
    if (return_hist):
        return best_offset, max_line_matches, (hist, hist_added)

    return best_offset, max_line_matches


def match_catalogs(ref_wl, comp_wl, ref_cat, comp_cat, max_delta_wl, match_counter=None):

    # check if # of values matches
    if (ref_wl.shape[0] != len(ref_cat.index) or
        comp_wl.shape[0] != len(comp_cat.index)):
        print("MATCHING ERROR")
        raise ValueError("Number of points and catalog don't match")

    ref_df = ref_cat.reset_index(drop=True)
        #pandas.DataFrame(ref_cat, index=numpy.arange(len(ref_cat.index))))
    ref_df.columns = ["ref_%s" % c for c in ref_df.columns]
    # ref_df.info()

    merged_df = comp_cat.reset_index(drop=True)
    #pandas.DataFrame(comp_cat, index=numpy.arange(len(comp_cat.index)))
    merged_df['wl_distance'] = numpy.nan
    merged_df['comp_wl'] = comp_wl
    # merged_df.info()

    if (match_counter is not None):
        match_counter[0] += 1
        numpy.savetxt("match_catalog_ref_wl.%d" % (match_counter[0]), ref_wl)
        numpy.savetxt("match_catalog_comp_wl.%d" % (match_counter[0]), comp_wl)
        ref_cat.to_csv("match_catalog_ref_cat.%d" % (match_counter[0]), index=False)
        comp_cat.to_csv("match_catalog_comp_cat.%d" % (match_counter[0]), index=False)

    # print("REF index:\n", ref_cat.index)
    # print("COMP index:\n", comp_cat.index)

    _ref_wl = ref_wl.reshape((1, -1))
    _comp_wl = comp_wl.reshape((-1, 1))
    # print("REF:", _ref_wl.shape, "   COMP:", _comp_wl.shape)

    diff = numpy.fabs(_ref_wl - _comp_wl)
    # print(diff.shape)
    closest = numpy.argmin(diff, axis=1)
    # print(closest)
    # print(closest.shape)

    for i in range(comp_wl.shape[0]):
        distance = diff[i, closest[i]]
        # print(distance, max_delta_wl)
        if (distance < max_delta_wl):
            merged_df.loc[i, ref_df.columns] = ref_df.loc[closest[i], ref_df.columns]
            merged_df.loc[i, 'wl_distance'] = distance

    return merged_df


def gauss(x, center, sigma, amplitude, background):
    return amplitude * numpy.exp(-(x - center) ** 2 / (2 * sigma ** 2)) + background


def normalized_gaussian(x, mu, sig):
    return 1.0 / (numpy.sqrt(2.0 * numpy.pi) * sig) * numpy.exp(-numpy.power((x - mu) / sig, 2.0) / 2)


def wl2pixel(wl, hdr):
    px = (wl - hdr['CRVAL1']) / hdr['CD1_1'] + hdr['CRPIX1'] - 1
    return numpy.max([0, numpy.min([px, hdr['NAXIS1']])]).astype(int)


def fit_gauss(p, x, flux, noise=100):
    model = gauss(x, center=p[0], sigma=p[1], amplitude=p[2], background=p[3])
    diff = model - flux
    return (diff / noise) ** 2
