import math

import numpy as np
from scipy.special import gammaln

from photoholmes.utils.image import read_image, read_jpeg_data

# LN10
M_LN10 = 2.30258509299404568401799145468436421

# PI
M_PI = 3.14159265358979323846264338327950288

# FALSE
FALSE = 0

# TRUE
TRUE = 1


# fatal error, print a message to standard-error output and exit.
def error(msg):
    print("error: " + msg)
    exit()


# memory allocation, initialization to 0, print an error and exit if fail.
def xcalloc(n_items, size):
    if size == 0:
        error("xcalloc: zero size")
    p = np.zeros(n_items * size)
    if p is None:
        error("xcalloc: out of memory")
    return p


# convert rgb image to luminance.
def rgb2luminance(input, output, X, Y, C):
    if C >= 3:
        for x in range(X):
            for y in range(Y):
                output[x + y * X] = round(
                    0.299 * input[x + y * X + 0 * X * Y]
                    + 0.587 * input[x + y * X + 1 * X * Y]
                    + 0.114 * input[x + y * X + 2 * X * Y]
                )
    else:
        np.copyto(output, input)


# computes the logarithm of NFA to base 10.
# NFA = NT.b(n,k,p)
# the return value is log10(NFA)
# n,k,p - binomial parameters.
# logNT - logarithm of Number of Tests
TABSIZE = 100000


def log_nfa(n, k, p, logNT):
    inv = np.zeros(TABSIZE)
    tolerance = 0.1
    log1term, term, bin_term, mult_term, bin_tail, err = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    p_term = p / (1.0 - p)
    if n < 0 or k < 0 or k > n or p < 0.0 or p > 1.0:
        error("wrong n, k or p values in nfa()")
    if n == 0 or k == 0:
        return logNT
    log1term = (
        gammaln(n + 1.0)
        - gammaln(k + 1.0)
        - gammaln(n - k + 1.0)
        + k * math.log(p)
        + (n - k) * math.log(1.0 - p)
    )
    term = math.exp(log1term)
    if term == 0.0:
        if k > n * p:
            return log1term / M_LN10 + logNT
        else:
            return logNT
    bin_tail = term
    for i in range(k + 1, n + 1):
        bin_term = (n - i + 1) * (inv[i] if i < TABSIZE and inv[i] != 0 else (1.0 / i))
        mult_term = bin_term * p_term
        term *= mult_term
        bin_tail += term
        if bin_term < 1.0:
            err = term * ((1.0 - pow(mult_term, (n - i + 1))) / (1.0 - mult_term) - 1.0)
            if err < tolerance * abs(-math.log10(bin_tail) - logNT) * bin_tail:
                break
    return math.log10(bin_tail) + logNT


# computes the vote map.
def compute_grid_votes_per_pixel(image, votes, X, Y):
    cos_t = np.zeros((8, 8))
    zeros = xcalloc(X * Y, sizeof(int))
    for k in range(8):
        for l in range(8):
            cos_t[k][l] = math.cos((2.0 * k + 1.0) * l * M_PI / 16.0)
    for x in range(X - 7):
        for y in range(Y - 7):
            z = 0
            const_along_x = TRUE
            const_along_y = TRUE
            for xx in range(8):
                for yy in range(8):
                    if image[x + xx + (y + yy) * X] != image[x + 0 + (y + yy) * X]:
                        const_along_x = FALSE
                    if image[x + xx + (y + yy) * X] != image[x + xx + (y + 0) * X]:
                        const_along_y = FALSE
            for i in range(8):
                for j in range(8):
                    if i > 0 or j > 0:
                        dct_ij = 0.0
                        for xx in range(8):
                            for yy in range(8):
                                dct_ij += (
                                    image[x + xx + (y + yy) * X]
                                    * cos_t[xx][i]
                                    * cos_t[yy][j]
                                )
                        dct_ij *= (
                            0.25
                            * (1.0 / sqrt(2.0) if i == 0 else 1.0)
                            * (1.0 / sqrt(2.0) if j == 0 else 1.0)
                        )
                        if abs(dct_ij) < 0.5:
                            z += 1
            for xx in range(x, x + 8):
                for yy in range(y, y + 8):
                    if z == zeros[xx + yy * X]:
                        votes[xx + yy * X] = -1
                    if z > zeros[xx + yy * X]:
                        zeros[xx + yy * X] = z
                        votes[xx + yy * X] = (
                            -1
                            if const_along_x or const_along_y
                            else (x % 8) + (y % 8) * 8
                        )
    for xx in range(X):
        for yy in range(7):
            votes[xx + yy * X] = -1
        for yy in range(Y - 7, Y):
            votes[xx + yy * X] = -1
    for yy in range(Y):
        for xx in range(7):
            votes[xx + yy * X] = -1
        for xx in range(X - 7, X):
            votes[xx + yy * X] = -1
    free(zeros)


# detect the main grid of the image from the vote map.
def detect_global_grids(votes, lnfa_grids, X, Y):
    logNT = 2.0 * math.log10(64.0) + 2.0 * math.log10(X) + 2.0 * math.log10(Y)
    grid_votes = np.zeros(64)
    max_votes = 0
    most_voted_grid = -1
    p = 1.0 / 64.0
    for i in range(64):
        grid_votes[i] = 0
    for x in range(X):
        for y in range(Y):
            if votes[x + y * X] >= 0 and votes[x + y * X] < 64:
                grid = votes[x + y * X]
                grid_votes[grid] += 1
                if grid_votes[grid] > max_votes:
                    max_votes = grid_votes[grid]
                    most_voted_grid = grid
    for i in range(64):
        n = X * Y / 64
        k = grid_votes[i] / 64
        lnfa_grids[i] = log_nfa(n, k, p, logNT)
    if (
        most_voted_grid >= 0
        and most_voted_grid < 64
        and lnfa_grids[most_voted_grid] < 0.0
    ):
        return most_voted_grid
    return -1


# detects zones which are inconsistent with a given grid.
def detect_forgeries(
    votes,
    forgery_mask,
    forgery_mask_reg,
    foreign_regions,
    X,
    Y,
    grid_to_exclude,
    grid_max,
):
    logNT = 2.0 * math.log10(64.0) + 2.0 * math.log10(X) + 2.0 * math.log10(Y)
    p = 1.0 / 64.0
    forgery_n = 0
    mask_aux = xcalloc(X * Y, sizeof(int))
    used = xcalloc(X * Y, sizeof(int))
    reg_x = xcalloc(X * Y, sizeof(int))
    reg_y = xcalloc(X * Y, sizeof(int))
    W = 9
    min_size = math.ceil(64.0 * logNT / math.log10(64.0))
    for i in range(X * Y):
        used[i] = FALSE
    for x in range(X):
        for y in range(Y):
            if (
                used[x + y * X] == FALSE
                and votes[x + y * X] != grid_to_exclude
                and votes[x + y * X] >= 0
                and votes[x + y * X] <= grid_max
            ):
                reg_size = 0
                grid = votes[x + y * X]
                x0 = x
                y0 = y
                x1 = x
                y1 = y
                used[x + y * X] = TRUE
                reg_x[reg_size] = x
                reg_y[reg_size] = y
                reg_size += 1
                for i in range(reg_size):
                    for xx in range(reg_x[i] - W, reg_x[i] + W + 1):
                        for yy in range(reg_y[i] - W, reg_y[i] + W + 1):
                            if xx >= 0 and xx < X and yy >= 0 and yy < Y:
                                if (
                                    used[xx + yy * X] == FALSE
                                    and votes[xx + yy * X] == grid
                                ):
                                    used[xx + yy * X] = TRUE
                                    reg_x[reg_size] = xx
                                    reg_y[reg_size] = yy
                                    reg_size += 1
                                    if xx < x0:
                                        x0 = xx
                                    if yy < y0:
                                        y0 = yy
                                    if xx > x1:
                                        x1 = xx
                                    if yy > y1:
                                        y1 = yy
                if reg_size >= min_size:
                    n = (x1 - x0 + 1) * (y1 - y0 + 1) / 64
                    k = reg_size / 64
                    lnfa = log_nfa(n, k, p, logNT)
                    if lnfa < 0.0:
                        foreign_regions[forgery_n].x0 = x0
                        foreign_regions[forgery_n].x1 = x1
                        foreign_regions[forgery_n].y0 = y0
                        foreign_regions[forgery_n].y1 = y1
                        foreign_regions[forgery_n].grid = grid
                        foreign_regions[forgery_n].lnfa = lnfa
                        forgery_n += 1
                        for i in range(reg_size):
                            forgery_mask[reg_x[i] + reg_y[i] * X] = 255
    for x in range(W, X - W):
        for y in range(W, Y - W):
            if forgery_mask[x + y * X] != 0:
                for xx in range(x - W, x + W + 1):
                    for yy in range(y - W, y + W + 1):
                        mask_aux[xx + yy * X] = forgery_mask_reg[xx + yy * X] = 255
    for x in range(W, X - W):
        for y in range(W, Y - W):
            if mask_aux[x + y * X] == 0:
                for xx in range(x - W, x + W + 1):
                    for yy in range(y - W, y + W + 1):
                        forgery_mask_reg[xx + yy * X] = 0
    free(mask_aux)
    free(used)
    free(reg_x)
    free(reg_y)
    return forgery_n


def zero(
    input,
    input_jpeg,
    luminance,
    luminance_jpeg,
    votes,
    votes_jpeg,
    lnfa_grids,
    foreign_regions,
    foreign_regions_n,
    missing_regions,
    missing_regions_n,
    mask_f,
    mask_f_reg,
    mask_m,
    mask_m_reg,
    X,
    Y,
    C,
    C_jpeg,
):
    main_grid = -1
    rgb2luminance(input, luminance, X, Y, C)
    compute_grid_votes_per_pixel(luminance, votes, X, Y)
    main_grid = detect_global_grids(votes, lnfa_grids, X, Y)
    *foreign_regions_n = detect_forgeries(
        votes, mask_f, mask_f_reg, foreign_regions, X, Y, main_grid, 63
    )
    if main_grid > -1 and input_jpeg is not None:
        rgb2luminance(input_jpeg, luminance_jpeg, X, Y, C_jpeg)
        compute_grid_votes_per_pixel(luminance_jpeg, votes_jpeg, X, Y)
        for x in range(X):
            for y in range(Y):
                if votes[x + y * X] == main_grid:
                    votes_jpeg[x + y * X] = -1
        *missing_regions_n = detect_forgeries(
            votes_jpeg, mask_m, mask_m_reg, missing_regions, X, Y, -1, 0
        )
    return main_grid


def main():
    input = iio_read_image_double_split(sys.argv[1])
    X, Y, C = len(input[0]), len(input), 1

    if len(sys.argv) == 3:
        input_jpeg = iio.read_image_double_split(sys.argv[2])
        XX, YY, CC = len(input_jpeg[0]), len(input_jpeg), 1
        if X != XX or Y != YY:
            print("image and image_jpeg99 have different size")
            return

    # allocate memory
    luminance = [[0.0] * X for _ in range(Y)]
    luminance_jpeg = [[0.0] * X for _ in range(Y)]
    votes = [[0] * X for _ in range(Y)]
    votes_jpeg = [[0] * X for _ in range(Y)]
    foreign_regions = [[0] * X for _ in range(Y)]
    missing_regions = [[0] * X for _ in range(Y)]
    mask_f = [[0] * X for _ in range(Y)]
    mask_f_reg = [[0] * X for _ in range(Y)]
    mask_m = [[0] * X for _ in range(Y)]
    mask_m_reg = [[0] * X for _ in range(Y)]

    # run algorithm
    main_grid = zero.zero(
        input,
        input_jpeg,
        luminance,
        luminance_jpeg,
        votes,
        votes_jpeg,
        lnfa_grids,
        foreign_regions,
        foreign_regions_n,
        missing_regions,
        missing_regions_n,
        mask_f,
        mask_f_reg,
        mask_m,
        mask_m_reg,
        X,
        Y,
        C,
        CC,
    )

    # print detection result
    if main_grid == -1:
        # main grid not found
        print("No overall JPEG grid found.")

    if main_grid > -1:
        # print main grid
        print(
            "main grid found: #%d (%d,%d) log(nfa) = %g"
            % (main_grid, main_grid % 8, main_grid / 8, lnfa_grids[main_grid])
        )
        global_grids += 1

    for i in range(64):
        # print list of meaningful grids
        if lnfa_grids[i] < 0.0 and i != main_grid:
            print(
                "meaningful global grid found: #%d (%d,%d) log(nfa) = %g"
                % (i, i % 8, i / 8, lnfa_grids[i])
            )
            global_grids += 1

    if foreign_regions_n != 0:
        for i in range(foreign_regions_n):
            if main_grid != -1:
                print(
                    "\nA meaningful grid different from the main one " "was found here:"
                )
            else:
                print("\nA meaningful grid was found here:")
            print(
                "bounding box: %d %d to %d %d [%dx%d]"
                % (
                    foreign_regions[i].x0,
                    foreign_regions[i].y0,
                    foreign_regions[i].x1,
                    foreign_regions[i].y1,
                    foreign_regions[i].x1 - foreign_regions[i].x0 + 1,
                    foreign_regions[i].y1 - foreign_regions[i].y0 + 1,
                )
            )
            print(
                "grid: #%d (%d,%d)"
                % (
                    foreign_regions[i].grid,
                    foreign_regions[i].grid % 8,
                    foreign_regions[i].grid / 8,
                )
            )
            print("log(nfa) = %g" % foreign_regions[i].lnfa)

    if main_grid > -1 and missing_regions_n > 0:
        for i in range(missing_regions_n):
            print("\nA region with missing JPEG grid was found here:")
            print(
                "bounding box: %d %d to %d %d [%dx%d]"
                % (
                    missing_regions[i].x0,
                    missing_regions[i].y0,
                    missing_regions[i].x1,
                    missing_regions[i].y1,
                    missing_regions[i].x1 - missing_regions[i].x0 + 1,
                    missing_regions[i].y1 - missing_regions[i].y0 + 1,
                )
            )
            print(
                "grid: #%d (%d,%d)"
                % (
                    missing_regions[i].grid,
                    missing_regions[i].grid % 8,
                    missing_regions[i].grid / 8,
                )
            )
            print("log(nfa) = %g" % missing_regions[i].lnfa)

    if foreign_regions_n + missing_regions_n == 0 and main_grid < 1:
        print(
            "\nNo suspicious traces found in the image " "with the performed analysis."
        )

    if main_grid > 0:
        print(
            "\nThe most meaningful JPEG grid origin is not (0,0)."
            "This may indicate that the image has been cropped."
        )

    if global_grids > 1:
        print("\nThere is more than one meaningful grid. " "This is suspicious.")

    if foreign_regions_n + missing_regions_n > 0:
        print(
            "\nSuspicious traces found in the image."
            "This may be caused by image manipulations such as resampling, "
            "copy-paste, splicing.  Please examine the deviant meaningful region "
            "to make your own opinion about a potential forgery."
        )

    # store vote map and forgery detection outputs
    iio.write_image_double("luminance.png", luminance, X, Y)
    iio.write_image_int("votes.png", votes, X, Y)
    iio.write_image_int("votes_jpeg.png", votes_jpeg, X, Y)
    iio.write_image_int("mask_f.png", mask_f_reg, X, Y)
    iio.write_image_int("mask_m.png", mask_m_reg, X, Y)

    # free memory
    del input
    if input_jpeg is not None:
        del input_jpeg
    del luminance
    del luminance_jpeg
    del votes
    del votes_jpeg
    del foreign_regions
    del missing_regions
    del mask_f
    del mask_f_reg
    del mask_m
    del mask_m_reg


if __name__ == "__main__":
    main()
