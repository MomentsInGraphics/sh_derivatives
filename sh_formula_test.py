import numpy as np
from math import sqrt, factorial, pi


def sh_normalization(l, m):
    """Evaluates the normalization factor for spherical harmonics."""
    # Slightly unusual parentheses here to benefit from long integer division
    return sqrt(((2 * l + 1) * factorial(l - m)) / (4 * factorial(l + m))) / sqrt(pi)


def cos_sin(x, y, m):
    """
    Returns a pair (cos, sin) for the polynomial trigonometric functions of the
    given frequency. Implemented inefficiently for testing purposes.
    """
    if m == 0:
        return np.ones_like(x), np.zeros_like(x)
    else:
        cp, sp = cos_sin(x, y, m - 1)
        c = x * cp - y * sp
        s = y * cp + x * sp
        return c, s


def cos_sin_derivative(x, y, m, dx, dy):
    """
    Computes derivatives of polynomial trigonometric functions taken w.r.t. x
    dx times and w.r.t. y dy times.
    """
    if dx == dy == 0:
        return cos_sin(x, y, m)
    elif m < dx + dy:
        return np.zeros_like(x), np.zeros_like(x)
    elif dx > 0:
        c, s = cos_sin_derivative(x,  y, m - 1, dx - 1, dy)
        return m * c, m * s
    else:
        c, s = cos_sin_derivative(x, y, m - 1, dx, dy - 1)
        return -m * s, m * c


def scaled_ass_legendre(z, l, m):
    """Evaluates an associated Legendre polynomial (inefficiently)."""
    if l == m == 0:
        return np.ones_like(z)
    elif l < m:
        return np.zeros_like(z)
    elif l == m:
        return (1 - 2 * l) * scaled_ass_legendre(z, l - 1, l - 1)
    elif l == m + 1:
        return (2 * m + 1) * z * scaled_ass_legendre(z, m, m)
    else:
        return (2 * l - 1) / (l - m) * z * scaled_ass_legendre(z, l - 1, m) - (l + m - 1) / (l - m) * scaled_ass_legendre(z, l - 2, m)


def sh_derivative(x, y, z, l, m, dx, dy, dz):
    """
    Evaluates an arbitrary derivative of a spherical harmonic at a point on the
    unit sphere (inefficiently).
    """
    K = sh_normalization(l, abs(m))
    Q = (-1)**dz * scaled_ass_legendre(z, l, abs(m) + dz)
    c, s = cos_sin_derivative(x, y, abs(m), dx, dy)
    if m < 0:
        return K * Q * sqrt(2.0) * s
    elif m == 0 and dx + dy > 0:
        return 0.0
    elif m == 0 and dx + dy == 0:
        return K * Q
    else:
        return K * Q * sqrt(2.0) * c


def sh_4_grad_hess(point):
    out_shs = [[0.0] * 1 for _ in range(25)]
    out_grads = [[0.0] * 3 for _ in range(25)]
    out_hesss = [[0.0] * 6 for _ in range(25)]
    x = point[0]
    y = point[1]
    z = point[2]
    z2 = z * z
    c0 = 1.0
    s0 = 0.0
    d = 0.28209479177387814
    out_shs[0] = d
    d = 0.48860251190291992 * z
    out_shs[2] = d
    a = z2 - 0.33333333333333331
    d = 0.94617469575756008 * a
    out_shs[6] = d
    b = z * (a - 0.26666666666666666)
    d = 1.865881662950577 * b
    out_shs[12] = d
    a = z * b - 0.25714285714285712 * a
    d = 3.70249414203215066 * a
    out_shs[20] = d
    c1 = x
    s1 = y
    d = -0.48860251190291992
    out_shs[1] = s1 * d
    out_shs[3] = c1 * d
    out_grads[1][0] = s0 * d
    out_grads[3][0] = c0 * d
    out_grads[1][1] = c0 * d
    out_grads[3][1] = -s0 * d
    d = 0.48860251190291992
    out_grads[2][2] = d
    d = -1.09254843059207918 * z
    out_shs[5] = s1 * d
    out_shs[7] = c1 * d
    out_grads[5][0] = s0 * d
    out_grads[7][0] = c0 * d
    out_grads[5][1] = c0 * d
    out_grads[7][1] = -s0 * d
    d = 1.89234939151512016 * z
    out_grads[6][2] = d
    a = z2 - 0.2
    d = -2.28522899732232876 * a
    out_shs[11] = s1 * d
    out_shs[13] = c1 * d
    out_grads[11][0] = s0 * d
    out_grads[13][0] = c0 * d
    out_grads[11][1] = c0 * d
    out_grads[13][1] = -s0 * d
    d = 5.59764498885173101 * a
    out_grads[12][2] = d
    b = z * (a - 0.22857142857142856)
    d = -4.68332580490102401 * b
    out_shs[19] = s1 * d
    out_shs[21] = c1 * d
    out_grads[19][0] = s0 * d
    out_grads[21][0] = c0 * d
    out_grads[19][1] = c0 * d
    out_grads[21][1] = -s0 * d
    d = 14.80997656812860264 * b
    out_grads[20][2] = d
    c2 = x * c1 - y * s1
    s2 = y * c1 + x * s1
    d = 0.54627421529603959
    out_shs[4] = s2 * d
    out_shs[8] = c2 * d
    d = 1.09254843059207918
    out_grads[4][0] = s1 * d
    out_grads[8][0] = c1 * d
    out_grads[4][1] = c1 * d
    out_grads[8][1] = -s1 * d
    out_hesss[4][0] = s0 * d
    out_hesss[8][0] = c0 * d
    out_hesss[4][1] = c0 * d
    out_hesss[8][1] = -s0 * d
    out_hesss[4][3] = -s0 * d
    out_hesss[8][3] = -c0 * d
    d = -1.09254843059207918
    out_grads[5][2] = s1 * d
    out_grads[7][2] = c1 * d
    out_hesss[5][2] = s0 * d
    out_hesss[7][2] = c0 * d
    out_hesss[5][4] = c0 * d
    out_hesss[7][4] = -s0 * d
    d = 1.89234939151512016
    out_hesss[6][5] = d
    d = 1.4453057213202769 * z
    out_shs[10] = s2 * d
    out_shs[14] = c2 * d
    d = 2.89061144264055381 * z
    out_grads[10][0] = s1 * d
    out_grads[14][0] = c1 * d
    out_grads[10][1] = c1 * d
    out_grads[14][1] = -s1 * d
    out_hesss[10][0] = s0 * d
    out_hesss[14][0] = c0 * d
    out_hesss[10][1] = c0 * d
    out_hesss[14][1] = -s0 * d
    out_hesss[10][3] = -s0 * d
    out_hesss[14][3] = -c0 * d
    d = -4.57045799464465752 * z
    out_grads[11][2] = s1 * d
    out_grads[13][2] = c1 * d
    out_hesss[11][2] = s0 * d
    out_hesss[13][2] = c0 * d
    out_hesss[11][4] = c0 * d
    out_hesss[13][4] = -s0 * d
    d = 11.19528997770346201 * z
    out_hesss[12][5] = d
    a = z2 - 0.14285714285714285
    d = 3.31161143515146028 * a
    out_shs[18] = s2 * d
    out_shs[22] = c2 * d
    d = 6.62322287030292056 * a
    out_grads[18][0] = s1 * d
    out_grads[22][0] = c1 * d
    out_grads[18][1] = c1 * d
    out_grads[22][1] = -s1 * d
    out_hesss[18][0] = s0 * d
    out_hesss[22][0] = c0 * d
    out_hesss[18][1] = c0 * d
    out_hesss[22][1] = -s0 * d
    out_hesss[18][3] = -s0 * d
    out_hesss[22][3] = -c0 * d
    d = -14.04997741470307204 * a
    out_grads[19][2] = s1 * d
    out_grads[21][2] = c1 * d
    out_hesss[19][2] = s0 * d
    out_hesss[21][2] = c0 * d
    out_hesss[19][4] = c0 * d
    out_hesss[21][4] = -s0 * d
    d = 44.42992970438580613 * a
    out_hesss[20][5] = d
    c0 = x * c2 - y * s2
    s0 = y * c2 + x * s2
    d = -0.59004358992664352
    out_shs[9] = s0 * d
    out_shs[15] = c0 * d
    d = -1.77013076977993045
    out_grads[9][0] = s2 * d
    out_grads[15][0] = c2 * d
    out_grads[9][1] = c2 * d
    out_grads[15][1] = -s2 * d
    d = 1.4453057213202769
    out_grads[10][2] = s2 * d
    out_grads[14][2] = c2 * d
    d = -3.54026153955986089
    out_hesss[9][0] = s1 * d
    out_hesss[15][0] = c1 * d
    out_hesss[9][1] = c1 * d
    out_hesss[15][1] = -s1 * d
    out_hesss[9][3] = -s1 * d
    out_hesss[15][3] = -c1 * d
    d = 2.89061144264055381
    out_hesss[10][2] = s1 * d
    out_hesss[14][2] = c1 * d
    out_hesss[10][4] = c1 * d
    out_hesss[14][4] = -s1 * d
    d = -4.57045799464465752
    out_hesss[11][5] = s1 * d
    out_hesss[13][5] = c1 * d
    d = -1.77013076977993067 * z
    out_shs[17] = s0 * d
    out_shs[23] = c0 * d
    d = -5.31039230933979223 * z
    out_grads[17][0] = s2 * d
    out_grads[23][0] = c2 * d
    out_grads[17][1] = c2 * d
    out_grads[23][1] = -s2 * d
    d = 6.62322287030292056 * z
    out_grads[18][2] = s2 * d
    out_grads[22][2] = c2 * d
    d = -10.62078461867958445 * z
    out_hesss[17][0] = s1 * d
    out_hesss[23][0] = c1 * d
    out_hesss[17][1] = c1 * d
    out_hesss[23][1] = -s1 * d
    out_hesss[17][3] = -s1 * d
    out_hesss[23][3] = -c1 * d
    d = 13.24644574060584112 * z
    out_hesss[18][2] = s1 * d
    out_hesss[22][2] = c1 * d
    out_hesss[18][4] = c1 * d
    out_hesss[22][4] = -s1 * d
    d = -28.09995482940614409 * z
    out_hesss[19][5] = s1 * d
    out_hesss[21][5] = c1 * d
    c1 = x * c0 - y * s0
    s1 = y * c0 + x * s0
    d = 0.62583573544917614
    out_shs[16] = s1 * d
    out_shs[24] = c1 * d
    d = 2.50334294179670458
    out_grads[16][0] = s0 * d
    out_grads[24][0] = c0 * d
    out_grads[16][1] = c0 * d
    out_grads[24][1] = -s0 * d
    d = -1.77013076977993067
    out_grads[17][2] = s0 * d
    out_grads[23][2] = c0 * d
    d = 7.51002882539011374
    out_hesss[16][0] = s2 * d
    out_hesss[24][0] = c2 * d
    out_hesss[16][1] = c2 * d
    out_hesss[24][1] = -s2 * d
    out_hesss[16][3] = -s2 * d
    out_hesss[24][3] = -c2 * d
    d = -5.31039230933979223
    out_hesss[17][2] = s2 * d
    out_hesss[23][2] = c2 * d
    out_hesss[17][4] = c2 * d
    out_hesss[23][4] = -s2 * d
    d = 6.62322287030292056
    out_hesss[18][5] = s2 * d
    out_hesss[22][5] = c2 * d
    return out_shs, out_grads, out_hesss


def sh_4_sloan(x, y, z):
    """
    Automatically generated code of Sloan to evaluate bands 0 to 4 of SH,
    ported to Python for comparison.
    """
    shs = [0.0 for _ in range(25)]
    z2 = z * z

    shs[0] = 0.2820947917738781
    shs[2] = 0.4886025119029199 * z
    shs[6] = 0.9461746957575601 * z2 + -0.3153915652525201
    shs[12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    shs[20] = 1.984313483298443 * z * shs[12] - 1.006230589874905 * shs[6]
    c0 = x
    s0 = y

    a = -0.48860251190292
    shs[3] = a * c0
    shs[1] = a * s0
    b = -1.092548430592079 * z
    shs[7] = b * c0
    shs[5] = b * s0
    c = -2.285228997322329 * z2 + 0.4570457994644658
    shs[13] = c * c0
    shs[11] = c * s0
    a = z * (-4.683325804901025 * z2 + 2.007139630671868)
    shs[21] = a * c0
    shs[19] = a * s0
    c1 = x * c0 - y * s0
    s1 = x * s0 + y * c0

    a = 0.5462742152960395
    shs[8] = a * c1
    shs[4] = a * s1
    b = 1.445305721320277 * z
    shs[14] = b * c1
    shs[10] = b * s1
    c = 3.31161143515146 * z2 + -0.47308734787878
    shs[22] = c * c1
    shs[18] = c * s1
    c0 = x * c1 - y * s1
    s0 = x * s1 + y * c1

    a = -0.5900435899266435
    shs[15] = a * c0
    shs[9] = a * s0
    b = -1.770130769779931 * z
    shs[23] = b * c0
    shs[17] = b * s0
    c1 = x * c0 - y * s0
    s1 = x * s0 + y * c0

    c = 0.6258357354491763
    shs[24] = c * c1
    shs[16] = c * s1
    return shs


def test_ass_legendre():
    """
    Tests associated Legendre functions by comparing to the formulas from
    Wikipedia:
    https://en.wikipedia.org/wiki/Associated_Legendre_polynomials#The_first_few_associated_Legendre_functions
    """
    z = np.linspace(-1.0, 1.0, 10001)
    tests = [
        (0, 0, 1.0),
        (1, 0, z),
        (1, 1, -1.0),
        (2, 0, 0.5 * (3.0 * z**2 - 1.0)),
        (2, 1, -3.0 * z),
        (2, 2, 3.0),
        (3, 0, 0.5 * (5.0 * z**3 - 3.0 * z)),
        (3, 1, 1.5 * (1.0 - 5.0 * z**2)),
        (3, 2, 15.0 * z),
        (3, 3, -15.0),
        (4, 0, 0.125 * (35.0 * z**4 - 30.0 * z**2 + 3.0)),
        (4, 1, -2.5 * (7.0 * z**3 - 3.0 * z)),
        (4, 2, 7.5 * (7.0 * z**2 - 1.0)),
        (4, 3, -105.0 * z),
        (4, 4, 105.0),
    ]
    for l, m, val in tests:
        print("legendre_%d_%d" % (l, m), np.linalg.norm(scaled_ass_legendre(z, l, m) - val))


def test_sh():
    """
    Compares the inefficient evaluation of SH implemented in this file against
    the automatically generated code, including derivatives.
    """
    point_count = 1000
    points = np.random.normal(size=(3, point_count))
    points /= np.linalg.norm(points, axis=0)
    x, y, z = [points[i] for i in range(3)]
    # Evaluate SH and its derivatives with the automatically generated code
    shs, grads, hesss = sh_4_grad_hess(points)
    sloan_shs = sh_4_sloan(x, y, z)
    # Evaluate using the inefficient recurrent implementations
    i = 0
    for l in range(4):
        for m in range(-l, l + 1):
            # Compare the SH itself
            print("sh_%d_%d:" % (l, m), np.linalg.norm(shs[i] - sh_derivative(x, y, z, l, m, 0, 0, 0)))
            print("sh_sloan_%d_%d:" % (l, m), np.linalg.norm(shs[i] - sloan_shs[i]))
            # Compare the gradient
            for j, (dx, dy, dz) in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
                print("grad_%d_%d_%d:" % (l, m, j), np.linalg.norm(grads[i][j] - sh_derivative(x, y, z, l, m, dx, dy, dz)))
            # Compare the Hessian
            for j, (dx, dy, dz) in enumerate([(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]):
                print("hess_%d_%d_%d:" % (l, m, j), np.linalg.norm(hesss[i][j] - sh_derivative(x, y, z, l, m, dx, dy, dz)))
            i += 1


if __name__ == "__main__":
    test_ass_legendre()
    test_sh()
