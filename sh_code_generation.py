import re
from collections import defaultdict
from math import pi, sqrt, prod, factorial
from itertools import combinations_with_replacement


def derivative_index(vars):
    """
    Provides the index at which the specified derivative of a 3D function
    is stored.
    :param vars: The variables with respect to which the derivative is taken. 0
        for x, 1 for y, 2 for z. Pass multiple variables for higher-order
        derivatives. By Schwarz's theorem, the order is irrelevant.
    :return: The index of the derivative in a flattened representation.
    """
    entries = list(combinations_with_replacement([0, 1, 2], len(vars)))
    return entries.index(tuple(sorted(vars)))


def derivative_size(derivative_order):
    """
    Returns the number of different indices that derivative_index() can produce
    when it is fed the given number of variables.
    """
    # binomial(derivative_order + 2, 2)
    return ((derivative_order + 1) * (derivative_order + 2)) // 2


def derivative_name(derivative_order):
    """The identifier name to be used for derivatives of the given order."""
    return {0: "out_shs", 1: "out_grads", 2: "out_hesss"}.get(derivative_order, "out_ders_%d" % derivative_order)


def trig_derivative(cs, vars):
    """
    Determines a partial derivative of sine or cosine polynomials.
    :param cs: One of "c", "s", "-c" or "-s" for + or - cosine or sine.
    :param vars: List of variable indices with respect to which the derivative
        is taken.  0 for x, 1 for y.
    :return: A string like cs for the derivative. Note that the frequency and
        the constant factor also change but that is not reflected here.
    """
    for var in vars:
        if var == 1:
            cs = {"c": "-s", "s": "c", "-c": "s", "-s": "-c"}[cs]
    return cs


def leading_z_coeff(band, order):
    """
    Just before the sine/cosine term is multiplied, an SH basis function is a
    polynomial in z. This function returns the coefficient of its leading term.
    """
    numerator = (2 * band + 1) * prod([(2 * j - 1)**2 for j in range(1, band + 1)])
    denominator = 4 * factorial(band + order) * factorial(band - order)
    return (-1.0)**order * sqrt((((1 if order == 0 else 2) * numerator) / denominator) / pi)


class SHCodeGeneration:
    """
    Generates code to evaluate the spherical harmonics (SH) basis and its
    derivatives efficiently.
    """

    def __init__(self, function_name, band_max, derivative_order, all_bands, homogenized, use_double, convention, language):
        """
        Initializes this SH code generator to use the given parameters.
        :param function_name: The name of the generated function.
        :param band_max: The generated code will evaluate every band up
            to (and including) the given band (or every other band if
            all_bands is False). At least 2.
        :param derivative_order: The number of derivatives that should be
            produced. 0 for SH only, 1 for SH and gradient, 2 for SH, gradient
            and Hessian matrix and so forth.
        :param all_bands: Pass True to generate code for all SH bands. Pass
            False to only compute every other band, i.e. only even or only odd
            bands, depending on the parity of band_max.
        :param homogenized: Whether all polynomials should be homogenized to
            degree band_max without changing their values on the unit sphere.
            Not compatible with all_bands. Not supported if derivatives have
            been requested.
        :param use_double: False to use single-precision floats, True for
            double. Python always uses double.
        :param convention: A string indicating the used conventions for SH
            basis functions. Can be "sloan" or "descouteaux" to use conventions
            of one of the following two papers:
            Sloan 2013, "Efficient Spherical Harmonic Evaluation" in JCGT 2:2
            http://jcgt.org/published/0002/02/06/
            Descoteaux et al. 2007, "Regularized, fast, and robust analytical
            Q-ball imaging" in Magnetic Resonance in Medicine 58:3
            https://doi.org/10.1002/mrm.21277
            If you are working in computer graphics, you probably want "sloan",
            I just needed "descouteaux" for compatibility with a specific prior
            work.
        :param language: The language of the generated code. Can be "c", "cpp",
            "glsl", "hlsl" or "python". "c", "python" and "glsl" have been
            tested most thoroughly.
        """
        self.function_name = function_name
        self.band_max = band_max
        self.derivative_order = derivative_order
        self.convention = convention
        self.all_bands = all_bands
        self.homogenized = homogenized
        if self.homogenized and self.all_bands:
            raise ValueError("Homogenization of SH polynomials is not possible when both odd and even bands are requested.")
        if self.band_max <= 1:
            raise NotImplementedError("SH computation is only implemented for cases where band 2 or higher is included. Band 0 is constant, band 1 linear and you can find the corresponding formulas here in appendix A2: https://www.ppsloan.org/publications/StupidSH36.pdf")
        if self.homogenized and self.derivative_order > 0:
            raise NotImplementedError("Derivative computation for homogenized SH is not supported.")
        self.language = language
        if self.language == "python":
            use_double = True
        self.use_double = use_double
        self.float_type = "double" if use_double else "float"
        self.code = ""
        # List of output values that have been set, as triples
        # (derivative_order, sh_index, derivative_index). The others will be
        # set to zero.
        self.set_output = list()
        # How many pairs of sine and cosine of different frequency are kept
        # around
        self.trig_count = 2 if self.derivative_order < 2 else (self.derivative_order + 1)

    def generate(self):
        """Generates the requested code and returns a string."""
        self.code = ""
        self.start_function()
        self.initializations()
        for i in range(0, self.band_max + 1):
            self.generate_order(i)
        self.homogenize()
        self.output_zeros()
        self.end_function()
        return self.code

    def start_function(self):
        """Generates the function declaration and the start of the body."""
        if self.language == "python":
            self.code += "def %s(point):" % self.function_name
            return
        # All the C-like languages can be handled similarly
        sh_count = self.sh_count()
        out_prefix = dict(glsl="out ", hlsl="out ").get(self.language, "")
        in_prefix = dict(c="const ", cpp="const ").get(self.language, "")
        specifiers = dict(c="static inline ", cpp="inline ").get(self.language, "")
        vec_type = dict(glsl="dvec3" if self.use_double else "vec3", hlsl="%s3" % self.float_type).get(self.language, None)
        # Start the declaration
        self.code += "%svoid %s(%s%s out_shs[%d], " % (specifiers, self.function_name, out_prefix, self.float_type, sh_count)
        # Declare derivative parameters
        for i in range(1, self.derivative_order + 1):
            if i == 1 and vec_type is not None:
                self.code += "%s%s %s[%d], " % (out_prefix, vec_type, derivative_name(i), sh_count)
            else:
                self.code += "%s%s %s[%d][%d], " % (out_prefix, self.float_type, derivative_name(i), sh_count, derivative_size(i))
        # Declare the point parameter and finish
        if vec_type is not None:
            self.code += "%s%s point) {" % (in_prefix, vec_type)
        else:
            self.code += "%s%s point[3]) {" % (in_prefix, self.float_type)

    def new_line(self):
        """Starts a new line with indentation."""
        self.code += "\n    "

    def sh_count(self):
        """Returns the number of evaluated basis functions."""
        band = self.band_max + (1 if self.all_bands else 2)
        return band**2 if self.all_bands else (((band - 1) * band) // 2)

    def sh_index(self, band, order):
        """Returns the flattened SH index for the given band and order."""
        start = band**2 if self.all_bands else (((band - 1) * band) // 2)
        return start + band + order

    def band_needed(self, band):
        """
        Returns True iff the given band should be output by the generated code.
        """
        return band <= self.band_max and (self.all_bands or (self.band_max - band) % 2 == 0)

    def literal(self, float_value):
        """Returns a literal for the given float value."""
        if abs(float_value) >= 0.1 or float_value == 0.0:
            result = ("%.17f" if self.use_double else "%.9f") % float_value
        else:
            result = ("%.16e" if self.use_double else "%.8e") % float_value
        # Eliminate trailing zeros
        result = re.sub(r"([^.+])0+($|e)", r"\1\2", result)
        result = result.replace("0.20000000000000001", "0.2")
        # Use a proper type
        if self.language in ["c", "cpp", "hlsl"] and not self.use_double:
            result += "f"
        return result

    def assign(self, lhs, rhs):
        """
        Assigns the given right-hand side (code for an expression) to the given
        already declared left-hand side.
        """
        self.new_line()
        self.code += "%s = %s%s" % (lhs, rhs, "" if self.language == "python" else ";")

    def trig_factor(self, band, order):
        """
        Accounts for different conventions in determining which sine or cosine
        factor should be used for the specified SH coefficient. Returns "-s",
        "s", "-c" or "c".
        """
        if self.convention == "descouteaux":
            return ("-c" if order % 2 == 1 else "c") if order <= 0 else "s"
        elif self.convention == "sloan":
            return "s" if order <= 0 else "c"

    def assign_shs(self, band, order, poly_z):
        """
        Assigns a value to the SH coefficients (band, order) and
        (band, -order). The appropriate cosine and sine factors are
        incorporated automatically. It also sets derivatives based on poly_z.
        :param band: The band index.
        :param order: The absolute order (an index <= band).
        :param poly_z: The identifier providing the associated Legendre
            polynomial dependent on z that should be multiplied by sine or
            cosine terms and a constant factor to get the SH coefficients. The
            leading coefficient is supposed to be normalized to 1. Pass 1 if it
            is constant.
        """
        if not self.band_needed(band):
            return
        # Iterate through derivatives (including derivative zero, i.e. the SH
        # function itself) and group them by their constant factors
        factor_dict = defaultdict(list)
        for i in range(0, self.derivative_order + 1):
            for vars in combinations_with_replacement([0, 1, 2], i):
                xy_count = i - vars.count(2)
                m = order - (i - xy_count)
                if m < xy_count or (m == 0 and xy_count > 0):
                    continue
                factor = leading_z_coeff(band, m) * ((factorial(band - m) * factorial(m)) / (factorial(band - order) * factorial(m - xy_count)))
                factor_dict[factor].append(vars)
        # Now generate code to compute the derivatives
        for factor, vars_list in factor_dict.items():
            self.assign("d", self.literal(factor) if poly_z == 1 else "%s * %s" % (self.literal(factor), poly_z))
            for vars in vars_list:
                # Separate the derivatives that change cosine/sine terms for
                # use in trig_derivative()
                xy_vars = [var for var in vars if var != 2]
                xy_count = len(xy_vars)
                # Taking the n-th order derivative with respect to z is
                # accomplished by simply increasing the order of the associated
                # Legendre polynomial by n. Since we only have one associated
                # Legendre polynomial at hand, we instead decrease the output
                # order.
                m = order - (len(vars) - xy_count)
                # If we take derivatives of higher order than the degree of the
                # cosine/sine polynomial, we can simply rely on zero-
                # initialization
                if m < xy_count:
                    continue
                if len(vars) > 0:
                    lhs_0 = "%s[%d][%d]" % (derivative_name(len(vars)), self.sh_index(band, -m), derivative_index(vars))
                    lhs_1 = "%s[%d][%d]" % (derivative_name(len(vars)), self.sh_index(band, +m), derivative_index(vars))
                else:
                    lhs_0 = "%s[%d]" % (derivative_name(len(vars)), self.sh_index(band, -m))
                    lhs_1 = "%s[%d]" % (derivative_name(len(vars)), self.sh_index(band, +m))
                # Assemble the derivative from the derivative of sine/cosine
                # factors and the associated Legendre polynomial
                if m == 0:
                    self.assign(lhs_0, "d")
                else:
                    trig_index = (m - xy_count) % self.trig_count
                    self.assign(lhs_0, "%s%d * d" % (trig_derivative(self.trig_factor(band, -m), xy_vars), trig_index))
                    self.assign(lhs_1, "%s%d * d" % (trig_derivative(self.trig_factor(band, +m), xy_vars), trig_index))
                self.set_output.append((len(vars), self.sh_index(band, -m), derivative_index(vars)))
                self.set_output.append((len(vars), self.sh_index(band, +m), derivative_index(vars)))

    def initializations(self):
        """Declares and initializes variables that are reused."""
        ids = ["x", "y", "z", "z2"] + ["%s%d" % (cs, i) for i in range(self.trig_count) for cs in "cs"] + ["d", "a"]
        if self.band_max > 2:
            ids += ["b"]
        if self.homogenized:
            ids += ["r2", "rn"]
            if self.derivative_order > 0:
                ids += ["q"]
        if self.language != "python":
            self.new_line()
            self.code += "%s %s;" % (self.float_type, ", ".join(ids))
        else:
            for i in range(self.derivative_order + 1):
                self.new_line()
                self.code += "%s = [[0.0] * %d for _ in range(%d)]" % (derivative_name(i), derivative_size(i), self.sh_count())
        for i, id in enumerate("xyz"):
            self.assign(id, "point[%d]" % i)
        self.assign("z2", "z * z")
        if self.homogenized:
            self.assign("r2", "x * x + y * y + z2")
            if self.derivative_order > 0:
                self.assign("q", "%s / r2" % self.literal(1.0))
        if self.derivative_order > 0:
            self.assign("c0", self.literal(1.0))
            self.assign("s0", self.literal(0.0))

    def generate_order(self, order):
        """
        Generates code for SH basis functions with the given absolute order.
        """
        # band = order: The associated Legendre polynomial is constant
        self.assign_shs(order, order, 1)
        # band = order + 1: The associated Legendre polynomial is linear
        self.assign_shs(order + 1, order, "z")
        # band > order + 1: The associated Legendre polynomial arises from a
        # recurrence
        times_r2 = " * r2" if self.homogenized else ""
        for l in range(order + 2, self.band_max + 1):
            # A ping-pong scheme for intermediate variables. dst becomes the
            # new associated Legendre polynomial (scaled to have 1 as leading
            # coefficient)
            dst = "ab"[(l - order) % 2]
            src = "ab"[(l - 1 - order) % 2]
            # The constant factor used in the recurrence for the associated
            # Legendre polynomial with lower band index
            factor = ((l - 1)**2 - order**2) / ((2 * l - 1) * (2 * l - 3))
            # The degree in z increases by 1 or 2 in each step, depending on
            # whether we want all bands
            if self.all_bands:
                factor_z = "z * "
            else:
                factor_z = "" if self.band_needed(l) else "z2 * "
            # The recurrence formulas are special-cased a bit for low-degree
            # cases
            if l == order + 2:
                if self.all_bands or self.band_needed(l):
                    self.assign(dst, "z2 - %s%s" % (self.literal(factor), times_r2))
                else:
                    self.assign(dst, "(z2 - %s%s) * z" % (self.literal(factor), times_r2))
            elif l == order + 3:
                if self.all_bands or not self.band_needed(l):
                    self.assign(dst, "%s(a - %s%s)" % (factor_z, self.literal(factor), times_r2))
                else:
                    self.assign(dst, "a - %s * z%s" % (self.literal(factor), times_r2))
            # This is the general case
            else:
                self.assign(dst, "%s%s - %s * %s%s" % (factor_z, src, self.literal(factor), dst, times_r2))
            self.assign_shs(l, order, dst)
        # Update sine and cosine
        if order == 0:
            self.assign("c1", "x")
            self.assign("s1", "y")
        elif order < self.band_max:
            cur, nxt = order % self.trig_count, (order + 1) % self.trig_count
            self.assign("c%d" % nxt, "x * c%d - y * s%d" % (cur, cur))
            self.assign("s%d" % nxt, "y * c%d + x * s%d" % (cur, cur))

    def homogenize(self):
        """
        Homogenizes SH basis functions if requested, assuming that they all
        already have a degree matching their band.
        """
        if not self.homogenized:
            return
        self.assign("rn", "r2")
        for n in range(2, self.band_max + 1, 2):
            l = self.band_max - n
            for m in range(-l, l + 1):
                lhs = "out_shs[%d]" % self.sh_index(l, m)
                self.assign(lhs, "%s * rn" % lhs)
            if n + 2 <= self.band_max:
                self.assign("rn", "rn * r2")

    def output_zeros(self):
        """Assigns 0.0 to everything not set yet."""
        if self.language == "python":
            # Python has zero-initialization anyway
            return
        set_output = frozenset(self.set_output)
        sh_count = self.sh_count()
        for i in range(1, self.derivative_order + 1):
            for j in range(sh_count):
                for k in range(derivative_size(i)):
                    if (i, j, k) not in set_output:
                        self.assign("%s[%d][%d]" % (derivative_name(i), j, k), self.literal(0.0))

    def end_function(self):
        """Ends the function definition."""
        if self.language != "python":
            self.code += "\n}\n"
        else:
            self.new_line()
            self.code += "return %s\n" % ", ".join([derivative_name(i) for i in range(self.derivative_order + 1)])


if __name__ == "__main__":
    # An example of code generation for the code on the poster
    print(SHCodeGeneration("eval_sh_0_2", 2, 1, False, False, False, "sloan", "glsl").generate())
