# This module provides classical simulations of basic quantum algorithms.

import random
import math
import cmath
import numpy


### CONSTANTS ###

# We haven't discussed this trivial case, but a 0-qbit state or gate is the
# complex scalar 1, represented as the following object. Notice that this
# object is neither the column vector numpy.array([1 + 0j]) nor the matrix
# numpy.array([[1 + 0j]]).
one = numpy.array(1 + 0j)

# Our favorite one-qbit states.
ket0 = numpy.array([1 + 0j, 0 + 0j])
ket1 = numpy.array([0 + 0j, 1 + 0j])
ketPlus = numpy.array([1 / math.sqrt(2), 1 / math.sqrt(2)])
ketMinus = numpy.array([1 / math.sqrt(2), -1 / math.sqrt(2)])

# Our favorite one-qbit gates.
i = numpy.array([
    [1 + 0j, 0 + 0j],
    [0 + 0j, 1 + 0j]])
x = numpy.array([
    [0 + 0j, 1 + 0j],
    [1 + 0j, 0 + 0j]])
y = numpy.array([
    [0 + 0j, 0 - 1j],
    [0 + 1j, 0 + 0j]])
z = numpy.array([
    [1 + 0j, 0 + 0j],
    [0 + 0j, -1 + 0j]])
h = numpy.array([
    [1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j],
    [1 / math.sqrt(2) + 0j, -1 / math.sqrt(2) + 0j]])

# Constant two-qbit gates.
cnot = numpy.array([
    [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j]])
swap = numpy.array([
    [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
    [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
    [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]])

# Constant three-qbit gates.
toffoli = numpy.array([
    [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j],
    [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j]])



### BIT STRINGS ###

# We represent an n-bit string --- that is, an element of {0, 1}^n --- in Python as a tuple of 0s and 1s.

def string(n, m):
    '''Converts a non-negative Python integer m to its corresponding bit string.
    As necessary, pads with leading 0s to bring the number of bits up to n.'''
    s = ()
    while m >= 1:
        s = (m % 2,) + s
        m = m // 2
    s = (n - len(s)) * (0,) + s
    return s

def integer(s):
    '''Converts a bit string s to its corresponding non-negative Python integer.'''
    m = 0
    for k in range(len(s)):
        m = 2 * m + s[k]
    return m

def next(s):
    '''Given an n-bit string s, returns the next n-bit string. The order is
    lexicographic, except that there is a string after 1...1, namely 0...0.'''
    k = len(s) - 1
    while k >= 0 and s[k] == 1:
        k -= 1
    if k < 0:
        return len(s) * (0,)
    else:
        return s[:k] + (1,) + (len(s) - k - 1) * (0,)

def nextTest(n):
    '''A unit test for some basic bit-string routines. Should print the integers
    from 0 to 2^n - 1.'''
    s = string(n, 0)
    m = integer(s)
    print(m)
    s = next(string(n, m))
    while s != n * (0,):
        m = integer(s)
        print(m)
        s = next(string(n, m))

def addition(s, t):
    '''Returns the mod-2 sum of two n-bit strings s and t.'''
    return tuple([(s[k] + t[k]) % 2 for k in range(len(s))])

def dot(s, t):
    '''Returns the mod-2 dot product of two n-bit strings s and t.'''
    return sum([s[k] * t[k] for k in range(len(s))]) % 2

def reduction(a):
    '''A is a list of m >= 1 bit strings of equal dimension n >= 1. In other
    words, A is a non-empty m x n binary matrix. Returns the reduced
    row-echelon form of A. A itself is left unaltered.'''
    b = a.copy()
    m = len(b)
    n = len(b[0])
    rank = 0
    for j in range(n):
        # Try to swap two rows to make b[rank, j] a leading 1.
        k = rank
        while k < m and b[k][j] == 0:
            i += 1
        if k != m:
            # Perform the swap.
            temp = b[k]
            b[k] = b[rank]
            b[rank] = temp
            # Reduce all leading 1s below the one we just made.
            for k in range(rank + 1, m):
                if b[k][j] == 1:
                    b[k] = addition(b[k], b[rank])
            rank += 1
    for j in range(n - 1, -1, -1):
        # Try to find the leading 1 in column j.
        k = m - 1
        while k >= 0 and b[k][j] != 1:
            k -= 1
        if k >= 0:
            # Use the leading 1 at b[k, j] to reduce 1s above it.
            for l in range(k):
                if b[l][j] == 1:
                    b[l] = addition(b[l], b[k])
    return b

def msb(x):
    '''Returns a mask of the most significant bit of the given integer x.'''
    count = 1
    while x | (x >> 1) != x:
        x |= x >> count
        count <<= 1
    return (x + 1) >> 1



### GENERATION ###

def function(n, m, f):
    '''Assumes that n, m >= 1. The argument f is a Python function that takes
    as input an n-bit string alpha and returns as output an m-bit string
    f(alpha). See deutschTest for examples of f. This function returns the
    (n + m)-qbit gate F that corresponds to f.'''
    F = numpy.zeros((2**(n + m), 2**(n + m)), dtype=numpy.array(0 + 0j).dtype)
    for a in range(2**n):
        for b in range(2**m):
            col = (a << m) | b
            row = (a << m) | (b ^ integer(f(string(n, a))))
            F[row, col] = 1
    return F

def fourier(n):
    '''Returns the n-qbit quantum Fourier transform gate T.'''
    T = numpy.zeros((2**n,2**n), dtype=numpy.array(0 + 0j).dtype)
    for a in range(2**n):
        for b in range(2**n):
            T[a, b] = numpy.exp(1j * 2 * math.pi * a * b / (2**n)) / (2**(n/2))
    return T

def get_ketrho(n):
    return numpy.array([2**(-n / 2) + 0j for k in range(1<<n)])

def get_r(n):
    return numpy.array([[2 * 2**(-n) + 0j - (1 if col == row else 0) for col in range(1<<n)] for row in range(1<<n)])



### OPERATIONS ###

def application(u, psi):
    '''Takes an n-qbit gate U and an n-qbit state |psi> and returns the output U|psi>.'''
    if u.shape[0] != u.shape[1]:
        print('ERROR: first argument must be a square matrix.')
        print('   Given numpy array with shape: {}'.format(u.shape))
        assert(u.shape[0] == u.shape[1])
    if u.shape[1] != psi.shape[0]:
        print('ERROR: requires the width of the matrix to equal the height of the vector.')
        print('    Given matrix with shape {} and vector with shape {}'.format(u.shape, psi.shape))
        assert(u.shape[1] == psi.shape[0])
    return numpy.dot(u, psi)

def tensor(a, b):
    '''Takes two n-qbit states or n-qbit gates and returns their tensor product'''
    if (a == numpy.array([1 + 0j])).all():
        return b.copy()
    if (b == numpy.array([1 + 0j])).all():
        return a.copy()
    if len(a.shape) != len(b.shape):
        print('ERROR: tensor product requires two states or gates of the same shape.')
        print('    Given numpy arrays with shapes: {} and {}'.format(a.shape, b.shape))
        assert(len(a.shape) == len(b.shape))
    n = a.shape[0]
    m = b.shape[0]
    vector = len(a.shape) == 1
    if (len(a.shape) > 2 or (len(a.shape) == 2 and a.shape[0] != a.shape[1]) or (n != msb(n))) or (
            len(b.shape) > 2 or (len(b.shape) == 2 and b.shape[0] != b.shape[1]) or (m != msb(m))):
        print('ERROR: tensor product requires vectors or square matrices with sizes which are a power of 2.')
        print('    Given numpy arrays with shapes: {} and {}'.format(a.shape, b.shape))
        exit(1)
    if vector:
        prod = numpy.zeros((n*m,), dtype=numpy.array(0 + 0j).dtype)
        for k in range(n):
            prod[k*m:(k+1)*m] = a[k] * b
        return prod
    else:
        prod = numpy.zeros((n*m, n*m), dtype=numpy.array(0 + 0j).dtype)
        for k in range(n):
            for j in range(n):
                prod[k*m:(k+1)*m, j*m:(j+1)*m] = a[k,j] * b
        return prod

def power(stateOrGate, m):
    '''Given an n-qbit gate or state and m >= 1, returns the mth tensor power,
    which is an (n * m)-qbit gate or state. Assumes n >= 1. For the sake of
    time and memory, m should be small.'''
    assert(msb(stateOrGate.shape[0]) == stateOrGate.shape[0])
    if m == 0:
        return numpy.array([1 + 0j])
    elif m == 1:
        return stateOrGate.copy()
    result = stateOrGate
    for k in range(1, m):
        result = tensor(result, stateOrGate)
    return result

def gcd(x, y):
    a = max(x, y)
    b = min(x, y)
    r = a % b
    while r != 0:
        a = b
        b = r
        r = a % b
    return b

def log(m, k, q):
    # m is modulo, k is base, q is remainder, returns l such that k**l = q (mod m)
    l = (k % m)
    count = 1
    while l != q and l % m != 1: #if l % m == 1, then there is a cycle and log does not exist (unless q == 1)
        l = (l * k) % m
        count += 1
    if l == q:
        return count
    else:
        return None

def period(m, k):
    return log(m, k, 1)

def periods(m):
    elements = [e for e in range(1, m) if gcd(e, m) == 1]
    print('Periods of elements in (Z/{}Z)* are:'.format(m))
    for k in elements:
        print('{}: {}'.format(k, period(m, k)))

def powerMod(k, l, m):
    '''Given a non-negative integer k, non-negative integer l, and positive
    integer m, Computes k^l mod m. Returns an integer in {0, ..., m - 1}.'''
    rax = 1
    while l > 0:
        if l & 1:
            rax = (rax * k) % m
        l >>= 1
        k = (k**2) % m
    return rax

def fraction(x0, j):
    '''x0 is a float in [0, 1). Calculates the continued fraction calculated
    with maximum recursion depth j. Returns the fractional approximation c/d
    as (c, d)'''
    if x0 == 0:
        return (0, 1)
    assert(x0 < 1)
    a0 = int(1 / x0)
    x1 = (1 / x0) - a0
    if j == 0:
        return (1, a0)
    else:
        c1, d1 = fraction(x1, j - 1)
        c = d1
        d = a0 * d1 + c1
        while gcd(c, d) > 1:
            gcd_cd = gcd(c, d)
            c = int(c / gcd_cd)
            d = int(d / gcd_cd)
        return (c, d)

def continuedFraction(n, m, x0):
    '''x0 is a float in [0, 1). Tries probing depths d = 0, 1, 2, ... until
    the resulting rational approximation x0 ~ c / d satisfies either d >= m or
    |x0 - c / d| <= 1 / 2^(n + 1). Returns a pair (c, d) with gcd(c, d) = 1.'''
    c = 0
    d = 1
    j = 0
    while d < m and abs(x0 - (c / d)) > 1 / 2**(n + 1):
        c, d = fraction(x0, j)
        j += 1
    return (c, d)

def distance(gate, m):
    '''Given an (n + 1)-qbit gate U (such as a controlled-V gate, where V is
    n-qbit), performs swaps to insert m extra wires between the first qbit and
    the other n qbits. Returns an (n + 1 + m)-qbit gate.'''
    n = round(math.log2(len(gate))) - 1
    if n == 1:
        swapper = swap
    else:
        swapper = tensor(swap, power(i, n - 1))
    u = gate
    for k in range(m):
        u = tensor(i, u)
        swapper = tensor(swapper, i)
        u = numpy.matmul(swapper, numpy.matmul(u, swapper))
    return u

def control(gate):
    '''Given an n-qbit gate U, returns the (n + 1)-qbit gate cU, in which the
    first qbit controls whether U is applied to the last n qbits.'''
    assert(gate.shape[0] == gate.shape[1])
    assert(msb(gate.shape[0]) == gate.shape[0])
    n_dim = gate.shape[0]
    cU = numpy.zeros((n_dim << 1, n_dim << 1), dtype=numpy.array(0 + 0j).dtype)
    for k in range(n_dim):
        cU[k, k] = 1
    cU[n_dim:, n_dim:] = gate
    return cU



### MEASUREMENT ###

def first(state):
    '''A function which takes an n-qbit state psi and performs a partial
    measurement on the first bit. Returns a tuple consisting of a classical
    one-qbit state (either ket0 or ket1) and an (n-1)-qbit state.'''
    if len(state.shape) != 1 or state.shape[0] != msb(state.shape[0]):
        print('ERROR: first requires an n-qbit state represented by a numpy array with shape (2^n,).')
        print('    Given numpy array with shape: {}'.format(state.shape))
        assert(len(state.shape) == 1)
        assert(state.shape[0] == msb(state.shape[0]))
    n = state.shape[0]
    chi = state[:n//2]
    sigma_sq = sum([ind * numpy.conj(ind) for ind in chi]).real
    omega = state[n//2:]
    tau_sq = sum([ind * numpy.conj(ind) for ind in omega]).real
    if random.random() < sigma_sq:
        return (ket0, (1 / math.sqrt(sigma_sq)) * chi)
    else:
        return (ket1, (1 / math.sqrt(tau_sq)) * omega)

def last(state):
    '''A function which takes an n-qbit state psi and performs a partial
    measurement on the last bit. Returns a tuple consisting of a classical
    one-qbit state (either ket0 or ket1) and an (n-1)-qbit state.'''
    if len(state.shape) != 1 or state.shape[0] != msb(state.shape[0]):
        print('ERROR: first requires an n-qbit state represented by a numpy array with shape (2^n,).')
        print('    Given numpy array with shape: {}'.format(state.shape))
        assert(len(state.shape) == 1)
        assert(state.shape[0] == msb(state.shape[0]))
    n = state.shape[0]
    chi = numpy.array([state[k*2] for k in range(n // 2)])
    sigma_sq = sum([ind * numpy.conj(ind) for ind in chi]).real
    omega = numpy.array([state[k*2+1] for k in range(n // 2)])
    tau_sq = sum([ind * numpy.conj(ind) for ind in omega]).real
    if random.random() < sigma_sq:
        return (ket0, (1 / math.sqrt(sigma_sq)) * chi)
    else:
        return (ket1, (1 / math.sqrt(tau_sq)) * omega)



### MISCELLANY ###

def uniform(n):
    '''Assumes n >= 0. Returns a uniformly random n-qbit state.'''
    if n == 0:
        return one
    else:
        psiNormSq = 0
        while psiNormSq == 0:
            reals = numpy.array(
                [random.normalvariate(0, 1) for k in range(2**n)])
            imags = numpy.array(
                [random.normalvariate(0, 1) for k in range(2**n)])
            psi = numpy.array([reals[k] + imags[k] * 1j for k in range(2**n)])
            psiNormSq = numpy.dot(numpy.conj(psi), psi).real
        psiNorm = math.sqrt(psiNormSq)
        return psi / psiNorm



### ALGORITHMS ###

def deutsch(f):
    '''Given a two-qbit gate representing a function f : {0, 1} -> {0, 1},
    outputs ket1 if f is constant and ket0 if f is not constant,'''
    # returns first((HxH)F(HxH)(ket1xket1)), where x indicates tensor product
    return first(application(tensor(h, h), application(f, application(tensor(h, h), tensor(ket1, ket1)))))[0]

def bernsteinVazirani(n, f):
    '''Given n >= 1 and an (n + 1)-qbit gate f representing a function
    {0, 1}^n -> {0, 1} defined by mod-2 dot product with an unknown w in
    {0, 1}^n, returns the list or tuple of n classical one-qbit states (ket0 or
    ket1) corresponding to w.'''
    # returns measurements of the first n qbits of (H^(n+1))F(H^(n+1))(|0...0>|1>)
    measurements = []
    state = application(power(h, n + 1), application(f, application(power(h, n + 1), tensor(power(ket0, n), ket1))))
    for i in range(n):
        bit, state = first(state)
        measurements.append(bit)
    return tuple(measurements)

def simon(n, f):
    '''The inputs are an integer n >= 1 and an (n + n - 1)-qbit gate f
    representing a function {0, 1}^n -> {0, 1}^(n-1) hiding an n-bit string w
    as in the Simon (1994) problem. Returns a list of n classical one-qbit
    states (ket0 or ket1) corresponding to a uniformly random bit string gamma
    that is perpendicular to w.'''
    state = application(f, application(tensor(power(h, n), power(i, n-1)), power(ket0, n + n - 1)))
    # state is now equal the product of the left half of the quantum circuit, including F
    for k in range(n-1):
        bit, state = last(state)
    # state is now an n-qbit state corresponding to the state directly prior to
    # applying the H^n and measurements in the top right of the quantum circuit
    state = application(power(h, n), state)
    gamma = []
    for k in range(n):
        bit, state = first(state)
        gamma.append(bit)
    return gamma

def simonComplete(n, f):
    '''Extracts the hidden n-bit delta from the given (n + n - 1)-qbit gate f
    representing a function {0, 1}^n -> {0, 1}^(n-1) as in the Simon (1994)
    problem. Returns a bit string (tuple of 0s and 1s) representing delta.'''
    gs = []
    while len(gs) < n - 1:
        gamma_str = simon(n, f)
        gamma = 0
        for bit in gamma_str:
            gamma = (gamma << 1) + (0 if (bit == ket0).all() else 1)
        # now have gamma as an n-bit number
        a = 0
        while a < len(gs):
            g = gs[a]
            if msb(gamma) > msb(g):
                # found the place to insert
                for b in range(a, len(gs)):
                    # mod out all later rows if it decreases their value
                    gamma = gamma ^ gs[b] if (gamma ^ gs[b] < gamma) else gamma
                gs.insert(a, gamma)
                break
            elif msb(gamma) == msb(g):
                # found a row with a matching msb, so mod out gamma with it
                gamma ^= g
            else:
                # mod out the current row with gamma if it decreases its value
                gs[a] = gamma ^ g if (gamma ^ g < g) else g
            a += 1
        if (a >= len(gs) or len(gs) == 0) and gamma:
            # if we didn't insert gamma before and gamma != 0
            gs.append(gamma)
    # Now we have a "matrix" of integers in rref
    # find unconstrained bit
    constrained = 0
    for g in gs:
        constrained |= msb(g)
    # unconstrained bit is only bit with value 0
    # treat it as a 1
    ones = (2**(n) - 1) ^ constrained
    zeros = 0
    a = 0
    while len(gs) > 0:
        # remove any 1s which are known to be zero
        gs[a] = gs[a] ^ (gs[a] & zeros)
        if gs[a] == msb(gs[a]):
            # all the other values in the row are zeros, so msb of the row is also 0
            zeros |= gs[a]
            gs.pop(a)
            if len(gs) > 0:
                a = a % len(gs)
        elif gs[a] == (msb(gs[a]) | (gs[a] & ones)):
            # all values in the row besides the msb are actually 1
            value = gs[a]
            k = 0
            value = 0
            while (msb(gs[a]) >> k) != 1:
                # XOR value with all non-msb bits of the string
                value ^= ((gs[a] >> k) & 1)
                k += 1
            # value is now equal to the true value of the msb bit
            if value:
                ones |= msb(gs[a])
            else:
                zeros |= msb(gs[a])
            gs.pop(a)
            if len(gs) > 0:
                a = a % len(gs)
        else:
            a = (a + 1) % len(gs)
    delta = string(n, ones)
    return delta

def shor(n, f):
    '''Assumes n >= 1. Given an (n + n)-qbit gate f representing a function
    f: {0, 1}^n -> {0, 1}^n of the form f(l) = k^l % m, returns a list of
    classical one-qbit states (ket0 or ket1) corresponding to an n-bit string
    that satisfies certain mathematical properties.'''
    state = application(f, application(tensor(power(h, n), power(i, n)), power(ket0, n<<1)))
    # state is now equal the product of the left half of the quantum circuit, including F
    for k in range(n):
        bit, state = last(state)
    # state is now an n-qbit state corresponding to the state directly prior to
    # applying T and measurements in the top right of the quantum circuit
    state = application(fourier(n), state)
    gamma = []
    for k in range(n):
        bit, state = first(state)
        gamma.append(bit)
    return gamma

def grover(n, f):
    '''Implements the Grover core subroutine with known k = 1. The f parameter
    is an (n + 1)-qbit gate representing an f: {0, 1^n} -> {0, 1} such that
    SUM_alpha f(alpha) = 1. Return a list or tuple of n classical one-qbit
    states (either ket0 or ket1), such that the corresponding n-bit string is
    usually equal to the alpha such that f(alpha) = 1.'''
    k = 1

    ketrho = get_ketrho(n)
    state = tensor(ketrho, application(h, ket1))
    r = get_r(n)
    r_apply_f = application(tensor(r, i), f)

    t = math.asin(k**(1 / 2) * 2**(-n / 2))
    raw_l = (math.pi / (4 * t)) - 0.5
    if raw_l % 1 < 0.5:
        l = int(raw_l)
    else:
        l = int(raw_l + 1)
    gate = r_apply_f * (1 if l & 1 else 0)
    l >>= 1
    while l > 0:
        r_apply_f = numpy.matmul(r_apply_f, r_apply_f)
        if l & 1:
            gate = gate + r_apply_f
        l >>= 1
    state = application(r_apply_f, state)
    output = []
    for x in range(n):
        bit, state = first(state)
        output.append(bit)
    return output



### Steane (1996) Seven-Bit Encoding

# The six crucial gates, which are also observables.
m07 = tensor(x, tensor(x, tensor(x, tensor(i, tensor(i, tensor(i, x))))))
m17 = tensor(x, tensor(x, tensor(i, tensor(x, tensor(i, tensor(x, i))))))
m27 = tensor(x, tensor(i, tensor(x, tensor(x, tensor(x, tensor(i, i))))))
n07 = tensor(z, tensor(z, tensor(z, tensor(i, tensor(i, tensor(i, z))))))
n17 = tensor(z, tensor(z, tensor(i, tensor(z, tensor(i, tensor(z, i))))))
n27 = tensor(z, tensor(i, tensor(z, tensor(z, tensor(z, tensor(i, i))))))

# Encoded gates except for CNOT.
i7 = power(i, 7)
x7 = power(x, 7)
z7 = power(z, 7)
h7 = power(h, 7)

# Encoding gate and encoded states
l7 = application(i7 + m07, application(i7 + m17, i7 + m27)) / 8
e7 = l7 * (2**(3 / 2))
ket07 = application(e7, power(ket0, 7))
ket17 = application(e7, power(ket1, 7))

# Code to check that the encoded states are orthonormal.
#print(numpy.dot(numpy.conj(ket07), ket07))
#print(numpy.dot(numpy.conj(ket07), ket17))
#print(numpy.dot(numpy.conj(ket17), ket17))

def error7(state7, report=False):
    '''Applies one of the 22 errors (including the trivial error) to state7,
    and returns the corrupted state.'''
    index = random.randrange(23)
    if index < 7:
        j = index
        error = tensor(power(i, 6 - j), tensor(x, power(i, j)))
        if report:
            print('Applied X error on bit', j)
        return application(error, state7)
    elif index < 14:
        j = index - 7
        error = tensor(power(i, 6 - j), tensor(y, power(i, j)))
        if report:
            print('Applied Y error on bit', j)
        return application(error, state7)
    elif index < 21:
        j = index - 14
        error = tensor(power(i, 6 - j), tensor(z, power(i, j)))
        if report:
            print('Applied Z error on bit', j)
        return application(error, state7)
    else:
        if report:
            print('Applied trivial I error')
        return state7

import time
def time_f(f, *args):
    start = time.time()
    rax = f(*args)
    end = time.time()
    print('Time for {}: {:.3f}s'.format(f.__name__, end - start))
    return rax

def detection7(state7, timed=False):
    '''Given a secen-qbit state that is the encoding of a one-qbit state, but
    possibly corrupted by one of the 22 errors. Implements the detection
    circuit. Returns a list or tuple consisting of seven elements: six
    classical one-qbit states (ket0 or ket1) and one seven-qbit state.'''
    state = tensor(application(power(h, 6), power(ket0, 6)), state7)
    observables = [n07, n17, n27, m07, m17, m27]
    if timed:
        for k in range(6):
            print('Beginning k =', k)
            cgate = time_f(control, observables[k])
            distanced = time_f(distance, cgate, k)
            i_padding = time_f(power, i, 5 - k)
            layer = time_f(tensor, i_padding, distanced)
            state = time_f(application, layer, state)
            print('Ending k =', k)
    else:
        for k in range(6):
            cgate = control(observables[k])
            distanced = distance(cgate, k)
            i_padding = power(i, 5 - k)
            layer = tensor(i_padding, distanced)
            state = application(layer, state)
    state = application(tensor(power(h, 6), i7), state)
    syndrome = []
    for k in range(6):
        bit, state = first(state)
        syndrome.append(bit)
    syndrome.append(state)
    return syndrome

def correction7(diagnosis):
    '''Given the output from detection7, returns the uncorrupted seven-qbit state.'''
    syndrome = tuple([(0 if (x == ket0).all() else 1) for x in diagnosis[:-1]])
    if syndrome[0:3] == (0, 0, 0):
        # No Z error has occurred. Worry about only X errors.
        if syndrome[3:6] == (1, 1, 1):
            correction = tensor(power(i, 0), tensor(x, power(i, 6)))
        elif syndrome[3:6] == (0, 1, 1):
            correction = tensor(power(i, 1), tensor(x, power(i, 5)))
        elif syndrome[3:6] == (1, 0, 1):
            correction = tensor(power(i, 2), tensor(x, power(i, 4)))
        elif syndrome[3:6] == (1, 1, 0):
            correction = tensor(power(i, 3), tensor(x, power(i, 3)))
        elif syndrome[3:6] == (1, 0, 0):
            correction = tensor(power(i, 4), tensor(x, power(i, 2)))
        elif syndrome[3:6] == (0, 1, 0):
            correction = tensor(power(i, 5), tensor(x, power(i, 1)))
        elif syndrome[3:6] == (0, 0, 1):
            correction = tensor(power(i, 6), tensor(x, power(i, 0)))
        else:
            correction = power(i, 7)
    elif syndrome[3:6] == (0, 0, 0):
        # No X error has occurred. Worry about only Z errors.
        if syndrome[0:3] == (1, 1, 1):
            correction = tensor(power(i, 0), tensor(z, power(i, 6)))
        elif syndrome[0:3] == (0, 1, 1):
            correction = tensor(power(i, 1), tensor(z, power(i, 5)))
        elif syndrome[0:3] == (1, 0, 1):
            correction = tensor(power(i, 2), tensor(z, power(i, 4)))
        elif syndrome[0:3] == (1, 1, 0):
            correction = tensor(power(i, 3), tensor(z, power(i, 3)))
        elif syndrome[0:3] == (1, 0, 0):
            correction = tensor(power(i, 4), tensor(z, power(i, 2)))
        elif syndrome[0:3] == (0, 1, 0):
            correction = tensor(power(i, 5), tensor(z, power(i, 1)))
        elif syndrome[0:3] == (0, 0, 1):
            correction = tensor(power(i, 6), tensor(z, power(i, 0)))
        else:
            correction = power(i, 7)
    else:
        # A Y error has occurred.
        if syndrome[0:3] == (1, 1, 1):
            correction = tensor(power(i, 0), tensor(y, power(i, 6)))
        elif syndrome[0:3] == (0, 1, 1):
            correction = tensor(power(i, 1), tensor(y, power(i, 5)))
        elif syndrome[0:3] == (1, 0, 1):
            correction = tensor(power(i, 2), tensor(y, power(i, 4)))
        elif syndrome[0:3] == (1, 1, 0):
            correction = tensor(power(i, 3), tensor(y, power(i, 3)))
        elif syndrome[0:3] == (1, 0, 0):
            correction = tensor(power(i, 4), tensor(y, power(i, 2)))
        elif syndrome[0:3] == (0, 1, 0):
            correction = tensor(power(i, 5), tensor(y, power(i, 1)))
        elif syndrome[0:3] == (0, 0, 1):
            correction = tensor(power(i, 6), tensor(y, power(i, 0)))
        else:
            correction = power(i, 7)
    return application(correction, diagnosis[-1])

def decoding7(state7):
    '''Assumes that the given 7-qbit state is an uncorrupted encoding of a
    classical one-qbit state. Returns the one-qbit state.'''
    if numpy.dot(ket07, state7) > numpy.dot(ket17, state7):
        return ket0
    else:
        return ket1



### TESTS ###

def firstTest():
    '''Constructs an unentangled two-qbit state |0> |psi> or |1> |psi>,
    measures the first qbit, and then reconstructs the state.'''
    print("One should see 0s.")
    psi = uniform(1)
    state = tensor(ket0, psi)
    meas = first(state)
    print(state - tensor(meas[0], meas[1]))
    psi = uniform(1)
    state = tensor(ket1, psi)
    meas = first(state)
    print(state - tensor(meas[0], meas[1]))

def firstTest345(n, m):
    '''Assumes n >= 1. Uses one more qbit than that, so that the total number
    of qbits is n + 1. The parameter m is how many tests to run. Should return
    a number close to 0.64 for large m.'''
    psi0 = 3 / 5
    beta = uniform(n)
    psi1 = 4 / 5
    gamma = uniform(n)
    chi = psi0 * tensor(ket0, beta) + psi1 * tensor(ket1, gamma)
    def f():
        if (first(chi)[0] == ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for k in range(m):
        acc += f()
    return acc / m

def lastTest():
    '''Constructs an unentangled two-qbit state |0> |psi> or |1> |psi>,
    measures the last qbit, and then reconstructs the state.'''
    print("One should see 0s.")
    psi = uniform(1)
    state = tensor(psi, ket0)
    meas = last(state)
    print(state - tensor(meas[1], meas[0]))
    #psi = uniform(1)
    state = tensor(psi, ket1)
    meas = last(state)
    print(state - tensor(meas[1], meas[0]))

def lastTest345(n, m):
    '''Assumes n >= 1. Uses one more qbit than that, so that the total number of
    qbits is n + 1. The parameter m is how many tests to run. Should return a
    number close to 0.64 for large m.'''
    psi0 = 3 / 5
    beta = uniform(n)
    psi1 = 4 / 5
    gamma = uniform(n)
    chi = psi0 * tensor(beta, ket0) + psi1 * tensor(gamma, ket1)
    def f():
        if (last(chi)[0] == ket0).all():
            return 0
        else:
            return 1
    acc = 0
    for k in range(m):
        acc += f()
    return acc / m

def tensorTest():
    print(tensor(h, h))
    print(tensor(i, h))
    print(tensor(h, i))
    print(tensor(ket0, ket0))
    print(tensor(ket0, ket1))
    print(tensor(ket1, ket0))
    print(tensor(ket1, ket1))

def deutschTest():
    print('One should see ket0s.')
    def f(x):
        return (1 - x[0],)
    print(deutsch(function(1, 1, f)))
    def f(x):
        return x
    print(deutsch(function(1, 1, f)))
    print('One should see ket1s.')
    def f(x):
        return (0,)
    print(deutsch(function(1, 1, f)))
    def f(x):
        return (1,)
    print(deutsch(function(1, 1, f)))

def bernsteinVaziraniTest(n):
    print('Testing Bernstein Vazirani...')
    delta = string(n, random.randrange(2**n))
    f = function(n, 1, lambda x: (dot(delta, x),))
    bv_delta_states = bernsteinVazirani(n, f)
    bv_delta = tuple([(0 if (bv_delta_states[k] == ket0).all() else 1) for k in range(len(bv_delta_states))])
    print('Actual delta:')
    print(delta)
    print('Berstein-Vazirani delta:')
    print(bv_delta)

def simonTest(n):
    print('Testing Simon...')
    if n > 7:
        print('You really don\'t want to run this with n > 7.')
        if input('Do you want to continue anyway? [y/n] ').lower() != 'y':
            quit()
    true_delta = random.randrange(2**n)
    mapping = [-1 for k in range(2**n)]  # set all outputs to -1 as a placeholder. Python uses sign and magnitude so this is okay.
    available = [k for k in range(2**(n-1))]
    for k in range(2**n):
        if len(available) > 0 and mapping[k] == -1:  # mapping[k ^ true_delta] must also be -1 since ^ is bijective
            index = random.randrange(len(available))
            output = available[index]
            available.pop(index)
            # map both k and (k XOR delta) to map to the given available output
            mapping[k] = output
            mapping[k ^ true_delta] = output
    f = function(n, n-1, lambda x: string(n, mapping[integer(x)]))
    delta = simonComplete(n, f)
    print('Actual delta:  {}'.format(string(n, true_delta)))
    print('Simon delta:   {}'.format(delta))

def shorTest(n, m):
    '''Tests Shor's algorithm by generating a random k that is coprime to m,
    builds a function f that computers powers of k modulo m, converts it to
    gate F, and runs Shor's quantum core subroutine.'''
    print('Testing Shor...')
    k = m
    while gcd(k, m) != 1:
        k = random.randrange(1, 1<<n)
    print('Given n = {}, m = {}, generated random k = {}'.format(n, m, k))
    f = function(n, n, lambda l: string(n, powerMod(k, integer(l), m)))
    def get_d(n, m, f):
        d = m
        while d >= m:
            b = 0
            b_list = shor(n, f)
            for bit in b_list:
                b <<= 1
                if (bit == ket1).all():
                    b += 1
            c, d = continuedFraction(n, m, b / (2**n))
        return d
    p = -1
    while p == -1:
        d = get_d(n, m, f)
        if k**d % m == 1:
            p = d
            break
        d1 = get_d(n, m, f)
        if k**d1 % m == 1:
            p = d1
            break
        lcm = int(d * d1 / gcd(d, d1))
        if k**lcm % m == 1:
            p = lcm
            break
    print('The period calculated using Shor\'s algorithm was found to be {}'.format(p))
    print('The period of k ({}) mod m ({}) = {}'.format(k, m, period(m, k)))

def groverTest(n):
    '''Tests Grover's algorithm with a known k=1 by generating a random delta
    and defining a function f: {0, 1}^n -> {0, 1} such that f(delta) = 1 and
    f(alpha) = 0 for all alpha != delta. Builds an (n+1)-qbit gate from that
    f and provides it to the grover() function, which returns the delta it
    discovered (and did not verify).'''
    print('Testing Grover...')
    true_delta = random.randrange(1<<n)
    print('True delta:   ', string(n, true_delta))
    f = function(n, 1, lambda x: (1,) if integer(x) == true_delta else (0,))
    grover_delta = tuple([0 if (d == ket0).all() else 1 for d in grover(n, f)])
    print('Grover delta: ', grover_delta)

def steaneTest7(time_detection=False):
    '''Randomly picks ket0 or ket1, encodes it to the Stean (1996) 7-bit
    encoding, introduces a single error (or the trivial non-error), then
    detects and corrects the error.'''
    print('Testing Steane...')
    if random.random() < 0.5:
        orig_bit, encoded = ket0, ket07
    else:
        orig_bit, encoded = ket1, ket17
    print('Original bit:', orig_bit)
    errored = error7(encoded, time_detection)
    detected = detection7(errored, time_detection)
    print('Error syndrome:', tuple([(0 if (state == ket0).all() else 1) for state in detected[:-1]]))
    corrected = correction7(detected)
    final_bit = decoding7(corrected)
    print('Final bit:', final_bit)


### MAIN ###

def main():
    firstTest()
    lastTest()
    #tensorTest()
    deutschTest()
    print('One should see 0.64:', firstTest345(4, 10000))
    print('One should see 0.64:', lastTest345(4, 10000))
    bernsteinVaziraniTest(8)
    simonTest(6)
    shorTest(5, 5)
    shorTest(6, 7)
    shorTest(7, 11)
    #shorTest(8, 15)
    groverTest(8)
    steaneTest7(True)

if __name__ == "__main__":
    main()
