import time
import copy
from random import randint
from functools import lru_cache, wraps
from collections import namedtuple


def fib_ordinary0(n):
    if n < 2:
        return n
    else:
        return fib_ordinary0(n-1) + fib_ordinary0(n-2)


def fib_ordinary1(n):
    def fibrec(n):
        if n < 2:
            return n
        if v[n] is None:
            v[n] = fibrec(n - 1) + fibrec(n - 2)

        return v[n]

    v = [None for i in range(n + 1)]
    return fibrec(n)


def fib_ordinary2(n):
    def f(u, v, i):
        if i == 0:
            return u
        return f(v, u + v, i - 1)

    return f(0, 1, n)


def fib_ordinary3(n):
    v = [0, 1]
    for i in range(2, n + 1):
        v.append(v[i - 1] + v[i - 2])

    return v[n]


def fib_ordinary4(n):
    u, v = 0, 1
    for i in range(n):
        u, v = v, v + u
    return u


isodd = lambda x: bool(x & 0x1)

def fib_record(n):
    def fib(n):
        if n not in numbers:
            numbers[n] = 0
        numbers[n] += 1
        if n < 2:
            return n
        x = fib((n >> 1) - 1)
        y = fib(n >> 1)
        if isodd(n):
            x += y
            return x * x + y * y
        else:
            return y * (y + 2 * x)
    numbers = {}
    return (fib(n), numbers)


def fib_fast_memoize(n):
    def dp(n):
        if n not in F:
            k = n >> 1
            dp(k - 1)
            dp(k)
            if isodd(n):
                F[k + 1] = F[k] + F[k - 1]
                F[n] = F[k]**2 + F[k + 1]**2
            else:
                F[n] = F[k] * (F[k] + 2 * F[k - 1])

    F = {0: 0, 1: 1}
    dp(n)

    return F[n]


@lru_cache(maxsize=128)
def fib_fast_lur_cache(n):
    if n < 16:
        return fib_ordinary4(n)
    k = n >> 1
    x = fib_fast_lur_cache(k - 1)
    y = fib_fast_lur_cache(k)
    if isodd(n):
        x += y
        return x * x + y * y
    else:
        return y * (y + 2 * x)


def fib_fast_bottom_up1(n):
    x, y, l = 1, 0, n.bit_length()
    for i in range(l - 1, 0, -1):
        xx = x**2
        yy = y**2
        xy = y * (y + 2 * x)
        if isodd(n>>i):
            x, y = xy, xx + yy
            y += x
        else:
            y, x = xy, xx + yy

    if isodd(n):
        x += y
        return x * x + y * y
    return y * (y + 2 * x)


def fib_fast_bottom_up2(n):
    x, y, l = 1, 0, n.bit_length()
    for i in range(l - 1, 0, -1):
        xx = x**2
        yy = y**2
        xy = (x + y)**2
        if isodd(n>>i):
            x = xy - xx
            y = xy + yy
        else:
            x = xx + yy
            y = xy - xx

    xy = (x + y)**2
    if isodd(n):
        return xy + y**2
    else:
        return xy - x**2


def fib_fast_bottom_up3(n):
    l = n.bit_length()
    F = {0: 0, 1: 1, 2: 1, 3: 2}
    for i in range(l - 3, -1, -1):
        m = n >> i
        k = m >> 1
        m -= isodd(m)
        g = -2 if isodd(k) else 2
        x = F[k-1]**2
        y = F[k]**2
        u = x + y
        v = 4 * y - x + g

        F[m-1] = u
        F[m] = v - u
        F[m+1] = v

    # print("up3: %03d:"%(n), sorted(F))
    return F[n]


def fib_fast_bottom_up4(n):
    if n < 2:
        return n
    x, y = 0, 1
    for i in range(n.bit_length() - 2, 0, -1):
        m = n >> i
        k = m >> 1
        g = -2 if isodd(k) else 2

        xx = x**2
        yy = y**2
        xy = xx + yy
        yx = 4 * yy - xx + g
        if isodd(m):
            x, y = yx - xy, yx
        else:
            x, y = xy, yx - xy

    k = n >> 1
    g = -2 if isodd(k) else 2

    xx = x**2
    yy = y**2
    xy = 4 * yy - xx + g
    if not isodd(n):
        xy -= xx + yy

    return xy


def fib_fast_bottom_up5(n):
    if n < 2:
        return n

    x, y = 1, 1
    for i in range(n.bit_length() - 1, 0, -1):
        if isodd(n>>i):
            x = y - x
            g = -2
        else:
            y = y - x
            g = 2

        u = x**2
        v = y**2
        x = u + v
        y = v * 4 - u + g

    if not isodd(n):
        y = y - x
    return y


def cache_fib_numbers_squares(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result

    numbers = None
    squares = None

    def cache_clear():
        nonlocal numbers, squares
        numbers = { 0: 0, 1: 1, 2: 1, 3: 2 }
        squares = { 0: 0, 1: 1, 2: 1, 3: 4 }

    def cache_table():
        return {"numbers": numbers.copy(), "squares": squares.copy()}

    wrapper.cache_clear = cache_clear
    wrapper.cache_table = cache_table
    wrapper.__cache_dict = lambda: (numbers, squares)

    cache_clear()

    return wrapper


@cache_fib_numbers_squares
def fib_fast_bottom_up_cached1(n):
    F, S = fib_fast_bottom_up_cached1.__cache_dict()

    if n not in F:
        for i in range(n.bit_length() - 3, -1, -1):
            m = n >> i
            if m in F and m - 1 in F:
                continue
            k = m >> 1
            m -= isodd(m)
            g = -2 if isodd(k) else 2

            if k - 1 not in S:
                S[k-1] = F[k-1]**2
            if k not in S:
                S[k] = F[k]**2

            x = S[k-1]
            y = S[k]
            u = x + y
            v = 4 * y - x + g

            F[m-1] = u
            F[m] = v - u
            F[m+1] = v

    return F[n]


@cache_fib_numbers_squares
def fib_fast_bottom_up_cached2(n):
    F, S = fib_fast_bottom_up_cached2.__cache_dict()

    if n not in F:
        for i in range(n.bit_length() - 3, -1, -1):
            m = n >> i
            if m in F and m - 1 in F:
                continue

            k = m >> 1
            m -= isodd(m)
            g = -2 if isodd(k) else 2

            if k - 1 not in S:
                S[k-1] = F[k-1]**2

            if k not in S:
                S[k] = F[k]**2

            x, y = S[k-1], S[k]

            if m - 1 not in F:
                F[m-1] = x + y

            if m not in F:
                F[m+1] = 4 * y - x + g
                F[m] = F[m+1] - F[m-1]

    return F[n]


def fib_fast_matrix_expt(n):
    def mul22(x, y):
        return [
            [   x[0][0] * y[0][0] + x[0][1] * y[1][0],
                x[0][0] * y[1][0] + x[0][1] * y[1][1]
            ],
            [
                x[1][0] * y[0][0] + x[1][1] * y[1][0],
                x[1][0] * y[1][0] + x[1][1] * y[1][1]
            ]
        ]

    def fast_expt(x, n):
        if n == 0:
            return [[1, 0], [0, 1]]

        m = fast_expt(x, n >> 1)
        y = mul22(m, m)

        if isodd(n):
            y = mul22(y, x)
        return y

    if n > 1:
        return fast_expt([[1, 1], [1, 0] ], n - 1)[0][0]
    return n



if __name__ == "__main__":
    functions = [
        fib_fast_bottom_up_cached1,
        fib_fast_bottom_up_cached2,
        fib_fast_bottom_up1,
        fib_fast_bottom_up2,
        fib_fast_bottom_up3,
        fib_fast_bottom_up4,
        fib_fast_bottom_up5,
        fib_fast_memoize,
        fib_fast_lur_cache,
        fib_fast_matrix_expt
    ]

    def bench(function, numbers):
        if not hasattr(function, "elapsed"):
            function.elapsed = 0

        s = time.time()
        for i in numbers:
            function(i)
        e = time.time()

        elapsed = e - s

        print("%32s: %2.4fs" % (function.__name__, elapsed))

        function.elapsed += elapsed


    def gen_randnums(n, mini=0, maxi=10000000):
        return [randint(mini, maxi) for i in range(n)]


    for i in range(101):
        y = fib_ordinary1(i)
        assert y == fib_ordinary2(i)
        assert y == fib_ordinary3(i)
        assert y == fib_ordinary4(i)
        assert y == fib_fast_bottom_up_cached1(i)
        assert y == fib_fast_bottom_up_cached2(i)
        assert y == fib_fast_bottom_up1(i)
        assert y == fib_fast_bottom_up2(i)
        assert y == fib_fast_bottom_up3(i)
        assert y == fib_fast_bottom_up4(i)
        assert y == fib_fast_bottom_up5(i)
        assert y == fib_fast_memoize(i)
        assert y == fib_fast_lur_cache(i)
        assert y == fib_fast_matrix_expt(i)


    for e in range(1, 8):
        numbers = gen_randnums(10, maxi=10**e)
        for i in range(2):
            for function in functions:
                bench(function, numbers)
            print()


    print("Time Elapsed:")
    for function in functions:
        print("%32s: %2.4fs" % (function.__name__, function.elapsed))
