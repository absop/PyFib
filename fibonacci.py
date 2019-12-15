def fib0(n):
    if n < 2:
        return n
    else:
        return fib0(n-1) + fib0(n-2)


def fib1(n):
    def fibrec(n):
        if n < 2:
            return n
        if v[n] is None:
            v[n] = fibrec(n - 1) + fibrec(n - 2)

        return v[n]

    v = [None for i in range(n + 1)]
    return fibrec(n)


def fib2(n):
    def f(u, v, i):
        if i == 0:
            return u
        return f(v, u + v, i - 1)

    return f(0, 1, n)


def fib3(n):
    v = [0, 1]
    for i in range(2, n + 1):
        v.append(v[i - 1] + v[i - 2])

    return v[n]


def fib4(n):
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
        if n & 0x1:
            x += y
            return x * x + y * y
        else:
            return y * (y + 2 * x)
    numbers = {}
    return (fib(n), numbers)


def fast_fib_memory1(n):
    def dp(F, n):
        if n not in F:
            k = n >> 1
            dp(F, k - 1)
            dp(F, k)
            if (n & 0x1) == 1:
                F[k + 1] = F[k] + F[k - 1]
                F[n] = F[k] * F[k] + F[k + 1] * F[k + 1]
            else:
                F[n] = F[k] * (F[k] + 2 * F[k - 1])

    memory = {0: 0, 1: 1}
    dp(memory, n)

    return memory[n]


def fast_fib_memory2(n):
    def dp(F, n):
        if n not in F:
            k = n >> 1
            dp(F, k)
            if (n & 0x1) == 1:
                dp(F, k + 1)
                F[n] = F[k]**2 + F[k + 1]**2
            else:
                dp(F, k - 1)
                F[n] = F[k] * (F[k] + 2 * F[k - 1])

    memory = {0: 0, 1: 1}
    dp(memory, n)

    return memory[n]


from functools import lru_cache
@lru_cache(maxsize=128)
def fast_fib_lur_cache(n):
    if n < 2:
        return n
    x = fast_fib_lur_cache((n >> 1) - 1)
    y = fast_fib_lur_cache(n >> 1)
    if isodd(n):
        x += y
        return x * x + y * y
    else:
        return y * (y + 2 * x)


def fast_fib_bottom_up(n):
    x, y, l = 0, 1, n.bit_length()
    for i in range(l - 1, 0, -1):
        x, y = x * (x + 2 * y), (x * x + y * y)
        if isodd(n >> i):
            x, y = x + y, x

    r = x * (x + 2 * y)
    if isodd(n):
        r += x * x + y * y
    return r


def fib_fast_expt(n):
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


for i in range(101):
    assert fib1(i) == fib2(i)
    assert fib1(i) == fib3(i)
    assert fib1(i) == fib4(i)
    assert fib1(i) == fast_fib_memory1(i)
    assert fib1(i) == fast_fib_memory2(i)
    assert fib1(i) == fast_fib_lur_cache(i)
    assert fib1(i) == fast_fib_bottom_up(i)
    assert fib1(i) == fib_fast_expt(i)
    # print("fib1(%3d) = %d" % (i, fib1(i)))
