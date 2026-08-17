"""Brazilian identifier validators used by PII recognizers."""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\D")


def _digits(value: str) -> str:
    return _DIGITS.sub("", value)


def _all_same(value: str) -> bool:
    return len(set(value)) <= 1


def is_valid_cpf(value: str) -> bool:
    cpf = _digits(value)
    if len(cpf) != 11 or _all_same(cpf):
        return False
    for idx in (9, 10):
        s = sum(int(cpf[i]) * ((idx + 1) - i) for i in range(idx))
        check = (s * 10) % 11
        if check == 10:
            check = 0
        if check != int(cpf[idx]):
            return False
    return True


def is_valid_cnpj(value: str) -> bool:
    cnpj = _digits(value)
    if len(cnpj) != 14 or _all_same(cnpj):
        return False
    weights_first = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    weights_second = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    for weights, idx in ((weights_first, 12), (weights_second, 13)):
        s = sum(int(cnpj[i]) * weights[i] for i in range(len(weights)))
        check = s % 11
        check = 0 if check < 2 else 11 - check
        if check != int(cnpj[idx]):
            return False
    return True


def is_valid_cnh(value: str) -> bool:
    cnh = _digits(value)
    if len(cnh) != 11 or _all_same(cnh):
        return False
    dsc = 0
    first = 0
    for i, mult in enumerate(range(9, 0, -1)):
        first += int(cnh[i]) * mult
    first = first % 11
    if first >= 10:
        first = 0
        dsc = 2
    if first != int(cnh[9]):
        return False

    second = 0
    for i, mult in enumerate(range(1, 10)):
        second += int(cnh[i]) * mult
    second = (second % 11) - dsc
    if second < 0:
        second += 11
    if second >= 10:
        second = 0
    return second == int(cnh[10])


def is_valid_titulo_eleitor(value: str) -> bool:
    t = _digits(value)
    if len(t) != 12 or _all_same(t):
        return False
    state = int(t[8:10])
    if state < 1 or state > 28:
        return False
    d1 = sum(int(t[i]) * (i + 2) for i in range(8)) % 11
    d1 = 0 if d1 == 10 else d1
    d2 = (int(t[8]) * 7 + int(t[9]) * 8 + d1 * 9) % 11
    d2 = 0 if d2 == 10 else d2
    return d1 == int(t[10]) and d2 == int(t[11])


def is_valid_pis(value: str) -> bool:
    p = _digits(value)
    if len(p) != 11 or _all_same(p):
        return False
    weights = [3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    s = sum(int(p[i]) * weights[i] for i in range(10))
    check = 11 - (s % 11)
    if check >= 10:
        check = 0
    return check == int(p[10])
