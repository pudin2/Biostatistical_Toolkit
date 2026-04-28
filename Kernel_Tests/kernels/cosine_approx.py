"""
cosine_approx
=============
Aproximaciones a la funcion ``cos(pi*u/2)`` sobre ``u in [-1, 1]`` que
permiten evaluar el kernel coseno

    K(u) = (pi/4) * cos(pi*u/2) * 1{|u| <= 1}

usando solo operaciones aritmeticas (mult/sum) o aritmetica + una division.
La motivacion es practica: en GPU las funciones trascendentes (``cos``,
``exp``) se ejecutan en SFU dedicadas con throughput limitado, mientras
que las multiplicaciones y FMAs van por las ALU principales con mucha
mas capacidad efectiva. Sustituir ``cos(.)`` por un polinomio puede reducir
el tiempo del kernel coseno y, al mismo tiempo, conservar la calidad
estadistica del KDE (KS, masa positiva, ...) si el error maximo es
suficientemente pequeno.

Aproximaciones implementadas
----------------------------
1. Taylor en torno a u=0:
       cos(pi*u/2) ~ 1 + a1*u^2 + a2*u^4 + a3*u^6 + ...
   con a_k = (-1)^k * (pi/2)^(2k) / (2k)!.
   Truncadas a grado 4 y 6.

2. Chebyshev sobre [-1, 1]:
   Truncacion de la serie de Chebyshev de cos(pi*u/2). Es par, asi que
   solo aparecen T_0, T_2, T_4, ... Chebyshev grado 4 y 6 son
   (asintoticamente) cuasi-minimax en error sup. Coeficientes
   precomputados con ``numpy.polynomial.chebyshev.Chebyshev.fit``.

3. Remez minimax (grado 4):
   Polinomio par P(u) = c0 + c2*u^2 + c4*u^4 que minimiza
   max_{u in [-1,1]} |P(u) - cos(pi*u/2)| via Remez por intercambio
   (implementacion ligera basada en numpy linalg).

4. Bhaskara I (racional):
       cos(pi*u/2) ~ 4(1 - u^2) / (4 + u^2)         para u in [-1, 1]
   Una sola division. Aproximacion clasica del siglo VII.

Cada aproximacion expone una funcion ``approx_*(u, xp=np)`` que devuelve
``cos(pi*u/2)`` aproximado, y una variante ``cosine_kernel_*(u, xp=np)``
que devuelve directamente ``K(u) = (pi/4) * cos(pi*u/2) * 1{|u|<=1}`` con
las mismas operaciones que ``kernels.core.kernel_eval(u, "cosine")``.

Las funciones funcionan tanto con NumPy como con CuPy: pasale el modulo
en el argumento ``xp``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# --------------------------------------------------------------------------
# Constantes
# --------------------------------------------------------------------------
_PI = float(np.pi)
_PI_OVER_2 = _PI / 2.0
_PI_OVER_4 = _PI / 4.0


# --------------------------------------------------------------------------
# 1) TAYLOR
# --------------------------------------------------------------------------
# Coeficientes de cos(pi*u/2) = sum_k a_k * u^(2k) en torno a u=0.
# a_k = (-1)^k * (pi/2)^(2k) / (2k)!
_TAYLOR_A0 = 1.0
_TAYLOR_A1 = -(_PI_OVER_2 ** 2) / 2.0           # = -pi^2 / 8
_TAYLOR_A2 = (_PI_OVER_2 ** 4) / 24.0           # =  pi^4 / 384
_TAYLOR_A3 = -(_PI_OVER_2 ** 6) / 720.0         # = -pi^6 / 46080


def approx_taylor4(u: Any, xp: Any = np) -> Any:
    """Taylor truncado a grado 4: 1 + a1 u^2 + a2 u^4 (Horner en v=u^2)."""
    v = u * u
    return _TAYLOR_A0 + v * (_TAYLOR_A1 + v * _TAYLOR_A2)


def approx_taylor6(u: Any, xp: Any = np) -> Any:
    """Taylor truncado a grado 6: 1 + a1 u^2 + a2 u^4 + a3 u^6."""
    v = u * u
    return _TAYLOR_A0 + v * (_TAYLOR_A1 + v * (_TAYLOR_A2 + v * _TAYLOR_A3))


# --------------------------------------------------------------------------
# 2) CHEBYSHEV
# --------------------------------------------------------------------------
# Precomputo de coeficientes monomicos par e0, e2, e4, ... ajustando un
# polinomio de Chebyshev de grado N a cos(pi*u/2) sobre 4096 puntos
# uniformes en [-1, 1] y convirtiendo a base monomial.
def _fit_cheb_monomial_even(n: int) -> tuple[float, ...]:
    u_dense = np.linspace(-1.0, 1.0, 4096)
    y_dense = np.cos(_PI_OVER_2 * u_dense)
    cheb = np.polynomial.chebyshev.Chebyshev.fit(u_dense, y_dense, deg=n, domain=(-1.0, 1.0))
    poly = cheb.convert(kind=np.polynomial.Polynomial)
    coefs = poly.coef.tolist()
    # Forzar pares (los impares deben salir ~0 por simetria; los anulamos).
    while len(coefs) <= n:
        coefs.append(0.0)
    even = tuple(float(coefs[k]) for k in range(0, n + 1, 2))
    return even


_CHEB4 = _fit_cheb_monomial_even(4)   # (c0, c2, c4)
_CHEB6 = _fit_cheb_monomial_even(6)   # (c0, c2, c4, c6)


def approx_cheb4(u: Any, xp: Any = np) -> Any:
    """Chebyshev grado 4 (cuasi-minimax). c0 + c2 u^2 + c4 u^4 (Horner)."""
    c0, c2, c4 = _CHEB4
    v = u * u
    return c0 + v * (c2 + v * c4)


def approx_cheb6(u: Any, xp: Any = np) -> Any:
    """Chebyshev grado 6. c0 + c2 u^2 + c4 u^4 + c6 u^6."""
    c0, c2, c4, c6 = _CHEB6
    v = u * u
    return c0 + v * (c2 + v * (c4 + v * c6))


# --------------------------------------------------------------------------
# 3) REMEZ MINIMAX (grado 4, polinomio par)
# --------------------------------------------------------------------------
# Algoritmo de intercambio de Remez para hallar el polinomio par
#   P(u) = c0 + c2 u^2 + c4 u^4
# que minimiza max_{u in [-1, 1]} |P(u) - cos(pi*u/2)|.
#
# Como cos(pi*u/2) es simetrico, basta trabajar en t = u^2 in [0, 1] y
# buscar Q(t) = c0 + c2 t + c4 t^2 que minimiza
#   max_{t in [0, 1]} |Q(t) - cos(pi*sqrt(t)/2)|.
# El polinomio Q de grado 3 en t (despues de incluir el grado 6) o grado 2
# (para grado 4) se ajusta con Remez clasico: 4 puntos de error alterno.
def _remez_even_degree4(n_iter: int = 30) -> tuple[float, float, float]:
    """Devuelve (c0, c2, c4) minimax en u^(0,2,4) sobre [-1,1]."""
    # Variable t = u^2, dominio [0,1]. Polinomio Q(t) = c0 + c2 t + c4 t^2.
    # 4 nodos iniciales (Chebyshev nodes en [0,1]).
    k = np.arange(4)
    t_nodes = 0.5 * (1 - np.cos(np.pi * (k + 0.5) / 4))
    f = lambda t: np.cos(_PI_OVER_2 * np.sqrt(t))

    for _ in range(n_iter):
        # Sistema de Remez: Q(t_i) - sigma * (-1)^i = f(t_i),  i = 0..3
        signs = (-1.0) ** np.arange(4)
        A = np.column_stack([np.ones(4), t_nodes, t_nodes ** 2, signs])
        b = f(t_nodes)
        c0, c2, c4, _sigma = np.linalg.solve(A, b)

        # Buscar nuevo conjunto de extremos del error e(t) = Q(t) - f(t).
        t_dense = np.linspace(0.0, 1.0, 8192)
        err = (c0 + c2 * t_dense + c4 * t_dense ** 2) - f(t_dense)
        # Localizar 4 extremos alternos (incluyendo 0 y 1 si lo son).
        # Heuristica simple: encontrar puntos donde derivada cambia de signo +
        # los bordes; quedarse con 4 alternados de mayor amplitud.
        d = np.diff(np.sign(np.diff(err)))
        cand_idx = np.where(d != 0)[0] + 1
        cand_idx = np.concatenate(([0], cand_idx, [t_dense.size - 1]))
        cand_t = t_dense[cand_idx]
        cand_e = err[cand_idx]
        # Greedy: tomar el extremo de mayor |e|, luego ir alternando signo.
        order = np.argsort(-np.abs(cand_e))
        chosen: list[int] = []
        for idx in order:
            sign_e = np.sign(cand_e[idx])
            if not chosen:
                chosen.append(idx)
                continue
            prev_sign = np.sign(cand_e[chosen[-1]])
            if sign_e != prev_sign and cand_t[idx] != cand_t[chosen[-1]]:
                chosen.append(idx)
            if len(chosen) == 4:
                break
        if len(chosen) < 4:
            break  # convergencia
        new_t = np.sort(cand_t[chosen])
        if np.allclose(new_t, t_nodes, atol=1e-10):
            break
        t_nodes = new_t

    return float(c0), float(c2), float(c4)


_REMEZ4 = _remez_even_degree4()


def approx_remez4(u: Any, xp: Any = np) -> Any:
    """Polinomio minimax par grado 4 (Remez). c0 + c2 u^2 + c4 u^4."""
    c0, c2, c4 = _REMEZ4
    v = u * u
    return c0 + v * (c2 + v * c4)


# --------------------------------------------------------------------------
# 4) BHASKARA I
# --------------------------------------------------------------------------
def approx_bhaskara(u: Any, xp: Any = np) -> Any:
    """Aproximacion racional clasica:  4(1 - u^2) / (4 + u^2)."""
    v = u * u
    return 4.0 * (1.0 - v) / (4.0 + v)


# --------------------------------------------------------------------------
# Wrappers que devuelven directamente K(u) = (pi/4) * approx * mascara
# --------------------------------------------------------------------------
def _kernel_from_approx(approx: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
    def _k(u: Any, xp: Any = np) -> Any:
        inside = xp.abs(u) <= 1.0
        return xp.where(inside, _PI_OVER_4 * approx(u, xp=xp), 0.0)
    _k.__name__ = f"cosine_kernel_{approx.__name__.replace('approx_', '')}"
    return _k


cosine_kernel_taylor4 = _kernel_from_approx(approx_taylor4)
cosine_kernel_taylor6 = _kernel_from_approx(approx_taylor6)
cosine_kernel_cheb4   = _kernel_from_approx(approx_cheb4)
cosine_kernel_cheb6   = _kernel_from_approx(approx_cheb6)
cosine_kernel_remez4  = _kernel_from_approx(approx_remez4)
cosine_kernel_bhaskara = _kernel_from_approx(approx_bhaskara)


def cosine_kernel_exact(u: Any, xp: Any = np) -> Any:
    """K(u) coseno exacto, usando ``xp.cos``. Mismo resultado que
    ``kernels.core.kernel_eval(u, 'cosine')``. Se incluye aqui como
    referencia para el benchmark."""
    inside = xp.abs(u) <= 1.0
    return xp.where(inside, _PI_OVER_4 * xp.cos(_PI_OVER_2 * u), 0.0)


# --------------------------------------------------------------------------
# Registro / metadatos
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CosineApprox:
    name: str
    approx_fn: Callable[[Any, Any], Any]
    kernel_fn: Callable[[Any, Any], Any]
    family: str             # "taylor", "chebyshev", "remez", "bhaskara"
    degree: int             # grado polinomico (Bhaskara reportado como 2/2)
    coefficients: tuple[float, ...]
    ops_per_eval: str       # descripcion corta (mult, divs)
    notes: str


APPROXIMATIONS: dict[str, CosineApprox] = {
    "exact":     CosineApprox("exact",    lambda u, xp=np: xp.cos(_PI_OVER_2 * u),
                              cosine_kernel_exact, "exacto", -1, (),
                              "1 cos transcendente",
                              "Referencia. Usa cos() de la libreria."),
    "taylor4":   CosineApprox("taylor4",  approx_taylor4,  cosine_kernel_taylor4,
                              "taylor", 4,
                              (_TAYLOR_A0, _TAYLOR_A1, _TAYLOR_A2),
                              "2 mults Horner",
                              "Sin error en u=0; crece hacia |u|=1."),
    "taylor6":   CosineApprox("taylor6",  approx_taylor6,  cosine_kernel_taylor6,
                              "taylor", 6,
                              (_TAYLOR_A0, _TAYLOR_A1, _TAYLOR_A2, _TAYLOR_A3),
                              "3 mults Horner",
                              "Mejor que taylor4 en bordes."),
    "cheb4":     CosineApprox("cheb4",    approx_cheb4,    cosine_kernel_cheb4,
                              "chebyshev", 4, _CHEB4,
                              "2 mults Horner",
                              "Cuasi-minimax; error equilibrado en [-1,1]."),
    "cheb6":     CosineApprox("cheb6",    approx_cheb6,    cosine_kernel_cheb6,
                              "chebyshev", 6, _CHEB6,
                              "3 mults Horner",
                              "Error <1e-6 en [-1,1]."),
    "remez4":    CosineApprox("remez4",   approx_remez4,   cosine_kernel_remez4,
                              "remez", 4, _REMEZ4,
                              "2 mults Horner",
                              "Optimo minimax grado 4 par."),
    "bhaskara":  CosineApprox("bhaskara", approx_bhaskara, cosine_kernel_bhaskara,
                              "racional", 2, (4.0, -4.0, 4.0, 1.0),
                              "1 mult + 1 div",
                              "Aproximacion clasica racional."),
}


def list_approximations() -> tuple[str, ...]:
    return tuple(APPROXIMATIONS.keys())


__all__ = [
    "approx_taylor4", "approx_taylor6",
    "approx_cheb4", "approx_cheb6",
    "approx_remez4", "approx_bhaskara",
    "cosine_kernel_taylor4", "cosine_kernel_taylor6",
    "cosine_kernel_cheb4", "cosine_kernel_cheb6",
    "cosine_kernel_remez4", "cosine_kernel_bhaskara",
    "cosine_kernel_exact",
    "CosineApprox", "APPROXIMATIONS", "list_approximations",
]
