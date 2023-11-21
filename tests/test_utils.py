import numpy
import pytest

from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps, get_random_mps
from mps_fqe.utils import (one_body_projection_mpo, two_body_projection_mpo,
                           three_body_projection_mpo, apply_fiedler_ordering)


@pytest.mark.parametrize("nele, sz, norb", [
    (2, 0, 2), (2, 0, 4),
    (1, 1, 2), (1, -1, 2),
    (3, 1, 3), (3, -1, 3)
])
def test_one_body_projection_mpo(nele, sz, norb):
    """Test that the spin-free and spin orbital 1-body projection operators are
    consistent.
    """
    iorb = 0
    jorb = 1
    numpy.random.seed(11)
    mps = get_random_mps(nele, sz, norb, bdim=50)
    p1 = one_body_projection_mpo(iorb, jorb, norb, spinfree=True)
    p1a = one_body_projection_mpo(iorb*2, jorb*2, norb, spinfree=False)
    p1b = one_body_projection_mpo(iorb*2 + 1, jorb*2 + 1, norb, spinfree=False)
    exp_sf = mps.expectationValue(p1)
    exp_ss = mps.expectationValue(p1a + p1b)
    assert abs(exp_sf - exp_ss) < 1e-12


@pytest.mark.parametrize("nele, sz, norb", [
    (2, 0, 2), (2, 0, 4),
    (1, 1, 2), (1, -1, 2),
    (3, 1, 3), (3, -1, 3)
])
def test_two_body_projection_mpo(nele, sz, norb):
    """Test that the spin-free and spin orbital 2-body projection operators are
    consistent.
    """
    iorb = 0
    jorb = 1
    korb = 0
    lorb = 1
    numpy.random.seed(22)
    mps = get_random_mps(nele, sz, norb, bdim=50)
    p1 = two_body_projection_mpo(iorb, jorb, korb, lorb, norb, spinfree=True)
    p1aa = two_body_projection_mpo(iorb*2, jorb*2, korb*2, lorb*2, norb, spinfree=False)
    p1bb = two_body_projection_mpo(
        iorb*2 + 1, jorb*2 + 1, korb*2 + 1, lorb*2 + 1, norb, spinfree=False)
    p1ab = two_body_projection_mpo(
        iorb*2, jorb*2 + 1, korb*2, lorb*2 + 1, norb, spinfree=False)
    p1ba = two_body_projection_mpo(
        iorb*2 + 1, jorb*2, korb*2 + 1, lorb*2, norb, spinfree=False)
    exp_sf = mps.expectationValue(p1)
    exp_ss = mps.expectationValue(p1aa + p1bb + p1ab + p1ba)
    assert abs(exp_sf - exp_ss) < 1e-12


@pytest.mark.parametrize("nele, sz, norb", [
    (2, 0, 2), (2, 0, 4),
    (1, 1, 2), (1, -1, 2),
    (3, 1, 3), (3, -1, 3)
])
def test_three_body_projection_mpo(nele, sz, norb):
    """Test that the spin-free and spin orbital 3-body projection operators are
    consistent.
    """
    iorb = 0
    jorb = 1
    korb = 1
    lorb = 1
    morb = 0
    zorb = 0
    numpy.random.seed(22)
    mps = get_random_mps(nele, sz, norb, bdim=50)
    p1 = three_body_projection_mpo(iorb, jorb, korb, lorb, morb, zorb, norb, spinfree=True)
    p1aaa = three_body_projection_mpo(
        iorb*2, jorb*2, korb*2, lorb*2, morb*2, zorb*2, norb, spinfree=False)
    p1bbb = three_body_projection_mpo(
        iorb*2 + 1, jorb*2 + 1, korb*2 + 1, lorb*2 + 1, morb*2 + 1, zorb*2 + 1,
        norb, spinfree=False)
    p1aab = three_body_projection_mpo(
        iorb*2, jorb*2, korb*2 + 1, lorb*2, morb*2, zorb*2 + 1,
        norb, spinfree=False)
    p1aba = three_body_projection_mpo(
        iorb*2, jorb*2 + 1, korb*2, lorb*2, morb*2 + 1, zorb*2,
        norb, spinfree=False)
    p1baa = three_body_projection_mpo(
        iorb*2 + 1, jorb*2, korb*2, lorb*2 + 1, morb*2, zorb*2,
        norb, spinfree=False)
    p1abb = three_body_projection_mpo(
        iorb*2, jorb*2 + 1, korb*2 + 1, lorb*2, morb*2 + 1, zorb*2 + 1,
        norb, spinfree=False)
    p1bab = three_body_projection_mpo(
        iorb*2 + 1, jorb*2, korb*2 + 1, lorb*2 + 1, morb*2, zorb*2 + 1,
        norb, spinfree=False)
    p1bba = three_body_projection_mpo(
        iorb*2 + 1, jorb*2 + 1, korb*2, lorb*2 + 1, morb*2 + 1, zorb*2,
        norb, spinfree=False)
    exp_sf = mps.expectationValue(p1)
    exp_ss = mps.expectationValue(
        p1aaa + p1aab + p1aba + p1baa + p1abb + p1bab + p1bba + p1bbb)
    assert abs(exp_sf - exp_ss) < 1e-12


def test_fiedler_order():
    norb = 4
    order = [0, 1, 2, 3]
    h2 = numpy.zeros((norb, norb, norb, norb))
    h1 = numpy.zeros((norb, norb))
    bonds = set()
    for i in order[:-1]:
        j = order[i + 1]
        bonds.add((i, j))
        h2[i, i, i, i] = 1
        h2[j, j, j, j] = 1
        h2[i, j, j, i] = -1 
        h2[j, i, i, j] = -1 
    
    _, _, out = apply_fiedler_ordering(h1, h2)
    obonds = set()
    for ii, i in enumerate(out[:-1]):
        j = out[ii + 1]
        i, j = (i, j) if i < j else (j, i)
        obonds.add((i, j))

    # only allow one wrong 'bond'
    assert len(bonds - obonds) <= 1
