import sys
import os
from functools import partial
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture as lf

# see https://stackoverflow.com/questions/56307329/how-can-i-parametrize-tests-to-run-with-different-fixtures-in-pytest

thisfile = os.path.abspath(__file__)
modulepath = os.path.dirname(os.path.dirname(thisfile))

sys.path.insert(0, modulepath)
import tensorcircuit as tc

# TODO(@refraction-ray):replace all assert np.allclose with np.testing.assert_all_close !


def test_wavefunction():
    qc = tc.Circuit(2)
    qc.apply_double_gate(
        tc.gates.Gate(np.arange(16).reshape(2, 2, 2, 2).astype(np.complex64)), 0, 1
    )
    assert np.real(qc.wavefunction()[2]) == 8
    qc = tc.Circuit(2)
    qc.apply_double_gate(
        tc.gates.Gate(np.arange(16).reshape(2, 2, 2, 2).astype(np.complex64)), 1, 0
    )
    qc.wavefunction()
    assert np.real(qc.wavefunction()[2]) == 4
    qc = tc.Circuit(2)
    qc.apply_single_gate(
        tc.gates.Gate(np.arange(4).reshape(2, 2).astype(np.complex64)), 0
    )
    qc.wavefunction()
    assert np.real(qc.wavefunction()[2]) == 2


def test_basics():
    c = tc.Circuit(2)
    c.x(0)
    assert np.allclose(c.amplitude("10"), np.array(1.0))
    c.CNOT(0, 1)
    assert np.allclose(c.amplitude("11"), np.array(1.0))


def test_measure():
    c = tc.Circuit(3)
    c.H(0)
    c.h(1)
    c.toffoli(0, 1, 2)
    assert c.measure(2)[0] in ["0", "1"]


def test_gates_in_circuit():
    c = tc.Circuit(2, inputs=np.eye(2 ** 2))
    c.iswap(0, 1)
    ans = tc.gates.iswapgate().tensor.reshape([4, 4])
    np.testing.assert_allclose(c.state().reshape([4, 4]), ans, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_jittable_measure(backend):
    @partial(tc.backend.jit, static_argnums=(2, 3))
    def f(param, key, n=6, nlayers=3):
        if key is not None:
            tc.backend.set_random_state(key)
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for j in range(nlayers):
            for i in range(n - 1):
                c.exp1(i, i + 1, theta=param[2 * j, i], unitary=tc.gates._zz_matrix)
            for i in range(n):
                c.rx(i, theta=param[2 * j + 1, i])
        return c.measure_jit(0, 1, 2, with_prob=True)

    if tc.backend.name == "tensorflow":
        import tensorflow as tf

        print(f(tc.backend.ones([6, 6]), None))
        print(f(tc.backend.ones([6, 6]), None))
        print(f(tc.backend.ones([6, 6]), tf.random.Generator.from_seed(23)))
        print(f(tc.backend.ones([6, 6]), tf.random.Generator.from_seed(24)))
    elif tc.backend.name == "jax":
        import jax

        print(f(tc.backend.ones([6, 6]), jax.random.PRNGKey(23)))
        print(f(tc.backend.ones([6, 6]), jax.random.PRNGKey(24)))

    # As seen here, though I have tried the best, the random API is still not that consistent under jit


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_jittable_depolarizing(backend):
    @tc.backend.jit
    def f1(key):
        n = 5
        if key is not None:
            tc.backend.set_random_state(key)
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for i in range(n):
            c.cnot(i, (i + 1) % n)
        for i in range(n):
            c.unitary_kraus(
                [
                    tc.gates._x_matrix,
                    tc.gates._y_matrix,
                    tc.gates._z_matrix,
                    tc.gates._i_matrix,
                ],
                i,
                prob=[0.2, 0.2, 0.2, 0.4],
            )
        for i in range(n):
            c.cz(i, (i + 1) % n)
        return c.wavefunction()

    @tc.backend.jit
    def f2(key):
        n = 5
        if key is not None:
            tc.backend.set_random_state(key)
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for i in range(n):
            c.cnot(i, (i + 1) % n)
        for i in range(n):
            c.unitary_kraus(
                tc.channels.depolarizingchannel(0.2, 0.2, 0.2),
                i,
            )
        for i in range(n):
            c.X(i)
        return c.wavefunction()

    @tc.backend.jit
    def f3(key):
        n = 5
        if key is not None:
            tc.backend.set_random_state(key)
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for i in range(n):
            c.cnot(i, (i + 1) % n)
        for i in range(n):
            c.depolarizing(i, px=0.2, py=0.2, pz=0.2)
        for i in range(n):
            c.X(i)
        return c.wavefunction()

    @tc.backend.jit
    def f4(key):
        n = 5
        if key is not None:
            tc.backend.set_random_state(key)
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for i in range(n):
            c.cnot(i, (i + 1) % n)
        for i in range(n):
            c.depolarizing2(i, px=0.2, py=0.2, pz=0.2)
        for i in range(n):
            c.X(i)
        return c.wavefunction()

    @tc.backend.jit
    def f5(key):
        n = 5
        if key is not None:
            tc.backend.set_random_state(key)
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for i in range(n):
            c.cnot(i, (i + 1) % n)
        for i in range(n):
            c.unitary_kraus2(
                tc.channels.depolarizingchannel(0.2, 0.2, 0.2),
                i,
            )
        for i in range(n):
            c.X(i)
        return c.wavefunction()

    for f in [f1, f2, f3, f4, f5]:
        if tc.backend.name == "tensorflow":
            import tensorflow as tf

            assert np.allclose(tc.backend.norm(f(None)), 1.0, atol=1e-4)
            assert np.allclose(
                tc.backend.norm(f(tf.random.Generator.from_seed(23))), 1.0, atol=1e-4
            )
            assert np.allclose(
                tc.backend.norm(f(tf.random.Generator.from_seed(24))), 1.0, atol=1e-4
            )

        elif tc.backend.name == "jax":
            import jax

            assert np.allclose(
                tc.backend.norm(f(jax.random.PRNGKey(23))), 1.0, atol=1e-4
            )
            assert np.allclose(
                tc.backend.norm(f(jax.random.PRNGKey(24))), 1.0, atol=1e-4
            )


def test_expectation():
    c = tc.Circuit(2)
    c.H(0)
    assert np.allclose(c.expectation((tc.gates.z(), [0])), 0, atol=1e-7)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_exp1(backend):
    @partial(tc.backend.jit, jit_compile=True)
    def sf():
        c = tc.Circuit(2)
        xx = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.complex64
        )
        c.exp1(0, 1, unitary=xx, theta=tc.num_to_tensor(0.2))
        s = c.state()
        return s

    @tc.backend.jit
    def s1f():
        c = tc.Circuit(2)
        xx = np.array(
            [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.complex64
        )
        c.exp(0, 1, unitary=xx, theta=tc.num_to_tensor(0.2))
        s1 = c.state()
        return s1

    s = sf()
    s1 = s1f()
    assert np.allclose(s, s1, atol=1e-4)


def test_complex128(highp):
    tc.set_backend("tensorflow")
    tc.set_dtype("complex128")
    c = tc.Circuit(2)
    c.H(1)
    c.rx(0, theta=tc.gates.num_to_tensor(1j))
    c.wavefunction()
    assert np.allclose(c.expectation((tc.gates.z(), [1])), 0)


def test_qcode():
    qcode = """
4
x 0
cnot 0 1
r 2 theta 1.0 alpha 1.57
"""
    c = tc.Circuit.from_qcode(qcode)
    assert c.measure(1)[0] == "1"
    assert c.to_qcode() == qcode[1:]


def universal_ad():
    @tc.backend.jit
    def forward(theta):
        c = tc.Circuit(2)
        c.R(0, theta=theta, alpha=0.5, phi=0.8)
        return tc.backend.real(c.expectation((tc.gates.z(), [0])))

    gg = tc.backend.grad(forward)
    vag = tc.backend.value_and_grad(forward)
    gg = tc.backend.jit(gg)
    vag = tc.backend.jit(vag)
    theta = tc.gates.num_to_tensor(1.0)
    grad1 = gg(theta)
    v2, grad2 = vag(theta)
    assert grad1 == grad2
    return v2, grad2


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_ad(backend):
    # this amazingly shows how to code once and run in very different AD-ML engines
    print(universal_ad())


def test_single_qubit():
    c = tc.Circuit(1)
    c.H(0)
    w = c.state()[0]
    assert np.allclose(w, np.array([1, 1]) / np.sqrt(2), atol=1e-4)


def test_expectation_between_two_states():
    zp = np.array([1.0, 0.0])
    zd = np.array([0.0, 1.0])
    assert tc.expectation((tc.gates.y(), [0]), ket=zp, bra=zd) == 1j

    c = tc.Circuit(3)
    c.H(0)
    c.ry(1, theta=tc.num_to_tensor(0.8))
    c.cnot(1, 2)

    state = c.wavefunction()
    x1z2 = [(tc.gates.x(), [0]), (tc.gates.z(), [1])]
    e1 = c.expectation(*x1z2)
    e2 = tc.expectation(*x1z2, ket=state, bra=state, normalization=True)
    assert np.allclose(e2, e1)

    c = tc.Circuit(3)
    c.H(0)
    c.ry(1, theta=tc.num_to_tensor(0.8 + 0.7j))
    c.cnot(1, 2)

    state = c.wavefunction()
    x1z2 = [(tc.gates.x(), [0]), (tc.gates.z(), [1])]
    e1 = c.expectation(*x1z2) / tc.backend.norm(state) ** 2
    e2 = tc.expectation(*x1z2, ket=state, normalization=True)
    assert np.allclose(e2, e1)

    c = tc.Circuit(2)
    c.X(1)
    s1 = c.state()
    c2 = tc.Circuit(2)
    c2.X(0)
    s2 = c2.state()
    c3 = tc.Circuit(2)
    c3.H(1)
    s3 = c3.state()
    x1x2 = [(tc.gates.x(), [0]), (tc.gates.x(), [1])]
    e = tc.expectation(*x1x2, ket=s1, bra=s2)
    assert np.allclose(e, 1.0)
    e2 = tc.expectation(*x1x2, ket=s3, bra=s2)
    assert np.allclose(e2, 1.0 / np.sqrt(2))


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_any_inputs_state(backend):
    c = tc.Circuit(2, inputs=tc.array_to_tensor(np.array([0.0, 0.0, 0.0, 1.0])))
    c.X(0)
    z0 = c.expectation((tc.gates.z(), [0]))
    assert z0 == 1.0
    c = tc.Circuit(2, inputs=tc.array_to_tensor(np.array([0.0, 0.0, 1.0, 0.0])))
    c.X(0)
    z0 = c.expectation((tc.gates.z(), [0]))
    assert z0 == 1.0
    c = tc.Circuit(2, inputs=tc.array_to_tensor(np.array([1.0, 0.0, 0.0, 0.0])))
    c.X(0)
    z0 = c.expectation((tc.gates.z(), [0]))
    assert z0 == -1.0
    c = tc.Circuit(
        2,
        inputs=tc.array_to_tensor(np.array([1 / np.sqrt(2), 0.0, 1 / np.sqrt(2), 0.0])),
    )
    c.X(0)
    z0 = c.expectation((tc.gates.z(), [0]))
    assert np.allclose(z0, 0.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb")])
def test_postselection(backend):
    c = tc.Circuit(3)
    c.H(1)
    c.H(2)
    c.mid_measurement(1, 1)
    c.mid_measurement(2, 1)
    s = c.wavefunction()
    assert np.allclose(tc.backend.real(s[3]), 0.5)


def test_unitary():
    c = tc.Circuit(2, inputs=np.eye(4))
    c.X(0)
    c.Y(1)
    answer = np.kron(tc.gates.x().tensor, tc.gates.y().tensor)
    assert np.allclose(c.wavefunction().reshape([4, 4]), answer, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_dqas_type_circuit(backend):
    eye = tc.gates.i().tensor
    x = tc.gates.x().tensor
    y = tc.gates.y().tensor
    z = tc.gates.z().tensor

    def f(params, structures):
        paramsc = tc.backend.cast(params, dtype="complex64")
        structuresc = tc.backend.softmax(structures, axis=-1)
        structuresc = tc.backend.cast(structuresc, dtype="complex64")
        c = tc.Circuit(5)
        for i in range(5):
            c.H(i)
        for j in range(2):
            for i in range(4):
                c.cz(i, i + 1)
            for i in range(5):
                c.any(
                    i,
                    unitary=structuresc[i, j, 0]
                    * (
                        tc.backend.cos(paramsc[i, j, 0]) * eye
                        + tc.backend.sin(paramsc[i, j, 0]) * x
                    )
                    + structuresc[i, j, 1]
                    * (
                        tc.backend.cos(paramsc[i, j, 1]) * eye
                        + tc.backend.sin(paramsc[i, j, 1]) * y
                    )
                    + structuresc[i, j, 2]
                    * (
                        tc.backend.cos(paramsc[i, j, 2]) * eye
                        + tc.backend.sin(paramsc[i, j, 2]) * z
                    ),
                )
        return tc.backend.real(c.expectation([tc.gates.z(), (2,)]))

    structures = tc.array_to_tensor(
        np.random.normal(size=[16, 5, 2, 3]), dtype="float32"
    )
    params = tc.array_to_tensor(np.random.normal(size=[5, 2, 3]), dtype="float32")

    vf = tc.backend.vmap(f, vectorized_argnums=(1,))

    assert np.allclose(vf(params, structures).shape, [16])

    vvag = tc.backend.vvag(f, argnums=0, vectorized_argnums=1)

    vvag = tc.backend.jit(vvag)

    value, grad = vvag(params, structures)

    assert np.allclose(value.shape, [16])
    assert np.allclose(grad.shape, [5, 2, 3])


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_mixed_measurement_circuit(backend):
    n = 4

    def f(params, structures):
        structuresc = tc.backend.softmax(structures, axis=-1)
        structuresc = tc.backend.cast(structuresc, dtype="complex64")
        c = tc.Circuit(n)
        for i in range(n):
            c.H(i)
        for j in range(2):
            for i in range(n):
                c.cnot(i, (i + 1) % n)
            for i in range(n):
                c.rz(i, theta=params[j, i])
        obs = []
        for i in range(n):
            obs.append(
                [
                    tc.gates.Gate(
                        sum(
                            [
                                structuresc[i, k] * g.tensor
                                for k, g in enumerate(tc.gates.pauli_gates)
                            ]
                        )
                    ),
                    (i,),
                ]
            )
        loss = c.expectation(*obs, reuse=False)
        return tc.backend.real(loss)

    # measure X0 to X3

    structures = tc.backend.cast(tc.backend.eye(n), "int32")
    structures = tc.backend.onehot(structures, num=4)

    f_vvag = tc.backend.jit(tc.backend.vvag(f, vectorized_argnums=1, argnums=0))
    v, g = f_vvag(tc.backend.ones([2, n], dtype="float32"), structures)
    np.testing.assert_allclose(
        v,
        np.array(
            [
                0.015747,
                0.026107,
                0.019598,
                0.025447,
            ]
        ),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        g[0],
        np.array([-0.038279, 0.065810, -0.0018669, 0.035806]),
        atol=1e-5,
    )


def test_circuit_add_demo():
    # to be refactored for better API
    c = tc.Circuit(2)
    c.x(0)
    c2 = tc.Circuit(2, mps_inputs=c.quvector())
    c2.X(0)
    answer = np.array([1.0, 0, 0, 0])
    assert np.allclose(c2.wavefunction(), answer, atol=1e-4)
    c3 = tc.Circuit(2)
    c3.X(0)
    c3.replace_mps_inputs(c.quvector())
    assert np.allclose(c3.wavefunction(), answer, atol=1e-4)


@pytest.mark.parametrize("backend", [lf("tfb"), lf("jaxb")])
def test_circuit_split(backend):
    n = 4

    def f(param, max_singular_values=None, max_truncation_err=None, fixed_choice=None):
        if (max_singular_values is None) and (max_truncation_err is None):
            split = None
        else:
            split = {
                "max_singular_values": max_singular_values,
                "max_truncation_err": max_truncation_err,
                "fixed_choice": fixed_choice,
            }
        c = tc.Circuit(
            n,
            split=split,
        )
        for i in range(n):
            c.H(i)
        for j in range(2):
            for i in range(n - 1):
                c.exp1(i, i + 1, theta=param[2 * j, i], unitary=tc.gates._zz_matrix)
            for i in range(n):
                c.rx(i, theta=param[2 * j + 1, i])
        loss = c.expectation(
            (
                tc.gates.z(),
                [1],
            ),
            (
                tc.gates.z(),
                [2],
            ),
        )
        return tc.backend.real(loss)

    s1 = f(tc.backend.ones([4, n]))
    s2 = f(tc.backend.ones([4, n]), max_truncation_err=1e-5)
    s3 = f(tc.backend.ones([4, n]), max_singular_values=2, fixed_choice=1)

    np.testing.assert_allclose(s1, s2, atol=1e-5)
    np.testing.assert_allclose(s1, s3, atol=1e-5)

    f_jit = tc.backend.jit(f, static_argnums=(1, 2, 3))

    s1 = f_jit(tc.backend.ones([4, n]))
    # s2 = f_jit(tc.backend.ones([4, n]), max_truncation_err=1e-5) # doesn't work now
    # this cannot be done anyway, since variable size tensor network will fail opt einsum
    s3 = f_jit(tc.backend.ones([4, n]), max_singular_values=2, fixed_choice=1)

    # np.testing.assert_allclose(s1, s2, atol=1e-5)
    np.testing.assert_allclose(s1, s3, atol=1e-5)

    f_vag = tc.backend.jit(
        tc.backend.value_and_grad(f, argnums=0), static_argnums=(1, 2, 3)
    )

    s1, g1 = f_vag(tc.backend.ones([4, n]))
    s3, g3 = f_vag(tc.backend.ones([4, n]), max_singular_values=2, fixed_choice=1)

    np.testing.assert_allclose(s1, s3, atol=1e-5)
    print(g1[:, :])
    print(g3[:, :])
    # DONE(@refraction-ray): nan on jax backend?
    # i see, complex value SVD is not supported on jax for now :)
    # I shall further customize complex SVD, finally it has applications

    # tf 2.6.2 also doesn't support complex valued SVD AD, weird...
    # if tc.backend.name == "tensorflow":
    np.testing.assert_allclose(g1, g3, atol=1e-5)


@pytest.mark.parametrize("backend", [lf("npb"), lf("tfb"), lf("jaxb")])
def test_gate_split(backend):
    n = 4

    def f(param, max_singular_values=None, max_truncation_err=None, fixed_choice=None):
        if (max_singular_values is None) and (max_truncation_err is None):
            split = None
        else:
            split = {
                "max_singular_values": max_singular_values,
                "max_truncation_err": max_truncation_err,
                "fixed_choice": fixed_choice,
            }
        c = tc.Circuit(
            n,
        )
        for i in range(n):
            c.H(i)
        for j in range(2):
            for i in range(n - 1):
                c.exp1(
                    i,
                    i + 1,
                    theta=param[2 * j, i],
                    unitary=tc.gates._zz_matrix,
                    split=split,
                )
            for i in range(n):
                c.rx(i, theta=param[2 * j + 1, i])
        loss = c.expectation(
            (
                tc.gates.z(),
                [1],
            ),
            (
                tc.gates.z(),
                [2],
            ),
        )
        return tc.backend.real(loss)

    s1 = f(tc.backend.ones([4, n]))
    s2 = f(tc.backend.ones([4, n]), max_truncation_err=1e-5)
    s3 = f(tc.backend.ones([4, n]), max_singular_values=2, fixed_choice=1)

    np.testing.assert_allclose(s1, s2, atol=1e-5)
    np.testing.assert_allclose(s1, s3, atol=1e-5)
