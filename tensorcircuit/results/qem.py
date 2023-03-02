from mitiq import zne, ddd
from mitiq.zne.inference import Factory
from mitiq.zne.scaling import fold_gates_at_random
from typing import Any, Sequence, Union, Optional, Dict, Tuple, Callable
import numpy as np
from random import choice
from itertools import product
import cirq

from .. import Circuit
from .. import backend,gates


zne_option = zne
def apply_zne(
    circuit: Any,
    executor: Callable[[Any], float],
    factory: Optional[Factory],
    scale_noise: Callable[[Any, float], Any] = fold_gates_at_random,
    num_to_average: int = 1,
    **kws: Any,
) -> float:
    """Apply zero-noise extrapolation (ZNE) and return the mitigated results.

    :param circuit: The aim circuit.
    :type circuit: Any
    :param executor: A executor that executes a circuit and return results.
    :type executor: Callable[[Any], float]
    :param factory: Determines the extropolation method.
    :type factory: Optional[Factory]
    :param scale_noise: The scaling function for the aim circuit, defaults to fold_gates_at_random
    :type scale_noise: Callable[[Any, float], Any], optional
    :param num_to_average: Number of times expectation values are computed by
            the executor, average each point, defaults to 1.
    :type num_to_average: int, optional
    :return: Mitigated average value by ZNE.
    :rtype: float
    """

    def executortc(c):
        c = Circuit.from_qiskit(c, c.num_qubits)
        return executor(c)


    circuit = circuit.to_qiskit()
    result = zne.execute_with_zne(
        circuit=circuit,
        executor=executortc,
        factory=factory,
        scale_noise=scale_noise,
        num_to_average=num_to_average,
        **kws
    )
    return result



def washcircuit(c,qlist):
    qir = c.to_qir()
    cnew = Circuit(c.circuit_param["nqubits"])
    for d in qir:
        if d["index"][0] in qlist:
            if_iden = np.sum(abs(np.array([[1,0],[0,1]])-d["gate"].get_tensor()))
            if if_iden > 1e-4:
                if "parameters" not in d: 
                    cnew.apply_general_gate_delayed(d["gatef"], d["name"])(cnew, *d["index"])
                else:
                    cnew.apply_general_variable_gate_delayed(d["gatef"], d["name"])(
                        cnew, *d["index"], **d["parameters"]
                    )
    return cnew

def use_qubits(c):
    qir = c.to_qir()
    qlist = []
    for d in qir:
        for i in range(len(d["index"])):
            if d["index"][i] not in qlist:
                qlist.append(d["index"][i])
    return qlist

def add_dd(c,rule):
    nqubit = c.circuit_param["nqubits"]
    input_circuit = c.to_qiskit()
    circuit_dd = dd_option.insert_ddd_sequences(input_circuit, rule=rule)
    circuit_dd = Circuit.from_qiskit(circuit_dd, nqubit)
    return circuit_dd

dd_option = ddd
def apply_dd(
    circuit: Any,
    executor: Callable[[Any], float],
    rule: Union[Callable[[int], Any], list],
    rule_args: Dict[str, Any] = {},
    num_trials: int = 1,
    full_output: bool = False,
    ignore_idle_qubit: bool =True,
    fulldd: bool  = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Apply dynamic decoupling (DD) and return the mitigated results.


    :param circuit: The aim circuit.
    :type circuit: Any
    :param executor: A executor that executes a circuit and return results.
    :type executor: Callable[[Any], float]
    :param rule: The rule to construct DD sequence, can use default rule "dd_option.rules.xx"
    or custom rule "['X','X']"
    :type rule: Union[Callable[[int], Any], list]
    :param rule_args:An optional dictionary of keyword arguments for ``rule``, defaults to {}
    :type rule_args: Dict[str, Any], optional
    :param num_trials: The number of independent experiments to average over, defaults to 1
    :type num_trials: int, optional
    :param full_output: If ``False`` only the mitigated expectation value is
            returned. If ``True`` a dictionary containing all DD data is
            returned too, defaults to False
    :type full_output: bool, optional
    :param ig_idle_qubit: ignore the DD sequences that added to unused qubits, defaults to True
    :type ig_idle_qubit: bool, optional
    :return: mitigated expectation value or mitigated expectation value and DD circuit information
    :rtype: Union[float, Tuple[float, Dict[str, Any]]]
    """

    if ignore_idle_qubit is True:
        qlist = use_qubits(circuit)
    else:
        qlist = list(range(circuit.circuit_param["nqubits"]))

    def executortc(c):
        c = Circuit.from_qiskit(c, c.num_qubits)
        c = washcircuit(c,qlist)
        return executor(c)


    def dd_rule(slack_length, spacing = -1):
        dd_sequence = dd_option.rules.general_rule(
        slack_length=slack_length,
        spacing=spacing,
        gates=gates,
        )
        return dd_sequence
    
    if isinstance(rule, list):
        gates =[]
        for i in rule:
            gates.append(getattr(cirq,i))
        rule = dd_rule

    if fulldd is True:
        c2=circuit
        c3=circuit
        while c2.to_openqasm() != c3.to_openqasm():
            c2 = c3
            c3 = add_dd(c2,rule)
            c3 = washcircuit(c3,qlist = use_qubits(circuit))
        if full_output is True:
            result = [executor(c3),c3]
        else:
            result = executor(c3)

    else:
        circuit = circuit.to_qiskit()
        result = ddd.execute_with_ddd(
            circuit=circuit,
            executor=executortc,
            rule=rule,
            rule_args=rule_args,
            num_trials=num_trials,
            full_output=full_output,
        )
        if full_output is True:
            cdd = result[1]['circuits_with_ddd'][0]
            result = [result[0],Circuit.from_qiskit(cdd, cdd.num_qubits)]
        else:
            result = result
    return result




def rc_candidates(gate):
    pauli = [m.tensor for m in gates.pauli_gates]
    if isinstance(gate, gates.Gate):
        gate = gate.tensor
    gatem = backend.reshapem(gate)
    r = []
    for combo in product(*[range(4) for _ in range(4)]):
        i = np.kron(pauli[combo[0]], pauli[combo[1]])@gatem@np.kron(pauli[combo[2]], pauli[combo[3]])
        if np.allclose(i, gatem, atol=1e-4):
            r.append(combo)
        elif np.allclose(i, -gatem, atol=1e-4):
            r.append(combo)
    return r


def apply_gate(c,i,j):
    if i ==0:
        c.i(j)
    elif i ==1:
        c.x(j)
    elif i ==2:
        c.y(j)
    elif i ==3:
        c.z(j)
    return c

def rc_circuit(c):
    qir = c.to_qir()
    cnew = Circuit(c.circuit_param["nqubits"])
    for d in qir:
        if len(d["index"]) == 2:
            rc_list = choice(rc_candidates(d["gate"]))

            cnew = apply_gate(cnew,rc_list[0],d["index"][0])
            cnew = apply_gate(cnew,rc_list[1],d["index"][1])
            if "parameters" not in d: 
                cnew.apply_general_gate_delayed(d["gatef"], d["name"])(cnew, *d["index"])
            else:
                cnew.apply_general_variable_gate_delayed(d["gatef"], d["name"])(
                    cnew, *d["index"], **d["parameters"]
                )
            cnew = apply_gate(cnew,rc_list[2],d["index"][0])
            cnew = apply_gate(cnew,rc_list[3],d["index"][1])
        else: 
            if "parameters" not in d: 
                cnew.apply_general_gate_delayed(d["gatef"], d["name"])(cnew, *d["index"])
            else:
                cnew.apply_general_variable_gate_delayed(d["gatef"], d["name"])(
                    cnew, *d["index"], **d["parameters"]
                )
    return cnew


def apply_rc(    
    circuit: Any,
    executor: Callable[[Any], float],
    num_to_average: int = 1,
    simplify = False,
    **kws: Any,
) -> float:
    """_summary_

    :param circuit: _description_
    :type circuit: Any
    :param executor: _description_
    :type executor: Callable[[Any], float]
    :param num_to_average: _description_, defaults to 1
    :type num_to_average: int, optional
    :param simplify: _description_, defaults to False
    :type simplify: bool, optional
    :return: _description_
    :rtype: float
    """
    exp = []
    for _ in range(num_to_average):
        circuit = rc_circuit(circuit)
        exp.append(executor(circuit))
    result = np.mean(exp)

    return result
    

 


    
    
    