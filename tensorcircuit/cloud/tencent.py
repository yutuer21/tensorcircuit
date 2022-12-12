"""
Cloud provider from Tencent
"""

from typing import Any, Dict, List, Optional, Sequence, Union
from json import dumps
import logging
from functools import partial
import re

from .config import tencent_base_url
from .utils import rpost_json
from .abstraction import Device, sep, Task
from ..abstractcircuit import AbstractCircuit
from ..utils import is_sequence, arg_alias

logger = logging.getLogger(__name__)


def tencent_headers(token: Optional[str] = None) -> Dict[str, str]:
    if token is None:
        token = "ANY;0"
    headers = {"Authorization": "Bearer " + token}
    return headers


def error_handling(r: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(r, dict):
        raise ValueError("failed to get legal response from the server")
    if "err" in r:
        raise ValueError(r["err"])
    return r


def list_devices(token: Optional[str] = None) -> List[Device]:
    json: Dict[Any, Any] = {}
    r = rpost_json(
        tencent_base_url + "device/find", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    ds = r["devices"]
    rs = []
    for d in ds:
        rs.append(Device.from_name("tencent" + sep + d["id"]))
    return rs


def list_properties(device: Device, token: Optional[str] = None) -> Dict[str, Any]:
    json = {"id": device.name}
    r = rpost_json(
        tencent_base_url + "device/detail", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    if "device" in r:
        return r["device"]  # type: ignore
    else:
        raise ValueError("No device with the name: %s" % device)


def _free_pi(s: str) -> str:
    # dirty trick to get rid of pi in openqasm from qiskit
    rs = []
    pistr = "3.141592653589793"
    s = s.replace("pi", pistr)
    for r in s.split("\n"):
        inc = re.search(r"\(.*\)", r)
        if inc is None:
            rs.append(r)
        else:
            v = r[inc.start() : inc.end()]
            v = eval(v)
            r = r[: inc.start()] + "(" + str(v) + ")" + r[inc.end() :]
            rs.append(r)
    return "\n".join(rs)


@partial(arg_alias, alias_dict={"compiling": ["compiled"]})
def submit_task(
    device: Device,
    token: str,
    lang: str = "OPENQASM",
    shots: Union[int, Sequence[int]] = 1024,
    version: str = "1",
    prior: int = 1,
    circuit: Optional[Union[AbstractCircuit, Sequence[AbstractCircuit]]] = None,
    source: Optional[Union[str, Sequence[str]]] = None,
    remarks: Optional[str] = None,
    compiling: bool = False,
    compiled_options: Optional[Dict[str, Any]] = None,
    measure: Optional[Sequence[int]] = None,
) -> List[Task]:
    if source is None:
        if compiled_options is None:
            compiled_options = {
                "basis_gates": ["h", "rz", "x", "y", "z", "cx"],
                "optimization_level": 2,
            }

        def c2qasm(c: Any, compiling: bool) -> str:
            from qiskit.compiler import transpile
            from qiskit.circuit import QuantumCircuit

            if compiling is True:
                if not isinstance(c, QuantumCircuit):
                    c = c.to_qiskit()

                nq = c.num_qubits
                c1 = transpile(c, **compiled_options)
                s = c1.qasm()
            else:
                if isinstance(c, QuantumCircuit):
                    s = c.qasm()
                    nq = c.num_qubits
                else:
                    s = c.to_openqasm()
                    nq = c._nqubits
            # s = _free_pi(s) # tQuk translation now supports this
            if measure is not None:  # ad hoc partial measurement
                slist = s.split("\n")[:-1]
                if len(slist) > 3 and not slist[3].startswith("creg"):
                    slist.insert(3, "creg c[%s];" % nq)
                for m in measure:
                    slist.append("measure q[%s]->c[%s];" % (m, m))
                slist.append("")
                s = "\n".join(slist)
            return s  # type: ignore

        if is_sequence(circuit):
            source = [c2qasm(c, compiling) for c in circuit]  # type: ignore
        else:
            source = c2qasm(circuit, compiling)
        lang = "OPENQASM"
    if is_sequence(source):
        # batched mode
        json = []
        if not is_sequence(shots):
            shots = [shots for _ in source]  # type: ignore
        for sc, sh in zip(source, shots):  # type: ignore
            json.append(
                {
                    "device": device.name,
                    "shots": sh,
                    "source": sc,
                    "version": version,
                    "lang": lang,
                    "prior": prior,
                    "remarks": remarks,
                }
            )

    else:
        json = {  # type: ignore
            "device": device.name,
            "shots": shots,
            "source": source,
            "version": version,
            "lang": lang,
            "prior": prior,
            "remarks": remarks,
        }
    r = rpost_json(
        tencent_base_url + "task/submit", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    try:
        rtn = []
        for t in r["tasks"]:
            if "err" in t:
                logger.warning(t["err"])
            else:
                rtn.append(Task(id_=t["id"], device=device))
        if len(rtn) == 1:
            return rtn[0]  # type: ignore
        else:
            return rtn
    except KeyError:
        raise ValueError(dumps(r))


def resubmit_task(task: Task, token: str) -> Task:
    # TODO(@refraction-ray): batch resubmit
    json = {"id": task.id_}
    r = rpost_json(
        tencent_base_url + "task/start", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    try:
        return Task(id_=r["tasks"][0]["id"])

    except KeyError:
        raise ValueError(dumps(r))


def remove_task(task: Task, token: str) -> Any:
    # TODO(@refraction-ray): batch cancel
    json = {"id": task.id_}
    r = rpost_json(
        tencent_base_url + "task/remove", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    return r


def list_tasks(device: Device, token: str, **filter_kws: Any) -> List[Task]:
    json = filter_kws
    if device is not None:
        json["device"] = device.name
    r = rpost_json(
        tencent_base_url + "task/find?pn=1&npp=50",
        json=json,
        headers=tencent_headers(token),
    )
    r = error_handling(r)
    try:
        rtn = []
        for t in r["tasks"]:
            rtn.append(
                Task(
                    id_=t["id"],
                    device=Device.from_name("tencent" + sep + t["device"]),
                )
            )
        return rtn
    except KeyError:
        raise ValueError(dumps(r))


def get_task_details(task: Task, device: Device, token: str) -> Dict[str, Any]:
    json = {"id": task.id_}
    r = rpost_json(
        tencent_base_url + "task/detail", json=json, headers=tencent_headers(token)
    )
    r = error_handling(r)
    try:
        if "result" in r["task"]:
            if "counts" in r["task"]["result"]:
                r["task"]["results"] = r["task"]["result"]["counts"]
            else:
                r["task"]["results"] = r["task"]["result"]
        return r["task"]  # type: ignore
    except KeyError:
        raise ValueError(dumps(r))

    # make the return at least contain the following terms across different providers
    """
    'id': 'd947cd76-a961-4c22-b295-76287c9fdaa3',
    'state': 'completed',
    'at': 1666752095915849,
    'shots': 1024,
    'source': 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\nh q[0];\nh q[1];\nh q[2];\n', 
    'device': 'simulator:aer',
    'results':  {'000': 123,
    '001': 126,
    '010': 131,
    '011': 128,
    '100': 122,
    '101': 135,
    '110': 128,
    '111': 131}
    """
