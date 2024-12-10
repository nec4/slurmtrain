import pytest
import numpy as np
from slurmtrain.slurmtrain import *


ok_objs = [
    "bgn1008",
    "bgn1008,bgn1009",
    "bgn1002-bgn1005",
    "bgn1001,bgn1002-bgn1003",
    "bgn1001-bgn1003,bgn1004",
]
ok_outputs = [
    ["bgn1008"],
    ["bgn1008", "bgn1009"],
    ["bgn1002", "bgn1003", "bgn1004", "bgn1005"],
    ["bgn1001", "bgn1002", "bgn1003"],
    ["bgn1001", "bgn1002", "bgn1003", "bgn1004"],
]

wrong_objs = [
    "1001bgn1,bgn1922",
    "object5476,silly",
    "bgn1001,bggggn1002",
    "bgn1001-bfn1010",
]

dpns = [1, 2, 3]

dirs = [f"dir{i}" for i in range(8)]
nodes = [f"node{str(i).zfill(3)}" for i in range(8)]
expected_assignments = [
    [(n, [d]) for n, d in zip(nodes, dirs)],
    [
        ("node000", ["dir0", "dir1"]),
        ("node001", ["dir2", "dir3"]),
        ("node002", ["dir4", "dir5"]),
        ("node003", ["dir6", "dir7"]),
    ],
    [
        ("node000", ["dir0", "dir1", "dir2"]),
        ("node001", ["dir3", "dir4", "dir5"]),
        ("node002", ["dir6", "dir7"]),
    ],
]


@pytest.mark.parametrize(
    "obj, out",
    [
        (ok_objs[0], ok_outputs[0]),
        (ok_objs[1], ok_outputs[1]),
        (ok_objs[2], ok_outputs[2]),
        (ok_objs[3], ok_outputs[3]),
        (ok_objs[4], ok_outputs[4]),
    ],
)
def test_parse_objlist(obj, out):
    tokenized = parse_objlist(obj)
    assert all([o == t for o, t in zip(out, tokenized)])


@pytest.mark.parametrize(
    "obj, expected_error",
    [
        (wrong_objs[0], ValueError),
        (wrong_objs[1], ValueError),
        (wrong_objs[2], ValueError),
        (wrong_objs[3], ValueError),
    ],
)
def test_parse_objlist_raises(obj, expected_error):
    with pytest.raises(expected_error):
        parse_objlist(obj)


@pytest.mark.parametrize(
    "nodes, dirs, dirs_per_node, expected_assignments",
    [
        (nodes, dirs, dpns[0], expected_assignments[0]),
        (nodes, dirs, dpns[1], expected_assignments[1]),
        (nodes, dirs, dpns[2], expected_assignments[2]),
    ],
)
def test_assignments(nodes, dirs, dirs_per_node, expected_assignments):
    assignments = partition_dirs(nodes, dirs, dirs_per_node)
    assert len(assignments) == len(expected_assignments)
    for a, ea in zip(assignments, expected_assignments):
        assert a == ea


def test_partition_raises():
    with pytest.raises(RuntimeError):
        partition_dirs(nodes[:2], dirs, 1)
