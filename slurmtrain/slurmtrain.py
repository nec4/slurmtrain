import subprocess
import pathlib
import time
import argparse
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
import math
import multiprocessing
import os
import re


def parse_input():
    parser = argparse.ArgumentParser(
        description="""

        Nuclear option/tool for submitting SLURM dependency trains from a directories to a specified subset of nodes when `singleton`
        dependency option is not sufficient.

        The tool assumes that compute node and directory names are specified as '(some alpha characters)(some numbers)',
        e.g., like 'bgn1003'. Directories containing the job scripts treat all '.sh' extended files as SLURM job scripts.

        """
    )

    parser.add_argument(
        "--nodelist",
        type=str,
        default=None,
        help="node range or list of the form `x1, x1`, `x1-x10` or a mixture, i,e, `x1,x2-x10`. Assumes that node names are of the form `^[a-zA-Z]` + `[0-9]$` (some alphas and then a numeric ending). All nodes must share the same basename.",
    )
    parser.add_argument(
        "--dependency-type",
        type=str,
        help="SLURM dependency type. Must be one of `after`, `afterany`, `afterok`, `afternotok`. Defaults to `afterany`.",
        default="afterany",
    )
    parser.add_argument(
        "--dirlist",
        type=str,
        help="Directories of jobscripts from which each node in `nodelist` will build dependency trains from, in the same order as the nodes given by `nodelist`. The number of directories must be equal to or less than `(len(nodelist) * dirs_per_node)`. Directories are specified in the same format as in `--nodelist`",
    )
    parser.add_argument(
        "--wait",
        type=float,
        help="Number of seconds to wait between each successive SLURM job submission",
        default=3,
    )
    parser.add_argument(
        "--additional-options",
        type=str,
        help="additional options of the form `flag1,opt1,flag2,opt2,...` for SLURM",
        default=None,
    )

    parser.add_arguments(
        "--reservation",
        type=str,
        help="If no --nodelist is specified, then a reservation must be specified for greedy job submission (Yes, even if that reservation is already specified in the jobscript)",
        default=None,
    )

    parser.add_argument(
        "--stop-on-stderr",
        help="If set, the train submitter program will exit on ANY nonzero `stderr` from a `subprocess` call",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--verbose",
        help="Prints submission debugging info to stdout",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--jobs-per-node",
        help="Maximum number of jobs to send to a node when --nodelist is not used",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-num-jobs",
        help="Maximum number of jobs that can exist in `squeue`",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--debug",
        help="If specified, the list of final nodes will be printed",
        action="store_true",
        default=1,
    )
    return parser


def parse_objlist(
    objlist: str, assert_base_name_same=True, assert_fixed_len=True
) -> List[str]:
    """
    Turns a single string objlist option into a full expanded list of tokenized objs. E.g.)

    `"bgn001,bgn002-bgn04"` returns `["bgn001", "bgn002", "bgn003", "bgn004"]`

    Parameters
    ----------
    objlist:
        input `str` that must be tokenized
    assert_base_name_same:
        if `True`, an assertion check is made to ensure that the alpha parts of
        the objs all match. Eg, bgn001, bgn002, bgn003, ... all share 'bgn'
    assert_fixed_len:
        if `True`, an assertion check is made to ensure that the length of all
        tokens are the same.

    Returns
    -------
    final_objs:
        `List[str]` of tokenized objects
    """

    # Tokenize
    # First, strip brackers and break apart at commas
    final_objs = []
    objs = objlist.split(",")

    if not all([o[0].isalpha() for o in objs]):
        raise ValueError(
            "not every object in the object list begins with an alphabetic characteter"
        )
    if not all([o[-1].isdigit() for o in objs]):
        raise ValueError("not every object in the object list ends with a digit")

    # add all individual objs, further process obj subranges
    for i, token in enumerate(objs):
        if "-" not in token:
            final_objs.append(token)
        else:
            # break obj subrange
            start, stop = token.split("-")
            start_num = "".join([c for c in start if c.isdigit()])
            stop_num = "".join([c for c in stop if c.isdigit()])
            start_name = "".join([c for c in start if c.isalpha()])
            stop_name = "".join([c for c in stop if c.isalpha()])

            if assert_base_name_same:
                if start_name != stop_name:
                    raise ValueError(
                        f"different range object names '{start_name}' and '{stop_name}' detected."
                    )

            left_pad = len(start_num)
            obj_range = np.arange(int(start_num), int(stop_num) + 1, 1)  # end-inclusive
            for obj in obj_range:
                full_name = f"{start_name}{str(obj).zfill(left_pad)}"
                final_objs.append(full_name)

    # assert length check
    if assert_fixed_len:
        fixed_len = len(final_objs[0])
        if not all([len(n) == fixed_len for n in final_objs]):
            raise ValueError("Not all tokenized objects have the same length")

    # enforce uniqueness
    return np.unique(final_objs).tolist()


def check_max_jobs(userid: str, max_num: int) -> bool:
    """Checks to see if the max number of pending + running jobs is not exceeded"""
    output = subprocess.run(["squeue", "-u", userid], capture_output=True)
    if len(output.stderr) > 0:
        print(output.stderr)
        exit(1)
    num_jobs = len(output.stdout.decode().split("\n")[1:])
    return num_jobs < max_num


def check_until_free(userid: str, max_num) -> None:
    while check_max_jobs(userid, max_num) == False:
        print("Oops, waiting for more space...")


def train_submitter(
    userid: str,
    nodes: List[str],
    filelists: List[pathlib.Path],
    dependency_type: str,
    wait=3,
    stop_on_stderr: bool = False,
    verbose: bool = False,
    additional_options: Optional[Dict] = None,
    jobs_per_node: int = 3,
    max_num: int = 999,
):
    """
    Submits the SLURM dependency train for supplied directories to the specified node

    Parameters
    ----------
    userid:
        `str` of the user ID
    nodes:
        `str` ID of the compute nodes
    filelists:
        `Union[pathlib.Path]` of job files to submit
    wait:
        `int` specifying the time to sleep (in seconds) between each SLURM job submission
    stop_on_stderr:
        If `True`, the train is program is stopped and exits on ANY nonzero `stderr` from
        `subprocess` calls
    dependency_type:
        One of `afterany`, `afterok`, or `afternotok`
    verbose:
        If `True`, submission information is logged to stdout
    additional_options:
        Additional (flagged) SLURM submission options (e.g., `{"--reservation": "my_reservation"}`). These
        will be passed to the `sbatch` call as `"=".join(key,value)`. These options
        apply to all jobs in the dependency train.
    jobs_per_node:
        `int` restricting how many jobs can exist in a node simultaneously
    max_num:
        `int` maximum number of running/pending jobs
    """
    assert dependency_type in ["afterany", "afterok", "afternotok"]
    assert wait >= 0

    for filelist in filelists:
        assert all([p.is_file() for p in filelist])

    if additional_options is not None:
        additional_options = ["=".join([k, v]) for k, v in additional_options.items()]

    assert len(nodes) == len(filelists)

    for node, filelist in zip(nodes, filelists):
        # we need `n_jobs_per_node` dependency trains per node
        trains = np.array_split(filelist, jobs_per_node)
        for job_files in trains:
            if len(job_files) == 0:
                continue
            if additional_options is not None:
                cmd = (
                    [
                        "sbatch",
                        "-J",
                        job_files[0].resolve().stem,
                        "-o",
                        str(job_files[0].parents[0])
                        + "/"
                        + job_files[0].resolve().stem
                        + ".out",
                        "-w",
                        node,
                    ]
                    + additional_options
                    + [str(job_files[0].resolve())]
                )
            else:
                cmd = [
                    "sbatch",
                    "-J",
                    job_files[0].resolve().stem,
                    "-o",
                    str(job_files[0].parents[0])
                    + "/"
                    + job_files[0].resolve().stem
                    + ".out",
                    "-w",
                    node,
                    str(job_files[0].resolve()),
                ]
            if verbose:
                print(
                    f"Submitting job [{job_files[0].resolve()}] on node [{node}]: {' '.join(cmd)}"
                )
            check_until_free(userid, max_num)
            res = subprocess.run(cmd, capture_output=True)
            if len(res.stderr) > 0:
                print(res.stderr)
                if stop_on_stderr:
                    exit()
            res = res.stdout.decode().split()[-1]
            prev_job = job_files[0].resolve().stem
            time.sleep(wait)
            if len(job_files) > 1:
                for idx, job in enumerate(job_files[1:]):
                    if additional_options is not None:
                        cmd = (
                            [
                                "sbatch",
                                "-J",
                                job.resolve().stem,
                                "-o",
                                str(job.parents[0]) + "/" + job.resolve().stem + ".out",
                                "-w",
                                node,
                                f"--dependency={dependency_type}:{res}",
                            ]
                            + additional_options
                            + [str(job.resolve())]
                        )
                    else:
                        cmd = [
                            "sbatch",
                            "-J",
                            job.resolve().stem,
                            "-o",
                            str(job.parents[0]) + "/" + job.resolve().stem + ".out",
                            "-w",
                            node,
                            f"--dependency={dependency_type}:{res}",
                            str(job.resolve()),
                        ]
                    if verbose:
                        print(
                            f"Submitting job [{job.resolve()}] on node [{node}], depending on job [{prev_job}]: {' '.join(cmd)}"
                        )

                    check_until_free(userid, max_num)
                    res = subprocess.run(
                        cmd,
                        capture_output=True,
                    )
                    if len(res.stderr) > 0:
                        print(res.stderr)
                        if stop_on_stderr:
                            exit()
                    res = res.stdout.decode().split()[-1]
                    prev_job = job.resolve().stem
                    time.sleep(wait)


def scan_reservation_open_nodes(
    userid: str, jobs_per_node: int, reservation: str
) -> str:
    """
    Helper fuction for scanning a set of nodes of a reservation, and conditionally
    dumping more jobs to them based on which nodes have room. Currently, this function
    assumes that `uid` is the only one with access to the reservation.

    Parameters
    ----------
    userid:
        `str` of the user ID
    jobs_per_node:
        `int` restricting how many jobs can exist in a node simultaneously
    reservation:
        `str`: specifying the reservation (mandatory)


    Returns
    -------
    found_node:
        `Union[str, NoneType]` representing either the first, greedily found free node as a `str`,
        or `None` if there are no free nodes available
    """

    used_nodes = {}  # nodes being used currently by jobs
    all_nodes = {}  # all nodes from the reservation

    squeue_output = subprocess.run(["squeue", "-u", userid], capture_output=True)
    try:
        jobs = squeue_output.stdout.decode("utf8").split("\n")[1:]  # skip SQUEUE header
    except:
        jobs = []

    scontrol_output = subprocess.run(
        ["scontrol", "show", "reservations"], capture_output=True
    )
    scontrol_output = scontrol_output.stdout.decode("utf8").split("\n")
    for idx, line in enumerate(scontrol_output):
        if line.startswith("ReservationName="):
            tokens = line.split()
            res_name = tokens[0].split("=")[-1]
            if res_name == reservation:
                nodelist = lines[idx + 1].split()[0]
                nodelist = nodelist.split("=")[-1]
                prefix = re.sub("[^a-zA-Z]+", "", nodelist)  # grab only alphas
                nodelist = re.sub("[^0-9,-]+", "", nodelist)  # grab internal nodelist
                nodes = nodelist.split(",")
                nodes = parse_objlist(nodes)
                for node in nodes:
                    all_nodes.append(prefix + node)

    if len(jobs > 1):
        for job in jobs:
            node = job.split()[-1]
            if node not in list(used_nodes.keys()):
                used_nodes[node] = 1
            else:
                used_nodes[node] = used_nodes[node] + 1

    # case zero, no running jobs
    if len(used_nodes) == 0:
        return all_nodes[0]

    # case one, we have a free node
    for node, num_jobs in used_nodes.items():
        if num_jobs < jobs_per_node:
            return node

    # case two, we need a new node:
    for node in all_nodes:
        if node not in used_nodes.keys():
            return node

    # Else, return None to trigger a short wait
    return None


def greedy_reservation_submitter(
    userid: str,
    filelists: List[pathlib.Path],
    reservation: str,
    wait=3,
    stop_on_stderr: bool = False,
    verbose: bool = False,
    additional_options: Optional[Dict] = None,
    jobs_per_node: int = 3,
    max_num: int = 999,
):
    """
    Greedily submits jobs to first available node. Useful for reservations
    with non-static nodelists. The function will run continuously rather than
    waiting for dependencies.

    Parameters
    ----------
    userid:
        `str` of the user ID
    filelists:
        `Union[pathlib.Path]` of job files to submit
    reservation:
        `str`: specifying the reservation (mandatory)
    wait:
        `int` specifying the time to sleep (in seconds) between each greedy node search
    stop_on_stderr:
        If `True`, the train is program is stopped and exits on ANY nonzero `stderr` from
        `subprocess` calls
    verbose:
        If `True`, submission information is logged to stdout
    additional_options:
        Additional (flagged) SLURM submission options (e.g., `{"--reservation": "my_reservation"}`). These
        will be passed to the `sbatch` call as `"=".join(key,value)`. These options
        apply to all jobs in the dependency train.
    jobs_per_node:
        `int` restricting how many jobs can exist in a node simultaneously
    max_num:
        `int` maximum number of running/pending jobs
    """
    if additional_options is not None:
        if "--reservation" in additional_options.keys():
            assert additional_options["--reservation"] == reservation
            del additional_options["--reservation"]

    assert wait >= 0

    for filelist in filelists:
        assert all([p.is_file() for p in filelist])

    if additional_options is not None:
        additional_options = ["=".join([k, v]) for k, v in additional_options.items()]

    for filelist in filelists:
        for file in filelist:
            if len(job_files) == 0:
                continue
            # greedily grab a node or otherwise wait
            node = None
            while node is None:
                node = scan_reservation_open_nodes(
                    uid=uid, jobs_per_node=jobs_per_node, reservation=reservation
                )
                sleep(wait)

            if additional_options is not None:
                cmd = (
                    [
                        "sbatch",
                        "-J",
                        job_files[0].resolve().stem,
                        "-o",
                        str(job_files[0].parents[0])
                        + "/"
                        + job_files[0].resolve().stem
                        + ".out",
                        "-w",
                        node,
                        "--reservation",
                        reservation,
                    ]
                    + additional_options
                    + [str(job_files[0].resolve())]
                )
            else:
                cmd = [
                    "sbatch",
                    "-J",
                    job_files[0].resolve().stem,
                    "-o",
                    str(job_files[0].parents[0])
                    + "/"
                    + job_files[0].resolve().stem
                    + ".out",
                    "-w",
                    node,
                    "--reservation",
                    reservation,
                    str(job_files[0].resolve()),
                ]
            if verbose:
                print(
                    f"Submitting job [{job_files[0].resolve()}] on node [{node}]: {' '.join(cmd)}"
                )
            check_until_free(userid, max_num)
            res = subprocess.run(cmd, capture_output=True)
            if len(res.stderr) > 0:
                print(res.stderr)
                if stop_on_stderr:
                    exit()


def equipartition_files(nodelist: List[str], dirlist: List[str]):
    """Equally scatters all SBATCH submission files found in every dir in dirlist
    over the provided nodelist
    """
    all_files = []
    for d in dirlist:
        path = pathlib.Path(d)
        files = sorted(list(path.glob("*.sh")))
        all_files.extend(files)

    num_nodes = len(nodelist)
    filesets = list(np.array_split(all_files, num_nodes))
    assert len(filesets) == len(nodelist)
    return nodelist, filesets


def main():
    parser = parse_input()

    opts = parser.parse_args()
    if opts.verbose:
        print("Running with options:")
        print(opts)

    final_dirs = parse_objlist(
        opts.dirlist, assert_base_name_same=False, assert_fixed_len=False
    )
    userid = subprocess.run(["whoami"], capture_output=True)
    userid = userid.stdout.decode().strip()

    if opts.additional_options is not None:
        additional_options = {}
        flagopts = opts.additional_options.split(",")
        assert len(flagopts) % 2 == 0
        flags = [f for f in flagopts[::2]]
        fopts = [o for o in flagopts[1::2]]
        assert len(flags) == len(fopts)
        for f, o in zip(flags, fopts):
            if len(f) > 1:
                pre = "--"
            else:
                pre = "-"
            additional_options.update({pre + f: o})

    else:
        additional_options = None

    if opts.nodelist is not None:
        final_nodes = parse_objlist(opts.nodelist)
        nodes, filelists = equipartition_files(final_nodes, final_dirs)
        train_submitter(
            userid=userid,
            nodes=nodes,
            filelists=filelists,
            dependency_type=opts.dependency_type,
            wait=opts.wait,
            stop_on_stderr=opts.stop_on_stderr,
            verbose=opts.verbose,
            additional_options=additional_options,
            jobs_per_node=opts.jobs_per_node,
            max_num=opts.max_num_jobs,
        )

    if opts.nodelist is None:
        if opts.reservation is None:
            raise RuntimeError(
                "With --nodelist unspecified, a reservation must be specified via --reservation (yes, even if that information is already in the jobscripts)"
            )
        greedy_submitter(
            userid=userid,
            filelists=filelists,
            reservation=opts.reservation,
            wait=opts.wait,
            stop_on_stderr=opts.stop_on_stderr,
            verbose=opts.verbose,
            additional_options=additional_options,
            jobs_per_node=opts.jobs_per_node,
            max_num=opts.max_num_jobs,
        )


if __name__ == "__main__":
    main()
