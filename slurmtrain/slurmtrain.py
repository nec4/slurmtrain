import subprocess
import pathlib
import time
import argparse
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
import math
import multiprocessing
import os


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
        help="node range or list of the form `x1, x1`, `x1-x10` or a mixture, i,e, `x1,x2-x10`. Assumes that node names are of the form `^[a-zA-Z]` + `[0-9]$` (some alphas and then a numeric ending). All nodes must share the same basename.",
    )
    parser.add_argument(
        "--dependency-type",
        type=str,
        help="SLURM dependency type. Must be one of `after`, `afterany`, `afterok`, `afternotok`. Defaults to `afterany`.",
        default="afterany",
    )
    parser.add_argument(
        "--dirs-per-node",
        type=int,
        help="Number of dependency trains/directories per node",
        default=1,
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
        "--num-avail-nodes",
        help="maximum number of nodes in the pool when not using --nodelist",
        type=int,
        default=1,
    )

    return parser


def parse_objlist(
    objlist: str, assert_base_name_same=True, assert_fixed_len=True
) -> List[str]:
    """
    Turns a single string objlist option into a full expanded list of tokenized objs. E.g.)

    `"bgn001,bgn002-bgn04"` returns `["bgn001", "bgn002", "bgn003", bgn"004"]`

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


def partition_submitter(
    userid: str,
    directories: Union[List[str], str],
    dependency_type: str,
    wait=3,
    stop_on_stderr: bool = False,
    verbose: bool = False,
    additional_options: Optional[Dict] = None,
    num_avail_nodes=1,
    jobs_per_node=1,
    max_num=999,
):
    """
    Submits the SLURM dependency train for supplied directories to the specified nodepool until all nodes are used

    Parameters
    ----------
    directories:
        `Union[List[str], str]` of directories from which job scripts should
        be submitted.
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
    """
    assert dependency_type in ["afterany", "afterok", "afternotok"]
    assert wait >= 0

    if isinstance(directories, str):
        directories = [directories]

    paths = [pathlib.Path(d) for d in directories]
    assert all([p.is_dir() for p in paths])

    if additional_options is not None:
        additional_options = ["=".join([k, v]) for k, v in additional_options.items()]

    for path in paths:
        all_job_files = np.array(sorted(list(path.glob("*.sh"))))
        node_subsets = np.array_split(all_job_files, num_avail_nodes)
        for node_subset in node_subsets:
            train_subsets = np.array_split(node_subset, jobs_per_node)
            for job_files in train_subsets:
                if not isinstance(job_files, np.ndarray):
                    job_files = np.array([job_files])
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
                        ]
                        + additional_options
                        + [str(job_files[0].resolve())]
                    )
                else:
                    print(job_files[0])
                    cmd = [
                        "sbatch",
                        "-J",
                        job_files[0].resolve().stem,
                        "-o",
                        str(job_files[0].parents[0])
                        + "/"
                        + job_files[0].resolve().stem
                        + ".out",
                        str(job_files[0].resolve()),
                    ]
                if verbose:
                    print(
                        f"Submitting job [{job_files[0].resolve().stem}]: {' '.join(cmd)}"
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
                                    str(job.parents[0])
                                    + "/"
                                    + job.resolve().stem
                                    + ".out",
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
                                f"--dependency={dependency_type}:{res}",
                                str(job.resolve()),
                            ]
                        if verbose:
                            print(
                                f"Submitting job [{job.resolve().stem}], depending on job [{prev_job}]: {' '.join(cmd)}"
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

    if isinstance(directories, str):
        directories = [directories]

    for filelist in filelists:
        assert all([p.is_file() for p in fileslist])

    if additional_options is not None:
        additional_options = ["=".join([k, v]) for k, v in additional_options.items()]

    assert len(nodes) == len(filelists)

    for node, filelist in zip(nodes, filelists):
        # we need `n_jobs_per_node` dependency trains per node
        trains = np.array_split(filelist, jobs_per_node)
        for job_files in trains:
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
                    f"Submitting job [{job_files[0].resolve().stem}] on node [{node}]: {' '.join(cmd)}"
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
                        f"Submitting job [{job.resolve().stem}] on node [{node}], depending on job [{prev_job}]: {' '.join(cmd)}"
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


def partition_dirs(
    nodelist: List[str],
    dirlist: List[str],
    dirs_per_node: int,
    expand_dirs: bool = False,
) -> List[Tuple[str, List[str]]]:
    """
    Assigns sets of directories to nodes for dependency trains, depending on
    the chosen number of directories per node. Nodes are filled sequentially
    until all directories are assigned

    Parameters
    ----------
    nodelist:
        `List[str]` of node names
    dirlist:
        `List[str]` of directories
    dirs_per_node:
        `int` specifying the maximum number of directories assigned to a node
    expand_dirs:
        if `True`, a single filelist will be returned instead of a list of directories

    Returns
    -------
    assignments:
        `List[Tuple[str,List[str]]]` of assignemts of the form (eg with
        dirs_per_node=2):

            [
                ("nodeX", ["dir1", "dir2", ...]),
                ("nodeY", ["dir3", "dir4", ...]),
                ...
            ]

        This output is meant to be passed to `train_submitter`. If `expand_dirs` is `True`,
        each node will be returned with a single list of all files across the directories
    """
    assignments = []
    num_groups = math.ceil(len(dirlist) / dirs_per_node)
    if num_groups > len(nodelist):
        raise RuntimeError(
            f"Not enough nodes ({len(nodelist)} for {len(dirlist)} and {dirs_per_node} directories per node."
        )

    # zip will automatically terminate at the end of the shortest list in the zip
    for node, group in zip(nodelist, range(num_groups)):
        dirs = dirlist[group * dirs_per_node : (group + 1) * dirs_per_node]
        assignments.append(tuple([node, dirs]))

    if expand_dirs:
        for idx, a in enumerate(assignments):
            node, dirs = a
            filelist = []
            for d in dirs:
                filelist.append(sorted(list(pathlib.Path(d).glob(".sh"))))
            assignments[idx] = tuple([node, filelist])
    return assignments


def main():
    parser = parse_input()

    opts = parser.parse_args()
    if opts.verbose:
        print("Running with options:")
        print(opts)

    if opts.nodelist is not None:
        final_nodes = parse_objlist(opts.nodelist)
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

    if opts.nodelist != None:
        assignments = partition_dirs(
            final_nodes, final_dirs, opts.dirs_per_node, expand_dirs=True
        )
        nodes = []
        filelists = []
        for a in assignments:
            nodes.append(a[0])
            filelists.append(a[1])

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
    else:
        partition_submitter(
            userid=userid,
            directories=final_dirs,
            dependency_type=opts.dependency_type,
            wait=opts.wait,
            stop_on_stderr=opts.stop_on_stderr,
            verbose=opts.verbose,
            additional_options=additional_options,
            jobs_per_node=opts.jobs_per_node,
            num_avail_nodes=opts.num_avail_nodes,
            max_num=opts.max_num_jobs,
        )


if __name__ == "__main__":
    main()
