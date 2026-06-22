"""Generate RT-C2R manifests filtered by RoboTwin scripted expert success.

This script runs RoboTwin's built-in task expert/planner for candidate seeds and
only keeps cases where the expert can solve the initialized scene.  The output
manifest is therefore safe to use with STRICT_SEED_MANIFEST=True.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.robotwin.generate_rt_c2r_benchmark import write_task_configs  # noqa: E402
from evaluation.robotwin.rt_c2r import (  # noqa: E402
    SPLIT_RANDOMIZATION,
    TASK_NAMES,
    manifest_record,
    task_config_name,
)


def robotwin_root_from_env() -> Path:
    root = os.environ.get("ROBOTWIN_ROOT") or os.environ.get("ROBOWIN_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return (PROJECT_ROOT / "RoboTwin").resolve()


def ensure_egl_vendor_dirs() -> None:
    for path in ("/etc/glvnd/egl_vendor.d", "/usr/share/glvnd/egl_vendor.d"):
        if os.path.isdir(path):
            continue
        if os.geteuid() == 0:
            os.makedirs(path, exist_ok=True)
            print(f"Created missing EGL vendor directory: {path}")
        else:
            raise FileNotFoundError(
                f"Missing EGL vendor directory: {path}. "
                "Create it before running SAPIEN/RoboTwin."
            )


def install_warp_torch_compat() -> None:
    import importlib
    import warp as wp

    try:
        wp.torch = importlib.import_module("warp.torch")
    except ModuleNotFoundError:
        wp.torch = SimpleNamespace(
            device_from_torch=wp.device_from_torch,
            device_to_torch=wp.device_to_torch,
            dtype_from_torch=wp.dtype_from_torch,
            dtype_to_torch=wp.dtype_to_torch,
            from_torch=wp.from_torch,
            to_torch=wp.to_torch,
            stream_from_torch=wp.stream_from_torch,
            stream_to_torch=wp.stream_to_torch,
        )


def configure_robotwin_imports(robotwin_root: Path) -> None:
    if not (robotwin_root / "envs").is_dir():
        raise FileNotFoundError(
            f"RoboTwin root is not configured correctly: {robotwin_root}. "
            "Set ROBOTWIN_ROOT=/path/to/RoboTwin."
        )
    for import_root in (PROJECT_ROOT, robotwin_root):
        import_root_str = str(import_root)
        if import_root_str not in sys.path:
            sys.path.insert(0, import_root_str)
    os.chdir(robotwin_root)
    install_warp_torch_compat()


def class_decorator(task_name: str):
    import importlib

    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception as exc:
        raise SystemExit(f"No task env found for {task_name}") from exc


def get_embodiment_config(robot_file: str) -> dict[str, Any]:
    import yaml

    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def build_task_args(
    robotwin_root: Path,
    task_name: str,
    split: str,
) -> dict[str, Any]:
    import yaml

    from envs import CONFIGS_PATH

    task_config = task_config_name(split)
    task_config_path = robotwin_root / "task_config" / f"{task_config}.yml"
    if not task_config_path.is_file():
        raise FileNotFoundError(
            f"Task config not found: {task_config_path}. Run with "
            "--write-task-configs or generate rt_c2r_*.yml first."
        )

    with task_config_path.open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["task_config_path"] = str(task_config_path)
    args["ckpt_setting"] = "expert_validation"
    args["save_root"] = str(PROJECT_ROOT / "results" / "rt_c2r_expert_validation")
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_log"] = False

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(cur_embodiment_type: str) -> str:
        robot_file = embodiment_types[cur_embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment file for {cur_embodiment_type}")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def close_env_quietly(task_env) -> None:
    try:
        task_env.close_env()
    except Exception:
        pass


def validate_candidate(
    task_env,
    args: dict[str, Any],
    seed: int,
    now_id: int,
    verbose_failures: bool = False,
) -> tuple[bool, str]:
    from envs.utils.create_actor import UnStableError

    try:
        task_env.setup_demo(now_ep_num=now_id, seed=seed, is_test=True, **args)
        task_env.play_once()
        valid = bool(task_env.plan_success and task_env.check_success())
        close_env_quietly(task_env)
        return valid, "ok" if valid else "expert_plan_or_success_check_failed"
    except UnStableError:
        close_env_quietly(task_env)
        return False, "unstable"
    except Exception as exc:
        close_env_quietly(task_env)
        if verbose_failures:
            import traceback

            traceback.print_exc()
        return False, f"{type(exc).__name__}: {exc}"


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_task_split(
    robotwin_root: Path,
    split: str,
    task_name: str,
    task_index: int,
    episodes_per_task: int,
    base_seed: int,
    max_candidates_per_task: int,
    output_file: Path,
    failure_file: Path | None,
    verbose_failures: bool,
) -> None:
    args = build_task_args(robotwin_root, task_name, split)
    task_env = class_decorator(task_name)
    task_env.suc = 0
    task_env.test_num = 0

    valid_count = 0
    candidate_id = 0
    while valid_count < episodes_per_task and candidate_id < max_candidates_per_task:
        record = manifest_record(
            split=split,
            task_name=task_name,
            task_index=task_index,
            episode_id=valid_count,
            candidate_id=candidate_id,
            base_seed=base_seed,
        )
        valid, reason = validate_candidate(
            task_env=task_env,
            args=args,
            seed=int(record["seed"]),
            now_id=valid_count,
            verbose_failures=verbose_failures,
        )
        if valid:
            record["expert_validated"] = True
            record["expert_validation"] = {
                "candidate_id": candidate_id,
                "rulebase": "RoboTwin scripted expert + CuRobo planner + check_success",
            }
            write_jsonl_record(output_file, record)
            valid_count += 1
            print(
                f"[{split}][{task_name}] kept {valid_count}/{episodes_per_task} "
                f"seed={record['seed']} candidate={candidate_id}",
                flush=True,
            )
        elif failure_file is not None:
            fail_record = dict(record)
            fail_record["expert_validated"] = False
            fail_record["failure_reason"] = reason
            write_jsonl_record(failure_file, fail_record)
        candidate_id += 1

    if valid_count < episodes_per_task:
        raise RuntimeError(
            f"Only found {valid_count}/{episodes_per_task} valid seeds for "
            f"split={split}, task={task_name} after {max_candidates_per_task} candidates."
        )


def validate_task_split_worker(worker_args: dict[str, Any]) -> dict[str, Any]:
    robotwin_root = Path(worker_args["robotwin_root"])
    configure_robotwin_imports(robotwin_root)
    output_file = Path(worker_args["output_file"])
    failure_file = (
        None
        if worker_args["failure_file"] is None
        else Path(worker_args["failure_file"])
    )
    if output_file.exists():
        output_file.unlink()
    if failure_file is not None and failure_file.exists():
        failure_file.unlink()

    validate_task_split(
        robotwin_root=robotwin_root,
        split=worker_args["split"],
        task_name=worker_args["task_name"],
        task_index=worker_args["task_index"],
        episodes_per_task=worker_args["episodes_per_task"],
        base_seed=worker_args["base_seed"],
        max_candidates_per_task=worker_args["max_candidates_per_task"],
        output_file=output_file,
        failure_file=failure_file,
        verbose_failures=worker_args["verbose_failures"],
    )
    return {
        "split": worker_args["split"],
        "task_name": worker_args["task_name"],
        "task_index": worker_args["task_index"],
        "output_file": str(output_file),
    }


def merge_split_outputs(
    split: str,
    tasks: list[str],
    task_index_by_name: dict[str, int],
    tmp_dir: Path,
    output_file: Path,
) -> None:
    if output_file.exists():
        output_file.unlink()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as out:
        for task_name in tasks:
            task_index = task_index_by_name[task_name]
            task_file = tmp_dir / split / f"{task_index:02d}_{task_name}.jsonl"
            if not task_file.is_file():
                raise FileNotFoundError(f"Missing validated task output: {task_file}")
            with task_file.open("r", encoding="utf-8") as src:
                for line in src:
                    out.write(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "robotwin" / "rt_c2r_validated_manifests",
    )
    parser.add_argument(
        "--failure-log-dir",
        default=str(PROJECT_ROOT / "evaluation" / "robotwin" / "rt_c2r_validation_failures"),
        help="Directory for rejected candidate logs. Pass an empty string to disable.",
    )
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--max-candidates-per-task", type=int, default=2000)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 for sequential validation.",
    )
    parser.add_argument(
        "--verbose-failures",
        action="store_true",
        help="Print full tracebacks for rejected candidate seeds.",
    )
    parser.add_argument("--base-seed", type=int, default=20260622)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLIT_RANDOMIZATION.keys()),
        choices=list(SPLIT_RANDOMIZATION.keys()),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASK_NAMES,
        choices=TASK_NAMES,
        help="Subset of task names to validate.",
    )
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=robotwin_root_from_env(),
    )
    parser.add_argument(
        "--write-task-configs",
        action="store_true",
        help="Generate rt_c2r_*.yml from demo_clean.yml before validation.",
    )
    parser.add_argument("--template-name", default="demo_clean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    robotwin_root = args.robotwin_root.expanduser().resolve()
    ensure_egl_vendor_dirs()

    if args.write_task_configs:
        write_task_configs(
            robotwin_root=robotwin_root,
            template_name=args.template_name,
            splits=args.splits,
            output_dir=None,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failure_log_dir = args.failure_log_dir
    if failure_log_dir == "":
        failure_log_dir = None
    else:
        failure_log_dir = Path(failure_log_dir)

    task_index_by_name = {task_name: idx for idx, task_name in enumerate(TASK_NAMES)}
    if args.num_workers <= 1:
        configure_robotwin_imports(robotwin_root)
        for split in args.splits:
            output_file = args.output_dir / f"{task_config_name(split)}.jsonl"
            if output_file.exists():
                output_file.unlink()
            for task_name in args.tasks:
                failure_file = None
                if failure_log_dir is not None:
                    failure_file = (
                        failure_log_dir
                        / split
                        / f"{task_index_by_name[task_name]:02d}_{task_name}.jsonl"
                    )
                    if failure_file.exists():
                        failure_file.unlink()

                validate_task_split(
                    robotwin_root=robotwin_root,
                    split=split,
                    task_name=task_name,
                    task_index=task_index_by_name[task_name],
                    episodes_per_task=args.episodes_per_task,
                    base_seed=args.base_seed,
                    max_candidates_per_task=args.max_candidates_per_task,
                    output_file=output_file,
                    failure_file=failure_file,
                    verbose_failures=args.verbose_failures,
                )
            print(f"Wrote validated split manifest: {output_file}")
        return

    tmp_dir = args.output_dir / "_tmp_by_task"
    worker_jobs: list[dict[str, Any]] = []
    for split in args.splits:
        for task_name in args.tasks:
            task_index = task_index_by_name[task_name]
            failure_file = None
            if failure_log_dir is not None:
                failure_file = failure_log_dir / split / f"{task_index:02d}_{task_name}.jsonl"
            worker_jobs.append(
                {
                    "robotwin_root": str(robotwin_root),
                    "split": split,
                    "task_name": task_name,
                    "task_index": task_index,
                    "episodes_per_task": args.episodes_per_task,
                    "base_seed": args.base_seed,
                    "max_candidates_per_task": args.max_candidates_per_task,
                    "output_file": str(tmp_dir / split / f"{task_index:02d}_{task_name}.jsonl"),
                    "failure_file": None if failure_file is None else str(failure_file),
                    "verbose_failures": args.verbose_failures,
                }
            )

    print(f"Validating {len(worker_jobs)} task/split jobs with {args.num_workers} workers")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_job = {
            executor.submit(validate_task_split_worker, job): job for job in worker_jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            job = future_to_job[future]
            result = future.result()
            print(
                f"Finished [{result['split']}][{result['task_name']}] "
                f"-> {result['output_file']}",
                flush=True,
            )

    for split in args.splits:
        output_file = args.output_dir / f"{task_config_name(split)}.jsonl"
        merge_split_outputs(
            split=split,
            tasks=args.tasks,
            task_index_by_name=task_index_by_name,
            tmp_dir=tmp_dir,
            output_file=output_file,
        )
        print(f"Wrote validated split manifest: {output_file}")


if __name__ == "__main__":
    main()
