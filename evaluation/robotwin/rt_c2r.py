"""RoboTwin RT-C2R benchmark definitions.

RT-C2R evaluates a clean-trained policy on clean and randomized RoboTwin
rollout distributions.  The split definitions here intentionally only touch
the domain randomization keys consumed by RoboTwin task_config YAML files.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any


TASK_NAMES = [
    "stack_bowls_three",
    "handover_block",
    "hanging_mug",
    "scan_object",
    "lift_pot",
    "put_object_cabinet",
    "stack_blocks_three",
    "place_shoe",
    "adjust_bottle",
    "place_mouse_pad",
    "dump_bin_bigbin",
    "move_pillbottle_pad",
    "pick_dual_bottles",
    "shake_bottle",
    "place_fan",
    "turn_switch",
    "shake_bottle_horizontally",
    "place_container_plate",
    "rotate_qrcode",
    "place_object_stand",
    "put_bottles_dustbin",
    "move_stapler_pad",
    "place_burger_fries",
    "place_bread_basket",
    "pick_diverse_bottles",
    "open_microwave",
    "beat_block_hammer",
    "press_stapler",
    "click_bell",
    "move_playingcard_away",
    "open_laptop",
    "move_can_pot",
    "stack_bowls_two",
    "place_a2b_right",
    "stamp_seal",
    "place_object_basket",
    "handover_mic",
    "place_bread_skillet",
    "stack_blocks_two",
    "place_cans_plasticbox",
    "click_alarmclock",
    "blocks_ranking_size",
    "place_phone_stand",
    "place_can_basket",
    "place_object_scale",
    "place_a2b_left",
    "grab_roller",
    "place_dual_shoes",
    "place_empty_cup",
    "blocks_ranking_rgb",
]


SPLIT_RANDOMIZATION = OrderedDict(
    [
        (
            "easy",
            {
                "random_background": False,
                "random_light": False,
                "cluttered_table": False,
                "random_table_height": 0,
                "random_head_camera_dis": 0,
                "random_embodiment": False,
            },
        ),
        (
            "background",
            {
                "random_background": True,
                "random_light": False,
                "cluttered_table": False,
                "random_table_height": 0,
                "random_head_camera_dis": 0,
                "random_embodiment": False,
            },
        ),
        (
            "light",
            {
                "random_background": False,
                "random_light": True,
                "cluttered_table": False,
                "random_table_height": 0,
                "random_head_camera_dis": 0,
                "random_embodiment": False,
            },
        ),
        (
            "clutter",
            {
                "random_background": False,
                "random_light": False,
                "cluttered_table": True,
                "random_table_height": 0,
                "random_head_camera_dis": 0,
                "random_embodiment": False,
            },
        ),
        (
            "height",
            {
                "random_background": False,
                "random_light": False,
                "cluttered_table": False,
                "random_table_height": 0.03,
                "random_head_camera_dis": 0,
                "random_embodiment": False,
            },
        ),
        (
            "hard",
            {
                "random_background": True,
                "random_light": True,
                "cluttered_table": True,
                "random_table_height": 0.03,
                "random_head_camera_dis": 0,
                "random_embodiment": False,
            },
        ),
    ]
)


def task_config_name(split: str) -> str:
    return f"rt_c2r_{split}"


def stable_eval_seed(
    split: str,
    task_name: str,
    episode_id: int,
    base_seed: int = 20260622,
) -> int:
    payload = f"rt-c2r-v1:{base_seed}:{split}:{task_name}:{episode_id}".encode()
    digest = hashlib.sha256(payload).digest()
    return 10000 + int.from_bytes(digest[:4], "big") % 900000000


def manifest_record(
    split: str,
    task_name: str,
    task_index: int,
    episode_id: int,
    base_seed: int = 20260622,
    candidate_id: int | None = None,
) -> dict[str, Any]:
    seed_source_id = episode_id if candidate_id is None else candidate_id
    record = {
        "benchmark": "robotwin_rt_c2r",
        "benchmark_version": "v1",
        "task": task_name,
        "task_index": task_index,
        "split": split,
        "task_config": task_config_name(split),
        "episode_id": episode_id,
        "seed": stable_eval_seed(split, task_name, seed_source_id, base_seed),
        "randomization": SPLIT_RANDOMIZATION[split],
    }
    if candidate_id is not None:
        record["candidate_id"] = candidate_id
    return record
