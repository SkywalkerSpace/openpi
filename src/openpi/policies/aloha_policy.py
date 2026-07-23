import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


# ================= 核心维度配置 =================
ARM_DOF = 7          # 单臂自由度
HAND_DOF = 21        # 单个灵巧手自由度 (21 维适用于常见五指灵巧手)
ACTION_DIM = 2 * (ARM_DOF + HAND_DOF)  # 外部仿真/硬件环境真实所需的总维度 (56)
MODEL_BASE_DIM = 64  # pi05_base 预训练时的底层最大预留维度 (通常为 64)
# ==============================================


def make_aloha_example() -> dict:
    """Creates a random input example for the 56-DOF policy."""
    return {
        "state": np.ones((ACTION_DIM,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "grasp the object",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Inputs wrapper modified for 56-DOF pi05_base mapping."""

    adapt_to_pi: bool = True
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        # ----------- State 补零逻辑 (56 -> 64) -----------
        state = np.asarray(data["state"])
        if state.shape[-1] < MODEL_BASE_DIM:
            pad_width = MODEL_BASE_DIM - state.shape[-1]
            state = np.pad(state, (0, pad_width), mode='constant')
        elif state.shape[-1] > MODEL_BASE_DIM:
            state = state[..., :MODEL_BASE_DIM]
        # ------------------------------------------------

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            
            # ----------- Action 补零逻辑 (56 -> 64) -----------
            if actions.shape[-1] < MODEL_BASE_DIM:
                pad_width = MODEL_BASE_DIM - actions.shape[-1]
                actions = np.pad(actions, ((0, 0), (0, pad_width)), mode='constant')
            elif actions.shape[-1] > MODEL_BASE_DIM:
                actions = actions[..., :MODEL_BASE_DIM]
            # --------------------------------------------------
                
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Outputs wrapper modified for 56-DOF pi05_base mapping."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        raw_actions = np.asarray(data["actions"])
        
        # 获取当前模型的实际输出维度 (例如 32)
        current_dim = raw_actions.shape[-1]
        
        # ----------- Action 维度补齐/截断逻辑 -----------
        if current_dim < ACTION_DIM:
            # 如果模型输出维度小于环境需要的 56 维，则在最后一个维度补零
            pad_width = ACTION_DIM - current_dim
            # 注意：raw_actions 的形状通常是 (horizon, action_dim) 比如 (50, 32)
            valid_actions = np.pad(raw_actions, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # 如果模型输出维度大于等于 56 维，则截断
            valid_actions = raw_actions[..., :ACTION_DIM]
        # --------------------------------------------------
        
        return {"actions": _encode_actions(valid_actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """透传所有自由度，不再反转符号。"""
    return np.ones(ACTION_DIM, dtype=np.float32)


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        state = _joint_flip_mask() * state
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
    return actions
