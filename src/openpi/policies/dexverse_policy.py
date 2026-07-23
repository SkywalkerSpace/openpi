"""
DexVerse Shadow Hand policy transform for π₀.₅ base model
state/action dim 根据你的实际机器人修改
"""
import dataclasses
from typing import ClassVar
import einops
import numpy as np
from openpi import transforms


def make_dexverse_example(state_dim: int = 24) -> dict:
    return {
        "state": np.zeros((state_dim,), dtype=np.float32),
        "images": {
            "cam_high":        np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist":  np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "prompt": "open the laptop",
    }


@dataclasses.dataclass(frozen=True)
class DexverseInputs(transforms.DataTransformFn):
    """
    Shadow Hand 输入 transform。
    state_dim: 机器人关节数，单手24 / 双手48 / 加上手臂更多
    """
    state_dim: int = 24

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high", "cam_left_wrist", "cam_right_wrist"
    )

    def __call__(self, data: dict) -> dict:
        # ── state ──────────────────────────────────────────────
        state = np.asarray(data["state"], dtype=np.float32)
        assert state.shape == (self.state_dim,), (
            f"state shape {state.shape} != ({self.state_dim},)"
        )

        # ── images：CHW uint8 → HWC ──────────────────────────
        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3, 4):
                img = einops.rearrange(img, "c h w -> h w c")
            return img

        in_images = data["images"]
        base_image = convert_image(in_images["cam_high"])

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra = {
            "left_wrist_0_rgb":  "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, src in extra.items():
            if src in in_images:
                images[dest] = convert_image(in_images[src])
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image":      images,
            "image_mask": image_masks,
            "state":      state,
        }
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        return inputs


@dataclasses.dataclass(frozen=True)
class DexverseOutputs(transforms.DataTransformFn):
    """
    输出 transform：取前 state_dim 维，不做关节翻转。
    """
    state_dim: int = 24

    def __call__(self, data: dict) -> dict:
        # π₀.₅ 输出维度可能 > state_dim，截取有效部分
        actions = np.asarray(data["actions"][:, :self.state_dim], dtype=np.float32)
        return {"actions": actions}
