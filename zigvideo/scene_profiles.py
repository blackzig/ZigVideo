"""Optional, reproducible composition/motion prompts for controlled CogVideoX tests.

These are hypotheses, not model guarantees. Preserve the ordinary --prompt path
and its original negative prompt unless an experiment is explicitly selected.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, distorted, deformed, abstract, unrecognizable subject, "
    "low quality, inconsistent motion"
)


@dataclass(frozen=True)
class SceneProfile:
    prompt: str
    negative_prompt: str
    goal: str


SCENE_PROFILES: dict[str, SceneProfile] = {
    "robot-walk": SceneProfile(
        prompt=(
            "A single small friendly humanoid robot walks slowly toward the camera "
            "on a rain-soaked futuristic city sidewalk at night. "
            "Full-body wide shot: the complete robot is visible from the top of its "
            "head to both feet at every moment, with clear space above the head "
            "and below the feet. The robot remains centered precisely in the "
            "middle of the image, within the narrow central strip that will "
            "be cropped to a vertical 9:16 video. "
            "The robot is modest in size within the scene, not a close-up. "
            "Its left foot visibly lifts, advances and touches down, then "
            "its right foot lifts, advances and touches down. "
            "Both legs alternate and the arms swing gently in opposite directions. "
            "A continuous single shot, stationary tripod camera, steady distance, "
            "clear silhouette. Wet pavement and soft neon lights in the distance, "
            "simple uncluttered background, cinematic but clearly lit."
        ),
        negative_prompt=(
            "cars, vehicles, buses, trucks, motorcycles, nearby machinery, "
            "obstructions, another character, extra arms, extra legs, "
            "close-up, face-only shot, cut-off head, cut-off feet, cropped legs, "
            "robot at frame edge, camera pan, zoom, camera shake, scene cut, "
            "standing still, sliding feet, floating, static pose, "
            + DEFAULT_NEGATIVE_PROMPT
        ),
        goal=(
            "One recognizable head-to-feet robot centered in the portrait-safe "
            "middle of the native landscape image, with visibly alternating steps. "
            "Not guaranteed by prompting; compare exported frames visually."
        ),
    ),
}


@dataclass(frozen=True)
class GenerationText:
    prompt: str
    negative_prompt: str
    scene_profile: str | None
    quality_goal: str | None


def resolve_generation_text(
    prompt: str | None,
    scene_profile: str | None = None,
    negative_prompt: str | None = None,
) -> GenerationText:
    """Require exactly one scene source and preserve the existing default."""
    if scene_profile is not None:
        if scene_profile not in SCENE_PROFILES:
            raise ValueError(f"Unknown --scene-profile: {scene_profile}")
        if prompt is not None:
            raise ValueError("Use --prompt OR --scene-profile, not both.")
        selected = SCENE_PROFILES[scene_profile]
        return GenerationText(
            prompt=selected.prompt,
            negative_prompt=(
                selected.negative_prompt
                if negative_prompt is None else _nonempty_negative(negative_prompt)
            ),
            scene_profile=scene_profile,
            quality_goal=selected.goal,
        )
    if prompt is None or not prompt.strip():
        raise ValueError("Provide a nonempty --prompt or choose --scene-profile.")
    return GenerationText(
        prompt=prompt,
        negative_prompt=(
            DEFAULT_NEGATIVE_PROMPT
            if negative_prompt is None else _nonempty_negative(negative_prompt)
        ),
        scene_profile=None,
        quality_goal=None,
    )


def _nonempty_negative(text: str) -> str:
    if not text.strip():
        raise ValueError("--negative-prompt must not be blank.")
    return text
