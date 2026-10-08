import pytest

from zigvideo.benchmark import summarize_stage_timings


def test_timing_reports_separate_denoising_and_vae():
    metrics = summarize_stage_timings([20.0, 87.0, 154.0], 200.0, 360.0)
    assert metrics["prompt_setup_and_first_step_seconds"] == 20.0
    assert metrics["subsequent_step_seconds"] == [67.0, 67.0]
    assert metrics["subsequent_step_mean_seconds"] == 67.0
    assert metrics["time_to_final_latents_seconds"] == 154.0
    assert metrics["vae_decode_seconds"] == 200.0
    assert metrics["remaining_pipeline_seconds"] == 6.0
    assert metrics["pipeline_total_seconds"] == 360.0


def test_timing_one_step_does_not_mislabel_pipeline_duration_as_step():
    metrics = summarize_stage_timings([106.4], 200.0, 309.49)
    assert metrics["subsequent_step_mean_seconds"] is None
    assert metrics["prompt_setup_and_first_step_seconds"] == 106.4
    assert metrics["vae_decode_seconds"] == 200.0
    assert metrics["pipeline_total_seconds"] == 309.49


def test_timing_rejects_invalid_step_checkpoints():
    with pytest.raises(ValueError, match="At least one"):
        summarize_stage_timings([], 1.0, 10.0)
    with pytest.raises(ValueError, match="monotonically"):
        summarize_stage_timings([4.0, 2.0], 1.0, 10.0)
