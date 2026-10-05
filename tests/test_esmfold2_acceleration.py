"""Native component equivalence; run in the isolated ESM runtime.

The CUDA cases exercise replay, ownership, input changes and RNG refusal.
Full checkpoint comparisons live in the public benchmark harness.
"""

from types import MethodType

import pytest
import torch

pytest.importorskip("esm")
from esm.models.esmfold2 import layers

from boltzgen.task.esmfold2.acceleration import (
    AcceleratedInference,
    MaskCache,
    _SOURCE_HASHES,
    cached_attention,
    check_source,
    acceleration_context,
)
from boltzgen.task.esmfold2.contract import ACCELERATION_REVISION
from boltzgen.task.esmfold2.sampling import sample_without_scalar_sync


def test_copied_functions_match_pinned_native_source():
    check_source(layers.DiffusionStructureHead.sample, _SOURCE_HASHES["sample"])
    check_source(layers.SWA3DRoPEAttention.forward, _SOURCE_HASHES["forward"])
    with pytest.raises(ValueError, match="Unsupported ESMFold2 source"):
        check_source(layers.SWA3DRoPEAttention.forward, "wrong-version")


def test_uninspectable_callable_preserves_working_native_forward(monkeypatch):
    native_forward = layers.SWA3DRoPEAttention.forward

    class WrappedForward:
        def __get__(self, instance, owner=None):
            return self if instance is None else MethodType(self, instance)

        def __call__(self, instance, *args, **kwargs):
            return native_forward(instance, *args, **kwargs)

    model = torch.nn.Module()
    model.attn = layers.SWA3DRoPEAttention(32, 4, half_window=3)
    model.eval().requires_grad_(False)
    x = torch.randn(1, 8, 32)
    params = (torch.randn(1, 8, 4), torch.randn(1, 8, 4))
    options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
    with torch.inference_mode():
        expected = model.attn(x, params)
        monkeypatch.setattr(layers.SWA3DRoPEAttention, "forward", WrappedForward())
        with acceleration_context(model, options) as execution:
            actual = model.attn(x, params)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert execution["effective"] == "off"
    assert "Cannot inspect ESMFold2 source" in execution["fallback_reason"]


def test_context_restores_methods_and_releases_state_after_failure():
    # Small real components exercise installation/cleanup without a checkpoint.
    model = torch.nn.Module()
    model.folding_trunk = layers.FoldingTrunk(n_layers=1, d_pair=32)
    model.structure_head = torch.nn.Module()
    model.structure_head.sample = lambda **kwargs: None
    model.structure_head.diffusion_module = _Denoiser()
    model.attn = layers.SWA3DRoPEAttention(32, 4)
    model.eval().requires_grad_(False)
    originals = (
        model.folding_trunk.forward,
        model.attn.forward,
        model.structure_head.sample,
    )
    options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
    with pytest.raises(RuntimeError, match="injected failure"):
        with acceleration_context(model, options):
            assert model.folding_trunk.forward != originals[0]
            raise RuntimeError("injected failure")
    assert (
        model.folding_trunk.forward,
        model.attn.forward,
        model.structure_head.sample,
    ) == originals
    with acceleration_context(model, options) as execution:
        pass
    assert execution["effective"] == "cached"


def test_mask_cache_budget_does_not_retain_large_masks():
    cache = MaskCache(max_bytes=1)
    x = torch.zeros(1, 8, 2)
    allowed, valid = cache.get((), x, 2)
    assert allowed.shape == (1, 1, 8, 8)
    assert valid.all()
    assert not cache.entries


@pytest.mark.parametrize("batch", [1, 5])
@pytest.mark.parametrize("holes", [False, True])
def test_cached_attention_matches_native_and_handles_new_layout(batch, holes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(17)
    module = layers.SWA3DRoPEAttention(32, 4, half_window=3).to(device).eval()
    x = torch.randn(batch, 19, 32, device=device)
    cos = torch.randn(batch, 19, 4, device=device)
    sin = torch.randn_like(cos)
    cache = MaskCache()
    with torch.inference_mode():
        for shift in (0, 1, 0):
            params = (cos, sin)
            if holes:
                indices = torch.arange(batch * 19, device=device)[shift::2]
                # Keep the native flash path off for a controlled SDPA comparison.
                params += (indices,)
            expected = module(x, params)
            actual = cached_attention(module, x, params, cache)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class _Denoiser(torch.nn.Module):
    def forward(self, x_noisy, t_hat, **kwargs):
        result = x_noisy / (1 + t_hat[:, None, None])
        return {
            "x_denoised": result,
            "token_repr": result,
            "atom_intermediates": result if kwargs["return_atom_repr"] else None,
        }


@pytest.mark.parametrize(
    "samples,steps,cap,noise,atom_repr",
    [(1, 1, None, 0.0, False), (5, 17, 256.0, 1.0, True)],
)
def test_schedule_hoisting_preserves_sampling_outputs_and_rng(
    samples, steps, cap, noise, atom_repr
):
    # Real native sampling, augmentation and alignment with a cheap deterministic
    # denoiser, so the test covers all random draws and the complete schedule.
    head = layers.DiffusionStructureHead.__new__(layers.DiffusionStructureHead)
    torch.nn.Module.__init__(head)
    settings = dict(
        sigma_data=16.0,
        gamma_0=0.605,
        gamma_min=1.107,
        noise_scale=0.0,
        step_scale=1.0,
        inference_s_max=160.0,
        inference_s_min=4e-4,
        inference_p=8.0,
        inference_num_steps=17,
    )
    for name, value in settings.items():
        setattr(head, name, value)
    head.diffusion_module = _Denoiser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    values = dict(
        z_trunk=None,
        s_inputs=torch.ones(1, 4, 3, device=device),
        s_trunk=None,
        relative_position_encoding=None,
        ref_pos=None,
        ref_charge=None,
        ref_mask=torch.ones(1, 12, device=device),
        ref_element=None,
        ref_atom_name_chars=None,
        ref_space_uid=None,
        tok_idx=torch.arange(12, device=device)[None],
        asym_id=None,
        residue_index=None,
        entity_id=None,
        token_index=None,
        sym_id=None,
        num_diffusion_samples=samples,
        num_sampling_steps=steps,
        max_inference_sigma=cap,
        noise_scale=noise,
        return_atom_repr=atom_repr,
    )
    torch.manual_seed(81)
    expected = head.sample(**values)
    rng_expected = (
        torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    )
    torch.manual_seed(81)
    actual = sample_without_scalar_sync(head, **values)
    rng_actual = (
        torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    )
    assert torch.equal(rng_expected, rng_actual)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph integration")
def test_graph_replay_owns_outputs_and_handles_input_changes():
    controller = AcceleratedInference(None)
    module = (
        layers.FoldingTrunk(n_layers=1, d_pair=32).cuda().eval().requires_grad_(False)
    )
    with torch.inference_mode():
        for size in (8, 8, 11, 8):
            pair = torch.randn(1, size, size, 32, device="cuda")
            mask = torch.ones(1, size, size, device="cuda")
            kwargs = dict(pair=pair, pair_attention_mask=mask)
            expected = module(**kwargs)
            output = controller._run(
                "trunk", module, kwargs, ("pair", "pair_attention_mask"), size
            )
            torch.testing.assert_close(output, expected, atol=0, rtol=0)
            preserved = output.clone()
            changed = dict(kwargs, pair=pair + 1)
            controller._run(
                "trunk", module, changed, ("pair", "pair_attention_mask"), size
            )
            assert torch.equal(preserved, output)
        assert controller.stats["trunk_captures"] == 3
        assert controller.stats["trunk_replays"] == 8
    controller.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph integration")
def test_random_forward_falls_back_without_consuming_extra_rng():
    controller = AcceleratedInference(None)

    def stochastic(pair):
        return pair + torch.rand_like(pair)

    with torch.inference_mode():
        pair = torch.zeros(1, 3, device="cuda")
        torch.manual_seed(5)
        expected = stochastic(pair)
        expected_rng = torch.cuda.get_rng_state()
        torch.manual_seed(5)
        actual = controller._run("test", stochastic, dict(pair=pair), ("pair",), 3)
        assert torch.equal(expected, actual)
        assert torch.equal(expected_rng, torch.cuda.get_rng_state())
        assert controller.stats["test_fallbacks"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph lifetime")
def test_repeated_requests_reuse_stream_without_resident_memory_growth():
    model = torch.nn.Module()
    model.folding_trunk = layers.FoldingTrunk(n_layers=1, d_pair=32).cuda()
    model.structure_head = torch.nn.Module()
    model.structure_head.sample = lambda **kwargs: None
    model.structure_head.diffusion_module = _Denoiser()
    model.eval().requires_grad_(False)
    options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
    memory, streams = [], []
    with torch.inference_mode():
        pair = torch.randn(1, 8, 8, 32, device="cuda")
        for _ in range(8):
            with acceleration_context(model, options):
                output = model.folding_trunk(pair)
                assert torch.isfinite(output).all()
                del output
            torch.cuda.synchronize()
            streams.append(model._boltzgen_capture_stream.cuda_stream)
            memory.append(torch.cuda.memory_allocated())
    assert len(set(streams)) == 1
    assert max(memory[1:]) - min(memory[1:]) < 2 * 1024**2, memory
