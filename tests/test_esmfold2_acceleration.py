"""Native component equivalence; run in the isolated ESM runtime.

The CUDA cases exercise replay, ownership, input changes and RNG refusal.
Full checkpoint comparisons live in the public benchmark harness.
"""

from functools import wraps
from contextlib import nullcontext
import linecache
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
    check_source(
        layers.DiffusionStructureHead.sample, _SOURCE_HASHES["sample"], inference_mode=True
    )
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


def test_lambda_with_incomplete_source_preserves_native_forward(monkeypatch):
    native = layers.SWA3DRoPEAttention.forward
    filename = "<esmfold2-lambda-wrapper>"
    source = (
        'setattr(attention_class, "forward",\n'
        '        lambda self, x, params: original(self, x, params))\n'
    )
    monkeypatch.setitem(
        linecache.cache, filename, (len(source), None, source.splitlines(True), filename)
    )
    monkeypatch.setattr(layers.SWA3DRoPEAttention, "forward", native)
    exec(compile(source, filename, "exec"), {
        "attention_class": layers.SWA3DRoPEAttention, "original": native,
    })
    model = torch.nn.Module()
    model.attn = layers.SWA3DRoPEAttention(32, 4).eval().requires_grad_(False)
    x = torch.randn(1, 8, 32)
    params = (torch.randn(1, 8, 4), torch.randn(1, 8, 4))
    options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
    with torch.inference_mode():
        expected = model.attn(x, params)
        with acceleration_context(model, options) as execution:
            actual = model.attn(x, params)
    assert execution["effective"] == "off"
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["attention", "sample"])
@pytest.mark.parametrize("scope", ["class", "instance", "subclass"])
def test_decorated_native_methods_fall_back_without_removing_wrapper(
    monkeypatch, method, scope
):
    cls, name = (
        (layers.SWA3DRoPEAttention, "forward")
        if method == "attention"
        else (layers.DiffusionStructureHead, "sample")
    )
    original = getattr(cls, name)
    sentinel = torch.ones(1)
    calls = []

    @wraps(original)
    def decorated(*args, **kwargs):
        calls.append(1)
        return sentinel

    attention_cls = layers.SWA3DRoPEAttention
    sample_cls = layers.DiffusionStructureHead
    if scope == "subclass":
        subclass = type("CustomImplementation", (cls,), {name: decorated})
        if method == "attention":
            attention_cls = subclass
        else:
            sample_cls = subclass
    model = torch.nn.Module()
    model.attn = attention_cls(32, 4)
    model.folding_trunk = layers.FoldingTrunk(n_layers=1, d_pair=32)
    model.structure_head = sample_cls.__new__(sample_cls)
    torch.nn.Module.__init__(model.structure_head)
    model.structure_head.diffusion_module = _Denoiser()
    model.eval().requires_grad_(False)
    if scope == "class":
        monkeypatch.setattr(cls, name, decorated)
    elif scope == "instance":
        module = model.attn if method == "attention" else model.structure_head
        monkeypatch.setattr(module, name, MethodType(decorated, module))
    options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
    with acceleration_context(model, options) as execution:
        actual = model.attn() if method == "attention" else model.structure_head.sample()
    assert execution["effective"] == "off"
    assert "wrapper" in execution["fallback_reason"]
    assert actual is sentinel
    assert calls == [1]


def test_rebound_attention_method_keeps_its_original_owner(monkeypatch):
    model = torch.nn.Module()
    model.attn = layers.SWA3DRoPEAttention(32, 4)
    model.folding_trunk = layers.FoldingTrunk(n_layers=1, d_pair=32)
    model.structure_head = torch.nn.Module()
    model.structure_head.sample = MethodType(
        layers.DiffusionStructureHead.sample, model.structure_head
    )
    model.structure_head.diffusion_module = _Denoiser()
    other = layers.SWA3DRoPEAttention(32, 4).eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    monkeypatch.setattr(model.attn, "forward", other.forward)
    x = torch.randn(1, 8, 32)
    params = (torch.randn(1, 8, 4), torch.randn(1, 8, 4))
    options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
    with torch.inference_mode():
        expected = model.attn(x, params)
        with acceleration_context(model, options) as execution:
            actual = model.attn(x, params)
    assert execution["effective"] == "off"
    assert model.attn.forward.__self__ is other
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_context_restores_methods_and_releases_state_after_failure():
    # Small real components exercise installation/cleanup without a checkpoint.
    model = torch.nn.Module()
    model.folding_trunk = layers.FoldingTrunk(n_layers=1, d_pair=32)
    model.structure_head = torch.nn.Module()
    model.structure_head.sample = MethodType(
        layers.DiffusionStructureHead.sample, model.structure_head
    )
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
    assert model.structure_head.__dict__["sample"] is originals[2]
    with acceleration_context(model, options) as execution:
        pass
    assert execution["effective"] == "cached"


@pytest.mark.parametrize("request_fails", [False, True])
def test_cleanup_restores_class_method_lookup(monkeypatch, request_fails):
    model = torch.nn.Module()
    model.folding_trunk = layers.FoldingTrunk(n_layers=1, d_pair=32)
    model.structure_head, _, _ = _sampling_case(1, 2, None, 0.0, False)
    model.eval().requires_grad_(False)
    native = layers.FoldingTrunk.forward
    pair = torch.randn(1, 4, 4, 32)

    @wraps(native)
    def instrumented(self, *args, **kwargs):
        return native(self, *args, **kwargs) + 1

    with torch.inference_mode():
        expected = model.folding_trunk(pair)
        monkeypatch.setattr(layers.FoldingTrunk, "forward", instrumented)
        options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
        error = pytest.raises(RuntimeError, match="request failed") if request_fails else nullcontext()
        with error:
            with acceleration_context(model, options):
                torch.testing.assert_close(model.folding_trunk(pair), expected + 1)
                if request_fails:
                    raise RuntimeError("request failed")
        monkeypatch.setattr(layers.FoldingTrunk, "forward", native)
        # Removing temporary class instrumentation must affect the same instance.
        torch.testing.assert_close(model.folding_trunk(pair), expected, atol=0, rtol=0)
    assert "forward" not in model.folding_trunk.__dict__


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


def _sampling_case(samples, steps, cap, noise, atom_repr):
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
    return head, values, device


@pytest.mark.parametrize(
    "samples,steps,cap,noise,atom_repr",
    [(1, 1, None, 0.0, False), (5, 17, 256.0, 1.0, True)],
)
def test_schedule_hoisting_preserves_sampling_outputs_and_rng(
    samples, steps, cap, noise, atom_repr
):
    # Real native sampling, augmentation and alignment with a cheap deterministic
    # denoiser, so the test covers all random draws and the complete schedule.
    head, values, device = _sampling_case(samples, steps, cap, noise, atom_repr)
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


@pytest.mark.parametrize("variant", ["subclass", "replaced_clone"])
def test_custom_inference_context_preserves_sampling_and_rng(monkeypatch, variant):
    class AuditedInference(torch.inference_mode):
        def __enter__(self):
            torch.rand(1)
            return super().__enter__()

    context = AuditedInference() if variant == "subclass" else torch.inference_mode()
    if variant == "replaced_clone":
        def changed_clone(self):
            torch.rand(1)
            return torch.inference_mode(self.mode)

        context.clone = MethodType(changed_clone, context)
    native = layers.DiffusionStructureHead.sample
    monkeypatch.setattr(
        layers.DiffusionStructureHead, "sample", context(native.__wrapped__)
    )
    head, values, device = _sampling_case(1, 2, None, 0.0, False)
    model = torch.nn.Module()
    model.structure_head = head
    model.folding_trunk = layers.FoldingTrunk(n_layers=1, d_pair=32)
    model.eval().requires_grad_(False)
    options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
    torch.manual_seed(81)
    expected = head.sample(**values)
    expected_cpu_rng = torch.get_rng_state()
    expected_cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
    torch.manual_seed(81)
    with acceleration_context(model, options) as execution:
        actual = head.sample(**values)
    assert execution["effective"] == "off"
    assert "inference wrapper" in execution["fallback_reason"]
    assert torch.equal(expected_cpu_rng, torch.get_rng_state())
    if device == "cuda":
        assert torch.equal(expected_cuda_rng, torch.cuda.get_rng_state())
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)


@pytest.mark.parametrize(
    "variant",
    ["sample", "attention", "factory", "class_sample", "attention_function", "sample_body"],
)
def test_callable_proxies_preserve_custom_execution(monkeypatch, variant):
    calls = []

    class MethodProxy:
        def __init__(self, method):
            self.method = method

        def __getattr__(self, name):
            return getattr(self.method, name)

        @property
        def __class__(self):
            return self.method.__class__

        def __get__(self, instance, owner=None):
            return self if instance is None else MethodType(self, instance)

        def __call__(self, *args, **kwargs):
            calls.append(1)
            torch.rand(1)
            return self.method(*args, **kwargs)

    head, values, device = _sampling_case(1, 2, None, 0.0, False)
    model = torch.nn.Module()
    model.structure_head = head
    model.folding_trunk = layers.FoldingTrunk(n_layers=1, d_pair=32)
    model.attn = layers.SWA3DRoPEAttention(32, 4)
    model.eval().requires_grad_(False)
    if variant in ("sample", "class_sample"):
        monkeypatch.setattr(head, "sample", MethodProxy(head.sample))
        if variant == "class_sample":
            monkeypatch.setattr(
                layers.DiffusionStructureHead, "sample",
                MethodProxy(layers.DiffusionStructureHead.sample),
            )
    elif variant == "attention":
        monkeypatch.setattr(model.attn, "forward", MethodProxy(model.attn.forward))
    elif variant == "attention_function":
        monkeypatch.setattr(
            layers.SWA3DRoPEAttention, "forward",
            MethodProxy(layers.SWA3DRoPEAttention.forward),
        )
    elif variant == "sample_body":
        monkeypatch.setattr(
            layers.DiffusionStructureHead, "sample",
            torch.inference_mode()(MethodProxy(layers.DiffusionStructureHead.sample.__wrapped__)),
        )
    else:
        context = torch.inference_mode()
        context.clone = MethodProxy(context.clone)
        monkeypatch.setattr(
            layers.DiffusionStructureHead, "sample",
            context(layers.DiffusionStructureHead.sample.__wrapped__),
        )
    x = torch.randn(1, 8, 32)
    params = (torch.randn(1, 8, 4), torch.randn(1, 8, 4))
    options = dict(acceleration="auto", acceleration_revision=ACCELERATION_REVISION)
    with torch.inference_mode():
        torch.manual_seed(81)
        expected_attention = model.attn(x, params)
        expected = head.sample(**values)
        expected_cpu_rng = torch.get_rng_state()
        expected_cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
        assert calls == [1]
        calls.clear()
        torch.manual_seed(81)
        with acceleration_context(model, options) as execution:
            actual_attention = model.attn(x, params)
            actual = head.sample(**values)
    assert execution["effective"] == "off"
    assert calls == [1]
    assert torch.equal(expected_cpu_rng, torch.get_rng_state())
    if device == "cuda":
        assert torch.equal(expected_cuda_rng, torch.cuda.get_rng_state())
    torch.testing.assert_close(actual_attention, expected_attention, atol=0, rtol=0)
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
    model.structure_head.sample = MethodType(
        layers.DiffusionStructureHead.sample, model.structure_head
    )
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
