# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepared grouped low-rank attention output projection through b12x."""

import torch
from b12x.gemm import wo_projection
from b12x.preparation import PreparedCall

from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer


class PreparedGroupedOutputProjection:
    def __init__(self, source, weights, output, *, session, name, workspace,
                 inv_rope=False, heads_per_group=None, nope_dim=448, rope_dim=64,
                 positions=None, cos_sin_cache=None):
        self.weights, self.output, self.workspace = weights, output, workspace
        self.inv_rope = bool(inv_rope)
        if self.inv_rope:
            if (heads_per_group is None or positions is None or cos_sin_cache is None
                    or source.ndim != 3
                    or source.shape[1:] != (weights.groups * heads_per_group, nope_dim + rope_dim)
                    or heads_per_group * (nope_dim + rope_dim) != weights.group_width):
                raise ValueError("inverse-RoPE attention output geometry does not match packed WO weights")
            invocation = {
                "operation": "inv_rope",
                "heads_per_group": int(heads_per_group),
                "nope_dim": int(nope_dim),
                "rope_dim": int(rope_dim),
                "positions_dtype": str(positions.dtype).removeprefix("torch."),
                "cos_sin_dtype": str(cos_sin_cache.dtype).removeprefix("torch."),
            }
            self.positions = positions
            self.cos_sin_cache = cos_sin_cache
            self.heads_per_group = int(heads_per_group)
            self.nope_dim = int(nope_dim)
            self.rope_dim = int(rope_dim)
        else:
            if source.shape[1:] != (weights.groups, weights.group_width):
                raise ValueError("attention output must match the packed projection groups and head width")
            invocation = {}
        self.plan = wo_projection.plan(
            wo_projection.Caps(device=source.device, max_tokens=source.shape[0],
                groups=weights.groups, group_width=weights.group_width, rank=weights.rank, hidden=weights.hidden),
            invocation=invocation,
        )
        self.operands = [tensor for packed in (weights.wo_a, weights.wo_b)
                         for tensor in (packed.values, packed.scale_rows.view(torch.uint8),
                                        packed.scale_mma.view(torch.uint8))]
        for tensor in self.operands:
            record_numeric_tensor(tensor)

        if self.inv_rope:
            def call(state, values, pos, cs, out, *, produce=None, owners=()):
                scratch = planned_scratch(state._scratch_state.scratch_specs(), workspace)
                binding = state.bind_inv_rope(o=values, positions=pos, cos_sin_cache=cs,
                    weights=weights, heads_per_group=self.heads_per_group,
                    nope_dim=self.nope_dim, rope_dim=self.rope_dim, scratch=scratch, out=out)
                return PreparedCall(run=lambda: state.run_inv_rope(binding), output=out, produce=produce,
                    owners=(*owners, weights, *prime_owners(workspace, scratch, binding)))

            def benchmark(state):
                generator = torch.Generator(device=source.device).manual_seed(4121)
                seed = torch.randn(source.shape, dtype=source.dtype, device=source.device, generator=generator) * .125
                values, out = torch.empty_like(source), torch.empty_like(output)
                pos = torch.arange(source.shape[0], dtype=positions.dtype, device=source.device)
                cs = torch.zeros(cos_sin_cache.shape, dtype=cos_sin_cache.dtype, device=source.device)
                cs[:, :self.rope_dim // 2].fill_(1.0)
                return call(state, values, pos, cs, out, produce=lambda: values.copy_(seed),
                    owners=(values, out, seed, pos, cs))

            declare(session, (self.plan.request(name=name,
                prepare_call=lambda state: call(state, source, positions, cos_sin_cache, output),
                benchmark_call=benchmark),))
        else:
            def call(state, values, out, *, produce=None, owners=()):
                scratch = planned_scratch(state._scratch_state.scratch_specs(), workspace)
                binding = state.bind(source_tgd=values, weights=weights, scratch=scratch, out=out)
                return PreparedCall(run=lambda: state.run(binding), output=out, produce=produce,
                    owners=(*owners, weights, *prime_owners(workspace, scratch, binding)))

            def benchmark(state):
                generator = torch.Generator(device=source.device).manual_seed(4121)
                seed = torch.randn(source.shape, dtype=source.dtype, device=source.device, generator=generator) * .125
                values, out = torch.empty_like(source), torch.empty_like(output)
                return call(state, values, out, produce=lambda: values.copy_(seed), owners=(values, out, seed))

            declare(session, (self.plan.request(name=name,
                prepare_call=lambda state: call(state, source, output), benchmark_call=benchmark),))
        register_finalizer(session, self.rebind, workspace=workspace)

    def rebind(self):
        self.scratch = planned_scratch(self.plan.scratch_specs(), self.workspace)

    def functional(self, *, session):
        self.output_layout = (tuple(self.output.shape), tuple(self.output.stride()), self.output.dtype)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.output = self.workspace = self.scratch = None
        if self.inv_rope:
            self.positions = self.cos_sin_cache = None

    def __call__(self, source, *, positions=None, cos_sin_cache=None):
        output, scratch = self.output, self.scratch
        if output is None:
            shape, stride, dtype = self.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=source.device)
            scratch, = (torch.empty(shape, dtype=dtype, device=source.device) for shape, dtype in self.scratch_layout)
        if self.inv_rope:
            pos = self.positions if positions is None else positions
            cs = self.cos_sin_cache if cos_sin_cache is None else cos_sin_cache
            wo_projection.run_inv_rope_out(source, pos, cs, self.weights, out=output, scratch=scratch, plan=self.plan)
        else:
            wo_projection.run_out(source, self.weights, out=output, scratch=scratch, plan=self.plan)
        return output
