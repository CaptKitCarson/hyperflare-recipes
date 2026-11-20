# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility re-export of the Qwen3.8 checkpoint schema owned by ignition.recipes.qwen38."""

from ignition.recipes.qwen38.checkpoint_schema import *  # noqa: F401,F403
from ignition.recipes.qwen38.checkpoint_schema import _hyperconnection, _linear, _mlp, _sparse_attention  # noqa: F401
