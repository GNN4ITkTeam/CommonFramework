# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IGNN3 — interaction network for directed graphs, built for C++ deployment.

The network itself lives in :class:`IGNN3Net`, a plain ``nn.Module``.
:class:`IGNN3` is the Lightning wrapper that owns it as ``self.net`` and adds
training, evaluation and export.  That split is deliberate: the artifact shipped
to production (TorchScript, ONNX or ``torch.export``) is literally
``IGNN3Net.forward``, the same code path evaluated in Python, so the two cannot
drift apart.

Aggregation uses native ``Tensor.scatter_add_`` rather than ``torch_scatter``,
because ONNX and ``torch.export`` cannot represent custom ops.  One consequence
is that there is a single aggregation implementation for training, Python
inference and every export backend.
"""

import os
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from acorn.utils import make_mlp
from ..edge_classifier_stage import EdgeClassifierStage


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


#: Accepted values for the config keys that select a combination mode.
_VALID_MODES = {
    "src_dst_aggr": ("concat", "inner_product_and_diff", "inner_divide_and_diff"),
    "edge_attr": ("concat", "inner_product_and_diff", "inner_divide_and_diff"),
    "aggr": ("sum", "mean"),
}

#: Keys that name an attribute of the model rather than a config key.  They were
#: accepted by YAML and ignored by the code, so a run configured with one of them
#: silently used the default instead.
_LEGACY_KEY_ALIASES = {
    "src_dst_aggr_mode": "src_dst_aggr",
    "edge_attr_mode": "edge_attr",
    "aggr_mode": "aggr",
}


def _resolve_mode(hparams, key: str) -> str:
    """Read a mode-selecting config key, rejecting misspelled keys and values.

    Two failure modes used to pass unnoticed, both producing a run that silently
    used ``concat`` / ``sum`` while its config and output paths claimed otherwise:

    * the key was spelled like the internal attribute (``src_dst_aggr_mode``),
      so ``hparams.get("src_dst_aggr")`` never saw it;
    * the value was misspelled, so every branch fell through to the default.

    A legacy key raises only when it is the sole spelling present, i.e. when it
    expresses an intent that is not being honoured.  When the canonical key is
    also set — as when a fixed config is layered over the hyperparameters stored
    in an old checkpoint — the canonical key wins and the stale one only warns,
    which keeps such checkpoints loadable.
    """
    default = _VALID_MODES[key][0]
    legacy = next(
        (k for k, canonical in _LEGACY_KEY_ALIASES.items() if canonical == key), None
    )

    if legacy is not None and legacy in hparams:
        if key in hparams:
            warnings.warn(
                f"Config key '{legacy}' is not read and is ignored; "
                f"'{key}: {hparams[key]}' applies."
            )
        else:
            raise ValueError(
                f"Unknown config key '{legacy}' — did you mean '{key}'? "
                f"Earlier versions ignored '{legacy}' without warning, so a run "
                f"configured this way actually used '{key}: {default}'. Set "
                f"'{key}: {default}' to reproduce that run, or "
                f"'{key}: {hparams[legacy]}' to apply what the config asked for."
            )

    value = hparams.get(key, default)
    if value not in _VALID_MODES[key]:
        raise ValueError(
            f"Invalid value '{value}' for config key '{key}'. "
            f"Expected one of: {', '.join(_VALID_MODES[key])}."
        )
    return value


class IGNN3Net(nn.Module):
    """Interaction network for directed graphs.

    Incoming and outgoing edge latents are aggregated and reduced separately
    (``in_out_diff_agg``), over ``n_graph_iters`` message passing iterations.

    ``forward`` is the only entry point.  Gradient checkpointing is a
    training-only concern and lives behind a branch TorchScript folds away, so
    the scripted and exported graph is exactly the plain path.
    """

    def __init__(self, hparams):
        super().__init__()
        # Scalar typed attributes — TorchScript requires consistent types, so
        # every hyperparameter used inside scripted methods is stored as one.
        self.checkpointing: bool = hparams.get("checkpointing", False)
        self.ckpting_reentrant: bool = hparams.get("ckpting_reentrant", False)
        if "n_graphs_iters_inference" in hparams:
            raise ValueError(
                "Config key 'n_graphs_iters_inference' has been removed: "
                "'n_graph_iters' now applies to training and inference alike. "
                "Drop the key from the config."
            )
        self.n_graph_iters: int = hparams.get("n_graph_iters", 8)
        self.in_out_diff_agg: bool = hparams.get("in_out_diff_agg", True)
        self.use_sum_aggr: bool = _resolve_mode(hparams, "aggr") == "sum"
        # str attributes used in scripted methods (src_dst_aggr, edge_attr_fn).
        # Validated here rather than in those methods: TorchScript requires every
        # branch to return, so they keep a fall-through default.
        self.src_dst_aggr_mode: str = _resolve_mode(hparams, "src_dst_aggr")
        self.edge_attr_mode: str = _resolve_mode(hparams, "edge_attr")

        hidden = hparams["hidden"]
        # src_dst_aggr always concatenates three hidden-sized blocks, and
        # edge_attr_fn two, whatever the mode.
        self.in_edge_net: int = hidden * 3
        self.in_node_net: int = hidden * 3 if self.in_out_diff_agg else hidden * 2

        self.node_encoder = make_mlp(
            input_size=len(hparams["node_features"]),
            sizes=[hidden] * hparams["n_node_encoder_layers"],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["output_activation"],
            layer_norm=hparams["layernorm"],
            output_layer_norm=hparams["output_layer_norm"],
        )

        edge_features = hparams.get("edge_features") or []
        if len(edge_features) != 0:
            # Edge features come from the batch.
            self.edge_encoder = make_mlp(
                input_size=len(edge_features),
                sizes=[hidden] * hparams["n_edge_encoder_layers"],
                hidden_activation=hparams["hidden_activation"],
                output_activation=hparams["output_activation"],
                layer_norm=hparams["layernorm"],
                output_layer_norm=hparams["output_layer_norm"],
            )
        else:
            # No edge features: they are built from the two node latents.
            self.edge_encoder = make_mlp(
                input_size=2 * hidden,
                sizes=[hidden] * hparams["n_edge_net_layers"],
                hidden_activation=hparams["hidden_activation"],
                output_activation=hparams["output_activation"],
                layer_norm=hparams["layernorm"],
                output_layer_norm=hparams["output_layer_norm"],
            )

        self.edge_network = nn.ModuleList(
            [
                make_mlp(
                    input_size=self.in_edge_net,
                    sizes=[hidden] * hparams["n_edge_net_layers"],
                    hidden_activation=hparams["hidden_activation"],
                    output_activation=hparams["output_activation"],
                    layer_norm=hparams["layernorm"],
                    output_layer_norm=hparams["output_layer_norm"],
                )
                for _ in range(self.n_graph_iters)
            ]
        )

        self.node_network = nn.ModuleList(
            [
                make_mlp(
                    input_size=self.in_node_net,
                    sizes=[hidden] * hparams["n_node_net_layers"],
                    hidden_activation=hparams["hidden_activation"],
                    output_activation=hparams["output_activation"],
                    layer_norm=hparams["layernorm"],
                    output_layer_norm=hparams["output_layer_norm"],
                )
                for _ in range(self.n_graph_iters)
            ]
        )

        self.output_edge_classifier = make_mlp(
            input_size=hidden,
            sizes=[hidden] * hparams["n_edge_decoder_layers"] + [1],
            hidden_activation=hparams["hidden_activation"],
            output_activation=hparams["edge_output_transform_final_activation"],
            layer_norm=hparams["layernorm"],
            output_layer_norm=hparams["edge_output_transform_final_layer_norm"],
        )

    # ── feature combination ──────────────────────────────────────────────── #

    def src_dst_aggr(
        self, e: torch.Tensor, xi: torch.Tensor, xj: torch.Tensor
    ) -> torch.Tensor:
        """Build the edge network input from an edge latent and its two nodes."""
        if self.src_dst_aggr_mode == "inner_product_and_diff":
            return torch.cat([e, xi * xj, torch.abs(xi - xj)], dim=-1)
        elif self.src_dst_aggr_mode == "inner_divide_and_diff":
            return torch.cat([e, xi / (xj + 1e-8), torch.abs(xi - xj)], dim=-1)
        else:
            # "concat", and the default for an unknown mode. TorchScript needs
            # every path to return.
            return torch.cat([e, xi, xj], dim=-1)

    def edge_attr_fn(self, xi: torch.Tensor, xj: torch.Tensor) -> torch.Tensor:
        """Build initial edge features from the two node latents.

        Only used when the config declares no ``edge_features``.
        """
        if self.edge_attr_mode == "inner_product_and_diff":
            return torch.cat([xi * xj, torch.abs(xi - xj)], dim=-1)
        elif self.edge_attr_mode == "inner_divide_and_diff":
            return torch.cat([xi / (xj + 1e-8), torch.abs(xi - xj)], dim=-1)
        else:
            return torch.cat([xi, xj], dim=-1)

    # ── aggregation (native scatter, ONNX / torch.export compatible) ─────── #

    def _scatter_add(
        self, src: torch.Tensor, index: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        out = torch.zeros(
            dim_size, src.shape[1], dtype=src.dtype, device=src.device
        )
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
        return out

    def _scatter_mean(
        self, src: torch.Tensor, index: torch.Tensor, dim_size: int
    ) -> torch.Tensor:
        total = self._scatter_add(src, index, dim_size)
        ones = torch.ones(src.shape[0], 1, dtype=src.dtype, device=src.device)
        count = self._scatter_add(ones, index, dim_size).clamp(min=1.0)
        return total / count

    def node_inputs(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate edge latents onto nodes and build the node network input.

        The single implementation shared by training, Python inference and
        every export backend.
        """
        num_nodes = x.shape[0]
        if self.use_sum_aggr:
            from_src = self._scatter_add(e, dst, num_nodes)
            from_dst = self._scatter_add(e, src, num_nodes)
        else:
            from_src = self._scatter_mean(e, dst, num_nodes)
            from_dst = self._scatter_mean(e, src, num_nodes)

        if self.in_out_diff_agg:
            return torch.cat([from_src, from_dst, x], dim=-1)
        return torch.cat([from_src + from_dst, x], dim=-1)

    def encode(
        self,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode node and edge features into their latent spaces."""
        x = self.node_encoder(x)
        if edge_attr is not None:
            e = self.edge_encoder(edge_attr)
        else:
            e = self.edge_encoder(self.edge_attr_fn(x[src], x[dst]))
        return x, e

    # ── the one entry point ──────────────────────────────────────────────── #

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one logit per edge.

        Gradient checkpointing trades memory for recomputation during the
        backward pass, so it is meaningless without one.  It therefore lives in
        a separate, training-only path: ``torch.jit.is_scripting()`` is folded
        to ``True`` at scripting time, which removes the branch from the graph
        entirely — no ``prim::If``, no reference to the checkpointed helper — so
        what is scripted and exported is exactly the plain path below.
        """
        if not torch.jit.is_scripting() and self.training and self.checkpointing:
            return self._forward_checkpointed(x, edge_index, edge_attr)

        src = edge_index[0]
        dst = edge_index[1]
        x, e = self.encode(x, src, dst, edge_attr)
        # zip over the ModuleLists so TorchScript never sees ModuleList[variable]:
        # only integer-literal subscripts are supported.
        for node_net, edge_net in zip(self.node_network, self.edge_network):
            e = edge_net(self.src_dst_aggr(e, x[src], x[dst]))
            x = node_net(self.node_inputs(x, e, src, dst))

        return self.output_edge_classifier(e).squeeze(-1)

    # ── gradient checkpointing (training, eager only) ────────────────────── #

    @torch.jit.unused
    def _message_step(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        i: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One message passing iteration, indexed — checkpointable.

        Variable ``ModuleList`` indexing is fine in eager mode; this method is
        never scripted, which is what lets the training loop be checkpointed.
        """
        e = self.edge_network[i](self.src_dst_aggr(e, x[src], x[dst]))
        x = self.node_network[i](self.node_inputs(x, e, src, dst))
        return x, e

    @torch.jit.unused
    def _forward_checkpointed(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``forward`` with each step wrapped in ``torch.utils.checkpoint``.

        Carries no maths of its own: every tensor operation comes from
        ``encode`` and ``_message_step``, the same ones ``forward`` calls.  The
        inputs are marked as requiring gradients because the reentrant
        implementation needs at least one input that does.
        """
        src = edge_index[0]
        dst = edge_index[1]
        x = x.requires_grad_(True)
        if edge_attr is not None:
            edge_attr = edge_attr.requires_grad_(True)

        x, e = checkpoint(
            self.encode, x, src, dst, edge_attr,
            use_reentrant=self.ckpting_reentrant,
        )
        for i in range(self.n_graph_iters):
            x, e = checkpoint(
                self._message_step, x, e, src, dst, i,
                use_reentrant=self.ckpting_reentrant,
            )

        return self.output_edge_classifier(e).squeeze(-1)


# ---------------------------------------------------------------------------
# The Lightning stage
# ---------------------------------------------------------------------------


class IGNN3(EdgeClassifierStage):
    """Lightning wrapper around :class:`IGNN3Net`.

    Owns the network as ``self.net`` and adds training, evaluation and the
    export helpers driven by ``ModelExportCallback``.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        # Downstream code (``_forward_from_batch``, the export helpers) reads
        # this key unconditionally.
        if "edge_features" not in self.hparams:
            self.hparams["edge_features"] = []
        self.net = IGNN3Net(self.hparams)
        self._register_load_state_dict_pre_hook(self._remap_legacy_state_dict)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.net(x, edge_index, edge_attr)

    def configure_model(self):
        """Compile the network, as the Lightning hook of the same name.

        ``LightningModule.configure_model`` is empty by default; the Trainer calls
        it for each of the fit/validate/test/predict stages, after the module is on
        its device and the strategy is set up, and ``load_from_checkpoint`` calls it
        too.  It is the strategy- and precision-aware place to transform the
        network, which ``__init__`` cannot be.  Lightning requires it to be
        idempotent, hence the early return once the network is already compiled.

        Note that only the network is compiled, never the Lightning module: it
        would pull ``training_step``, the metrics and ``self.log`` into the graph,
        where dynamo breaks for no gain.  Wrapping the network covers both the
        plain and the checkpointed path, since both go through its ``forward``.

        The whole network is compiled as one graph rather than each MLP separately.
        Aggregation uses native ``scatter_add_``, so dynamo traces the entire
        forward with zero graph breaks — measured on an A100 at 1.57x eager,
        against 1.28x per-MLP.  Compiling the root module leaves ``node_network`` /
        ``edge_network`` as plain ``ModuleList`` objects, so
        ``zip(node_network, edge_network)`` keeps working; it is compiling a
        ``ModuleList`` itself that would turn it into a non-iterable
        ``OptimizedModule``.

        ``dynamic=True`` is required, not merely preferable: node and edge
        counts differ for every event, and the static default recompiles on each
        new size, which costs far more than compilation gains.  It can be
        overridden through ``compile_kwargs``.

        Note: ``torch.compile`` is incompatible with non-reentrant
        ``torch.utils.checkpoint`` (``use_reentrant=False``).  When
        ``checkpointing=True`` and ``ckpting_reentrant=False``, compilation is
        skipped to avoid a ``FakeTensorMode`` / RNG-state clash during dynamo
        tracing.  Set ``ckpting_reentrant: true`` in the config to use both.

        With checkpointing, dynamo traces ``torch.utils.checkpoint`` as a higher
        order op, which its DDP optimizer cannot split into gradient buckets.
        Acorn always trains under ``DDPStrategy``, even on one GPU, so the
        splitting is turned off (``torch._dynamo.config.optimize_ddp``).  The
        network's parameters fit in one or two buckets anyway, so the lost
        overlap of communication and backward is negligible.  The setting is
        process-wide.
        """
        if not self.hparams.get("compile", False):
            return
        if hasattr(self.net, "_orig_mod"):
            return  # already compiled: the hook runs once per Trainer stage
        if self.net.checkpointing and not self.net.ckpting_reentrant:
            warnings.warn(
                "torch.compile is incompatible with non-reentrant gradient "
                "checkpointing (use_reentrant=False). Skipping compilation. "
                "Set ckpting_reentrant: true to enable both."
            )
            return
        if self.net.checkpointing:
            torch._dynamo.config.optimize_ddp = False
        compile_kwargs = {"dynamic": True, **(self.hparams.get("compile_kwargs") or {})}
        self.net = torch.compile(self.net, **compile_kwargs)

    # Submodules that used to live directly on the Lightning module, before the
    # network was extracted into ``self.net``.
    _LEGACY_NET_PREFIXES = (
        "node_encoder.",
        "edge_encoder.",
        "node_network.",
        "edge_network.",
        "output_edge_classifier.",
    )

    def _remap_legacy_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Normalise checkpoints written under an older layout.

        Two rewrites, both needed to make checkpoints interchangeable whatever
        the run that produced them:

        * ``_orig_mod.`` is normalised away.  ``torch.compile`` inserts it, in a
          different place depending on what was compiled —
          ``node_encoder._orig_mod.0.weight`` when each MLP was compiled
          separately, ``net._orig_mod.node_encoder.0.weight`` now that the whole
          network is — so it is stripped wherever it appears, then re-added only
          if *this* module is itself compiled.  That makes checkpoints load in
          both directions, compiled to plain and plain to compiled.
        * flat keys (``node_encoder.0.weight``) are moved under ``net.``, the
          layout since the network was extracted into :class:`IGNN3Net`.  Only
          the network's own submodules move: sibling keys such as the
          ``torchmetrics`` states created by ``EdgeClassifierStage`` stay put.

        Registered as a ``load_state_dict`` pre-hook so it applies to every load
        path — ``load_from_checkpoint``, a bare ``load_state_dict``, or a manual
        ``torch.load`` — and not just to Lightning's checkpoint hook.
        """
        # An OptimizedModule holds the real network under ``_orig_mod``.
        net_prefix = "net._orig_mod." if hasattr(self.net, "_orig_mod") else "net."
        for key in [k for k in state_dict if k.startswith(prefix)]:
            tail = key[len(prefix) :]
            new_tail = tail.replace("_orig_mod.", "")
            if new_tail.startswith(self._LEGACY_NET_PREFIXES):
                new_tail = f"{net_prefix}{new_tail}"
            elif new_tail.startswith("net."):
                new_tail = f"{net_prefix}{new_tail[len('net.') :]}"
            if new_tail != tail:
                state_dict[f"{prefix}{new_tail}"] = state_dict.pop(key)

    # ------------------------------------------------------------------ #
    #  Export helpers (torch.jit / torch.onnx / torch.export)            #
    # ------------------------------------------------------------------ #

    def _get_export_model(self) -> IGNN3Net:
        """Return a clean (non-compiled), eval-mode copy of the network.

        ``torch.compile`` wraps the network in an ``OptimizedModule``, which is
        not scriptable, and prefixes its state dict keys with ``_orig_mod.``.
        Building a fresh :class:`IGNN3Net` and loading the stripped weights
        guarantees the exported model is clean whatever the training config was.

        The prefix is removed wherever it occurs rather than only in the middle
        of a key: compiling the whole network puts it in front
        (``_orig_mod.node_encoder.0.weight``), compiling each MLP separately put
        it mid-key (``node_encoder._orig_mod.0.weight``), and checkpoints from
        both layouts must load.
        """
        dev = next(self.parameters()).device
        export_model = IGNN3Net(self.hparams)
        state = {
            k.replace("_orig_mod.", ""): v
            for k, v in self.net.state_dict().items()
        }
        export_model.load_state_dict(state)
        return export_model.to(dev).eval()

    def _make_export_example_inputs(self, export_cfg: dict) -> tuple:
        """Build dummy input tensors matching the model's feature dimensions."""
        n_nodes = export_cfg.get("example_num_nodes", 10_000)
        n_edges = export_cfg.get("example_num_edges", 100_000)
        x = torch.randn(n_nodes, len(self.hparams["node_features"]))
        edge_index = torch.randint(0, n_nodes, (2, n_edges))
        edge_features = self.hparams.get("edge_features") or []
        if edge_features:
            return (x, edge_index, torch.randn(n_edges, len(edge_features)))
        return (x, edge_index)

    def _save_jit(self, model: IGNN3Net, example_inputs: tuple, save_dir: str):
        """Save the network as TorchScript via ``torch.jit.script``.

        ``script`` compiles the actual Python control flow rather than tracing
        one execution path, so no tracing warnings are emitted and the
        training-only branch is folded away rather than baked in.
        """
        import json

        path = os.path.join(save_dir, "model.jit.pt")
        scripted = torch.jit.script(model.eval())
        hparams_bytes = json.dumps(dict(self.hparams), indent=2, default=str)
        torch.jit.save(scripted, path, _extra_files={"hparams.json": hparams_bytes})
        print(f"[export] Saved torch.jit.script model (with embedded hparams) → {path}")

    def _save_onnx(self, model: IGNN3Net, example_inputs: tuple, save_dir: str):
        """Save the network as ONNX via ``torch.onnx.export``."""
        import json

        import onnx

        path = os.path.join(save_dir, "model.onnx")
        dev = next(model.parameters()).device
        example_inputs = tuple(t.to(dev) for t in example_inputs)
        with torch.no_grad():
            torch.onnx.export(
                model.eval(),
                example_inputs,
                path,
                input_names=["x", "edge_index", "edge_attr"],
                output_names=["scores"],
                dynamic_axes={
                    "x": {0: "num_nodes"},
                    "edge_index": {1: "num_edges"},
                    "edge_attr": {0: "num_edges"},
                    "scores": {0: "num_edges"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
        # Embed hparams as ONNX metadata for downstream tools
        model_proto = onnx.load(path)
        meta = model_proto.metadata_props.add()
        meta.key = "hparams"
        meta.value = json.dumps(dict(self.hparams), default=str)
        onnx.save(model_proto, path)
        print(f"[export] Saved ONNX model (with embedded hparams) → {path}")

    def _save_torch_export(
        self,
        model: IGNN3Net,
        example_inputs: tuple,
        save_dir: str,
        export_cfg: dict,
    ):
        """Save the network via ``torch.export.export`` (.pt2 format).

        Set ``dynamic_shapes: true`` in the export config (default) to register
        node and edge counts as dynamic dimensions so the exported program
        accepts variable-size graphs.
        """
        dynamic_shapes = None
        if export_cfg.get("dynamic_shapes", True):
            node_dim = torch.export.Dim("num_nodes")
            edge_dim = torch.export.Dim("num_edges")
            # Shapes are listed in the same order as example_inputs
            if len(example_inputs) == 3:
                # (x, edge_index, edge_attr)
                dynamic_shapes = (
                    {0: node_dim},  # x:          [N, node_feat]
                    {1: edge_dim},  # edge_index: [2, E]
                    {0: edge_dim},  # edge_attr:  [E, edge_feat]
                )
            else:
                dynamic_shapes = ({0: node_dim}, {1: edge_dim})

        path = os.path.join(save_dir, "model.export.pt2")
        dev = next(model.parameters()).device
        example_inputs = tuple(t.to(dev) for t in example_inputs)
        with torch.no_grad():
            exported = torch.export.export(
                model.eval(), example_inputs, dynamic_shapes=dynamic_shapes
            )
        torch.export.save(exported, path)
        print(f"[export] Saved torch.export model → {path}")


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
# The model was introduced as ``InteractionGNNNew``.  Configs and checkpoints
# written under that name keep working: ``str_to_class`` resolves ``model:
# InteractionGNNNew`` through this alias.  Prefer ``IGNN3`` in new configs.
InteractionGNNNew = IGNN3
