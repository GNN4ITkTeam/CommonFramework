# Copyright (C) 2026 CERN for the benefit of the ATLAS collaboration

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

from functools import partial

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from pytorch_lightning.callbacks import Callback, ModelPruning
import warnings

import weakref

# Custom linear layer, excluding pruned elements from normalization
class LinkedMaskLayerNorm(nn.Module):
    def __init__(self, parent_module, target_layer_name, layer_norm):
        super().__init__()
        self.parent = weakref.ref(parent_module) if parent_module is not None else None
        self.target_layer_name = target_layer_name # Name of the linear layer to be normalized
        self.eps = layer_norm.eps

    def target_layer(self):
        parent = self.parent()
        # Fetch the target layer from the parent
        return parent.get_submodule(self.target_layer_name)

    def forward(self, x):
        target_layer = self.target_layer()
        # Dynamically retrieve the mask from the associated Linear layer
        if hasattr(target_layer, 'weight_mask'):
            mask = (target_layer.weight_mask.abs().sum(dim=1) > 0).float()
        else:
            # If no mask exists yet, act as a identity mask (all ones)
            mask = (target_layer.weight.abs().sum(dim=1) > 1e-9).float()

        # Apply the mask
        x = x * mask

        # Calculate statistics only across non-zero elements
        n_active = mask.sum()
        if n_active == 0:
            return x 

        # Calculate normalization parameters
        mean = x.sum(dim=-1, keepdim=True) / n_active
        var = (((x - mean) ** 2) * mask).sum(dim=-1, keepdim=True) / n_active
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        return x_norm * mask


# Custom class allowing the bias pruning
class StructuredModelPruning(ModelPruning):
    def __init__(self, restart_pruning=False, *args, **kwargs):
        super().__init__(*args,**kwargs)

        self.restart_pruning = restart_pruning


    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Load the pruning only after the model is loaded from checkpoint
        if self.restart_pruning:
            for name, module in pl_module.named_modules():
                if isinstance(module, nn.Linear):
                    self.reinitialize_pruning_from_current_weights(module)
            
            print(f"Restarting pruning, pruned so far {self.get_pruning_percentage()}")


    def reinitialize_pruning_from_current_weights(self, module, param_name="weight"):
        # Create binary mask from current weights
        weight = getattr(module, param_name)
        mask_w = (weight != 0).to(weight.dtype)
        mask_b = (mask_w.sum(dim=1) > 0).float()

        # Pruned was not performed before
        if (mask_w == 0).sum().item() <= 0:
            return
        
        # Apply the mask to reinitialize pruning structure
        prune.custom_from_mask(module, name="weight", mask=mask_w)
        prune.custom_from_mask(module, name="bias", mask=mask_b)


    def apply_pruning(self, amount):
        # Perform the weight pruning
        super().apply_pruning(amount)

        # Mask corresponding biases
        for module, name in self._parameters_to_prune:
            if hasattr(module, "weight_mask") and module.bias is not None:
                # Prepare a mask
                weight_mask = module.weight_mask
                bias_mask = (weight_mask.sum(dim=1) > 0).float()

                prune.custom_from_mask(module, name="bias", mask=bias_mask)

        print(f"Model pruned {self.get_pruning_percentage()}%")


    # Custom apply local pruning function, using the passed importance scores method
    def _apply_local_pruning(self, amount: float) -> None:
        for module, name in self._parameters_to_prune:
            importance_scores = module.grad_importance_score if hasattr(module, 'grad_importance_score') else None
            self.pruning_fn(module, name=name, amount=amount, importance_scores=importance_scores)


    def get_pruning_percentage(self):
        all_elements=0
        pruned_elems = 0
        for module, name in self._parameters_to_prune:
            if hasattr(module, "weight_mask"):
                weight_mask = module.weight_mask.detach()
                all_elements += weight_mask.numel()
                pruned_elems += (weight_mask == 0).sum().item()
        
        pruning_frac =  pruned_elems/all_elements if all_elements>0 else 0
    
        return round(pruning_frac*100, 2)


# Helper class to analyse impact of pruning on model's accuracy
class PruningPrintingCallback(Callback):

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch

        val_loss = trainer.callback_metrics["val_loss"].cpu().numpy()
        if 'eff' in trainer.callback_metrics and 'total_pur' in trainer.callback_metrics:
            efficiency = trainer.callback_metrics['eff']
            total_purity = trainer.callback_metrics['total_pur']

            pruning_perc = pl_module.get_pruning_percentage() if hasattr(pl_module, "get_pruning_percentage") else 0.
            print(f"\nEpoch {epoch} validation loss {val_loss.item():.6f} Efficiency: {efficiency:.4f}, Total Purity: {total_purity:.4f} Pruning {pruning_perc:.3}%")


# Setup of pruning callback
def setup_pruning(stage_module, trainer, config):
    parameters_to_prune = [
        (module, "weight")
        for name, module in stage_module.named_modules()
        if isinstance(module, nn.Linear) and name[-1] != "6" and name not in ["edge_output_transform.3", "node_network.7.0", "node_network.7.3"]
    ]

    pruning_importance_score = config.get("pruning_importance_score", "wmp") # Default weight magnitude pruning

    if pruning_importance_score == "gmp": # gradient magnitude pruning
        stage_module.calculate_grad_scores = True # calculate gradients after backward step

    pruning_amount = config.get("pruning_amount", 0.1)
    if isinstance(pruning_amount, dict):
        # Parse to a ModelPruning Callback function
        pruning_amount = partial(get_pruning_amount_generic, stage_module=stage_module, pruning_dict=parse_pruning_amount_dict(pruning_amount))
        
    n_retraining_epochs = config.get("n_retraining_epochs", 0)
    restart_pruning = config.get("restart_pruning", True)

    pruning_callback = StructuredModelPruning(
        pruning_fn="ln_structured",
        parameters_to_prune=parameters_to_prune,
        amount=pruning_amount,
        apply_pruning=partial(apply_pruning, stage_module=stage_module, n_retraining_epochs=n_retraining_epochs),
        pruning_dim=0, # prune the output channels/neurons
        use_global_unstructured=False,
        pruning_norm=config.get("pruning_aggregation", 2),
        make_pruning_permanent=True,
        restart_pruning=restart_pruning
    )

    trainer.callbacks.append(PruningPrintingCallback()) # before pruning is applied
    trainer.callbacks.append(pruning_callback)


    # Default values, overwritten by pruning restart
    if restart_pruning:
        stage_module.last_pruned = stage_module.hparams.get("last_pruned", -1)
    else:
        stage_module.last_pruned = -n_retraining_epochs # Prune in the first possible epoch


    print("Pruning configuration summary:\n" \
    f"  Importance score function {pruning_importance_score} \n"\
    f"  Importance score aggregation L{config.get('pruning_aggregation', 2)}\n" \
    f"  Pruning criteria TopK \n"\
    f"  Pruning amount {pruning_amount.func.__name__ if callable(pruning_amount) else pruning_amount}\n" \
    f"  Number of retraining epochs {n_retraining_epochs}\n" \
    f"  Restarting pruning {restart_pruning} from epoch {stage_module.last_pruned}\n")


def get_pruning_amount_generic(epoch, stage_module, pruning_dict):
    current_pruning = stage_module.get_pruning_percentage()
    for pruning_threshold, pruning_amount in pruning_dict.items():
        if current_pruning < pruning_threshold:
            return pruning_amount

    warnings.warn(f"Pruning amount for the current pruning percentage {current_pruning}% was not defined")
    return 0

# Ensure the pruning amounts are sorted by ascending pruning percentage conditions
def parse_pruning_amount_dict(pruning_dict):
    return dict(sorted(pruning_dict.items())) 

# frequency pruning
def apply_pruning(epoch, stage_module, n_retraining_epochs):
    
    if (epoch - stage_module.last_pruned >= n_retraining_epochs):
        stage_module.last_pruned = epoch

        # TODO check if this is saved in the checkpoint
        stage_module.hparams["last_pruned"] = epoch
        print(f"Applying pruning at epoch {epoch}")

        return True

    return False


# TODO: threshold pruning