# Copyright (C) 2023 CERN for the benefit of the ATLAS collaboration

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch

import brevitas.nn as qnn


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation=None,
    layer_norm=False,
    batch_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False))
        layers.append(output_activation())
    return nn.Sequential(*layers)

def make_quantized_mlp(

    input_size,  ##input parameters of neural net
    sizes,  
    weight_bit_width,  ##providing weights in form of array now
    activation_qnn = True,
    activation_bit_width=4,
    output_activation = False,
    output_activation_quantization = False,
    input_layer_quantization=False,
    input_layer_bitwidth = 11,
    layer_norm = True,

):
    """Construct a Qunatized MLP with specified fully-connected layers."""
    
    layers = []  
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    ##adding first layer for quantizng the input

    if(input_layer_quantization):
        ##quantizing the input layer
        layers.append(qnn.QuantIdentity(
                bit_width=input_layer_bitwidth,return_quant_tensor = True ))

    
    # Hidden layers of a quantized neural network

    for i in range(n_layers-1):

        layers.append(qnn.QuantLinear(sizes[i], sizes[i + 1],bias=False,
         weight_bit_width=weight_bit_width[0 if i ==0 else 1],return_quant_tensor = True))  ##adding first and hidden layer weights
        if layer_norm:   ##using batch norm
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        if activation_qnn:   ##if qnn activation is on , we use QuantReLU else nn.ReLU
            if i==0:
                activation_bit_index = 0
                print("first quantRelu")
            elif (i==n_layers-2 and (not output_activation)):
                activation_bit_index = 2
                print("last quantRelu")
            else:
                activation_bit_index = 1
                print("hidden quantRelu")
            layers.append(qnn.QuantReLU(bit_width = activation_bit_width[activation_bit_index],return_quant_tensor = True))
           
        else:
            layers.append(nn.ReLU())

    # Final layer
    layers.append(qnn.QuantLinear(sizes[-2], sizes[-1],bias=False, 
    weight_bit_width=weight_bit_width[-1],return_quant_tensor = output_activation)) # if no output activation, have to returned not-quant tensor!

    if output_activation:
        #print(f"adding output activation! {output_activation} {output_activation_quantization}")
        if layer_norm:
            layers.append(nn.BatchNorm1d(sizes[-1]))

        if output_activation_quantization:
            layers.append(qnn.QuantReLU(bit_width = activation_bit_width[-1],return_quant_tensor = True))
        else:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)

def get_optimizers(parameters, hparams):
    optimizer = [
        torch.optim.AdamW(
            parameters,
            lr=(hparams["lr"]),
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=True,
        )
    ]

    if "scheduler" not in hparams or hparams["scheduler"] is None or hparams["scheduler"] == "StepLR":
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer[0],
                    step_size=hparams["patience"],
                    gamma=hparams["factor"],
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    elif hparams["scheduler"] == "ReduceLROnPlateau":
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer[0],
                    mode="min",
                    factor=hparams["factor"],
                    patience=hparams["patience"],
                    verbose=True,
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    elif hparams["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = [
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer[0],
                    T_0=hparams["patience"],
                    T_mult=2,
                    eta_min=1e-8,
                    last_epoch=-1,
                ),
                "interval": "epoch",
                "frequency": 1,
            }
        ]
    else:
        raise ValueError(f"Unknown scheduler: {hparams['scheduler']}")

    return optimizer, scheduler