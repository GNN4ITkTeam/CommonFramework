import os
import numpy as np
import torch
from scipy.integrate import simps

import torch.nn as nn
import torch.nn.utils.prune as prune
import brevitas.nn as qnn
import brevitas.quant as quant
from brevitas.quant_tensor import QuantTensor
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup
from qonnx.util.inference_cost import inference_cost
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from gnn4itk_cf.stages.graph_construction.models.utils import (
    build_edges,
)
from gnn4itk_cf.stages.graph_construction.utils import build_signal_edges


def quantize_features(features, pre_point: int = 0, post_point: int = 0):
    features = torch.clamp(
        features, -(2**pre_point), 2**pre_point - 2 ** (-post_point)
    )

    features = features * (2**post_point)
    features = features.to(torch.int32)
    features = features.to(torch.float32)
    features = features / (2**post_point)

    return features


def make_quantized_mlp(
    input_size,  # input parameters of neural net
    sizes,
    weight_bit_width=[8, 8, 8],  # providing weights in form of array now
    bias=True,
    bias_quant=8,
    activation_qnn=True,
    activation_bit_width=[8, 8, 8],
    output_activation=False,
    output_activation_quantization=False,
    input_layer_quantization=True,
    input_layer_bitwidth=11,
    batch_norm=True,
    layer_norm=False,
):
    """Construct a Qunatized MLP with specified fully-connected layers."""

    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # adding first layer for quantizing the input

    if input_layer_quantization:
        # quantizing the input layer
        layers.append(
            qnn.QuantIdentity(bit_width=input_layer_bitwidth, return_quant_tensor=True)
        )

    # Hidden layers of a quantized neural network
    print(f"bias_quant: {bias_quant}")

    if bias_quant == 8:
        qnn_bias_quant = quant.Int8Bias
    elif bias_quant == 16:
        qnn_bias_quant = quant.Int16Bias
    elif bias_quant == 24:
        qnn_bias_quant = quant.Int24Bias
    else:
        qnn_bias_quant = quant.Int32Bias

    for i in range(n_layers - 1):
        layers.append(
            qnn.QuantLinear(
                sizes[i],
                sizes[i + 1],
                bias=bias,
                bias_quant=qnn_bias_quant,  # to be made configurable
                weight_bit_width=weight_bit_width[0 if i == 0 else 1],
                return_quant_tensor=True,
            )
        )  # adding first and hidden layer weights
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1], elementwise_affine=False))
        if batch_norm:  # using batch norm
            layers.append(
                nn.BatchNorm1d(sizes[i + 1], track_running_stats=True, affine=True)
            )  # check parameters!
        if activation_qnn:  # if qnn activation is on , we use QuantReLU else nn.ReLU
            if i == 0:
                activation_bit_index = 0
                print("first quantRelu")
            elif i == n_layers - 2 and (not output_activation):
                activation_bit_index = 2
                print("last quantRelu")
            else:
                activation_bit_index = 1
                print("hidden quantRelu")
            layers.append(
                qnn.QuantReLU(
                    bit_width=activation_bit_width[activation_bit_index],
                    return_quant_tensor=True,
                )
            )

        else:
            layers.append(nn.ReLU())

    # Final layer
    layers.append(
        qnn.QuantLinear(
            sizes[-2],
            sizes[-1],
            # input_quant = Int8ActPerTensorFloat, # it can tolerate biases like that, but that is not desired
            bias=bias,
            bias_quant=qnn_bias_quant,  # to be made configurable
            weight_bit_width=weight_bit_width[-1],
            return_quant_tensor=output_activation,
        )
    )  # if no output activation, have to returned not-quant tensor!

    if output_activation:
        # print(f"adding output activation! {output_activation} {output_activation_quantization}")
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1], elementwise_affine=False))
        if batch_norm:
            layers.append(
                nn.BatchNorm1d(sizes[-1], track_running_stats=True, affine=True)
            )  # check parameters!)

        if output_activation_quantization:
            layers.append(
                qnn.QuantReLU(
                    bit_width=activation_bit_width[-1], return_quant_tensor=False
                )
            )
        else:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class onnx_export(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            model = pl_module
            if trainer.current_epoch != 0:
                self.get_onnx_dir(trainer)  # setting up the onnx and pickle dir path

                model_copy = model.deepcopy_model()
                parameters_to_prune = [
                    (x, "weight")
                    for x in model_copy
                    if (
                        hasattr(x, "weight")
                        and (x.__class__.__name__ != "LayerNorm")
                        and (x.__class__.__name__ != "BatchNorm1d")
                    )
                ]
                if model.last_pruned > -1:
                    for paras in parameters_to_prune:
                        prune.remove(paras[0], name="weight")

                export_path, export_path_cleanup, export_json = self.get_onnx_paths(
                    trainer.current_epoch
                )

                export_qonnx(
                    model_copy,
                    export_path=export_path,
                    input_t=self.input_tensor(model.hparams).to("cuda"),
                    export_params="True",
                )  # exporting the model to calculate BOPs just before we do pruning

                # pickle_file_path = self.pickle_directory + f"/model_{trainer.current_epoch}_.pkl"
                # with open(pickle_file_path,'wb') as file:
                #     model.voila(dir(model_copy))
                #     pickle.dump(model.network,file)
                # file.close()

                cleanup(export_path, out_file=export_path_cleanup)
                inf_cost = inference_cost(
                    export_path_cleanup, output_json=export_json, discount_sparsity=True
                )

                del model_copy

                # check if efficiency is actually fine; at least 95% efficient
                if np.isclose(model.eff_98, 0.98, atol=0.03):
                    if (
                        model.pur_98 >= 0.95 * model.hparams["target_purity"]
                    ):  # within at least 5 % of target purity
                        weighted_bops = inf_cost["total_bops"]
                        if weighted_bops < model.final_bops:
                            model.final_bops = weighted_bops
                            model.final_pur_98 = model.pur_98
                        elif (
                            np.isclose(int(weighted_bops), int(model.final_bops))
                            and model.pur_98 > model.final_pur_98
                        ):
                            model.final_pur_98 = model.pur_98
                    else:
                        weighted_bops = (inf_cost["total_bops"] + 1) * np.exp(
                            5 * model.hparams["target_purity"] / model.pur_98
                        )  # exponential penalty factor
                else:  # if not, than blow it up
                    weighted_bops = np.inf

                model.log_dict(
                    {
                        "total_bops": inf_cost["total_bops"],
                        "total_mem_w_bits": inf_cost["total_mem_w_bits"],
                        "total_mem_o_bits": inf_cost["total_mem_o_bits"],
                        "weighted_bops": weighted_bops,  # we can use this to minimize BOPs for a decent performance
                        "final_pur_98": model.final_pur_98,
                        "final_bops": model.final_bops,
                    }
                )

    def input_tensor(self, config):
        batch_size = 1  # to measure BOPs per spacepoint
        input_tensor = torch.randn(batch_size, len(config["node_features"]))

        if config["quantized_network"]:
            input_bitwidth = config["integer_part"] + config["fractional_part"] + 1
            scale_tensor = torch.full(
                (1, len(config["node_features"])), 1.0 / (input_bitwidth)
            )
            zp = torch.tensor(0)
            signed = True
            input_tensor = QuantTensor(
                input_tensor, scale_tensor, zp, input_bitwidth, signed, training=False
            )
        else:
            input_tensor = input_tensor

        return input_tensor

    def get_onnx_paths(self, epoch):
        export_path = self.onnx_directory + f"/pruning_{epoch}.onnx"
        export_path_cleanup = self.onnx_directory + f"/pruning_{epoch}_clean.onnx"
        export_json = self.onnx_directory + f"/pruning_{epoch}.json"

        return export_path, export_path_cleanup, export_json

    def get_onnx_dir(self, trainer):
        if trainer.current_epoch != 0:  # just to avoid crashing at zero epoch
            model_checkpoint = [
                callback
                for callback in trainer.callbacks
                if isinstance(callback, ModelCheckpoint)
            ][0]
            file_directory = (
                model_checkpoint.kth_best_model_path
            )  # it gives path/best-version.ckpt
            file_directory = file_directory[:-5]  # removes .ckpt extension
            onnx_directory = file_directory.replace(
                "artifacts", "onnx_exports"
            )  # replace artifact with onnx_export directory
            if not os.path.exists(
                onnx_directory
            ):  # 3creating directory acc to the version
                os.makedirs(onnx_directory)

            self.onnx_directory = onnx_directory


class auc_score(Callback):
    """

    this class is called after training is finished and calculate auc for purity vs eff curve
    It calculates auc scores for both signal and total 'eff vs purity' curve.

    """

    # removed AUC score calculation callback on_train_end, since it leads to crashes at the end
    # def on_train_end(self, trainer, pl_module):
    #    if trainer.current_epoch != 0:
    #        self.auc(pl_module)

    def auc(self, model):
        model.eval()
        radius_arr = np.linspace(0.01, 0.1, 3)
        total_eff_arr = np.zeros(len(radius_arr))
        total_pur_arr = np.zeros(len(radius_arr))

        signal_eff_arr = np.zeros(len(radius_arr))
        signal_pur_arr = np.zeros(len(radius_arr))

        for batch in model.testset:
            (
                total_eff_list,
                total_pur_list,
                signal_eff_list,
                signal_pur_list,
            ) = self.eff_and_pur(model, batch, radius_arr, model.hparams["knn_test"])

            total_eff_arr += total_eff_list
            total_pur_arr += total_pur_list

            signal_eff_arr += signal_eff_list
            signal_pur_arr += signal_pur_list

        total_eff_arr = total_eff_arr / len(model.testset)  # taking avg
        total_pur_arr = total_pur_arr / len(model.testset)

        signal_eff_arr = signal_eff_arr / len(model.testset)
        signal_pur_arr = signal_pur_arr / len(model.testset)

        auc_value_total = simps(total_pur_arr, total_eff_arr)  # y axis is purity
        auc_value_signal = simps(signal_pur_arr, signal_eff_arr)  # y axis is purity

        # below crashes!
        # lightning_fabric.utilities.exceptions.MisconfigurationException: You can't `self.log()` inside `on_train_end`. HINT: You can still log directly to the logger by using `self.logger.experiment`.
        # model.log("auc_score_total", auc_value_total)
        # model.log("auc_score_signal", auc_value_signal)

    def eff_and_pur(self, model, batch, radius_arr, knn):
        """
        Gives efficiency and purity for a radius array [0.02 till 0.1]
        """

        embedding = model.apply_embedding(batch.to("cuda"))
        total_eff_arr = []
        total_pur_arr = []

        signal_eff_arr = []
        signal_pur_arr = []

        for radius in radius_arr:
            batch.edge_index = build_edges(
                query=embedding,
                database=embedding,
                indices=None,
                r_max=radius,
                k_max=500,
                backend="FRNN",
            )

            batch.edge_index, batch.y, batch.truth_map, true_edges = model.get_truth(
                batch, batch.edge_index
            )

            true_pred_edges = batch.edge_index[:, batch.y == 1]

            signal_true_edges = build_signal_edges(
                batch, model.hparams["weighting"], true_edges
            )
            weights = model.get_weights(batch)
            signal_true_pred_edges = batch.edge_index[:, (batch.y == 1) & (weights > 0)]

            signal_eff = signal_true_pred_edges.shape[1] / signal_true_edges.shape[1]
            signal_pur = signal_true_pred_edges.shape[1] / batch.edge_index.shape[1]

            total_eff = true_pred_edges.shape[1] / true_edges.shape[1]
            total_pur = true_pred_edges.shape[1] / batch.edge_index.shape[1]

            total_eff_arr.append(total_eff)
            total_pur_arr.append(total_pur)

            signal_eff_arr.append(signal_eff)
            signal_pur_arr.append(signal_pur)

        return (
            np.array(total_eff_arr),
            np.array(total_pur_arr),
            np.array(signal_eff_arr),
            np.array(signal_pur_arr),
        )
