"""
Capture the forward and backward graph of nanoGPT
"""
import functools
import os
import pickle
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List

import torch
import torch._inductor.compile_fx
from torch import nn


@dataclass
class AtenGraphs():
    forward_graph: torch.fx.GraphModule = None
    forward_input: List[torch.Tensor] = None  # Note that the input maybe a WeakRef Tensor that is not pickleable
    backward_graph: torch.fx.GraphModule = None
    backward_input: List[torch.Tensor] = None

    def as_dict(self):
        return {
            'forward_graph': self.forward_graph,
            'forward_input': self.forward_input,
            'backward_graph': self.backward_graph,
            'backward_input': self.backward_input
        }


class GraphCaptureContext():
    def __init__(self):
        self.graph = None

    def set_graph(self):
        @contextmanager
        def ctx():
            try:
                self.graph = AtenGraphs()
                yield self.graph
            finally:
                self.graph = None

        return ctx()


graph_capture = GraphCaptureContext()

import torch._dynamo.symbolic_convert
import torch._inductor.compile_fx as compile_fx

import functorch.compile


class GraphCaptureBackend(object):
    def __init__(self, decomposition=None):
        if decomposition is None:
            decomposition = []

        assert isinstance(decomposition, List)
        from torch._decomp import get_decompositions
        self.decomposition = get_decompositions(decomposition)

    def __call__(self, model_, example_inputs_):
        functorch.compile.config.use_functionalize = True
        functorch.compile.config.use_fake_tensor = True

        def inner_compile(gm: torch.fx.GraphModule,
                          example_inputs: List[torch.Tensor],
                          is_backward=False):
            if is_backward:
                graph_capture.graph.backward_graph = gm
                graph_capture.graph.backward_input = example_inputs
            else:
                graph_capture.graph.forward_graph = gm
                graph_capture.graph.forward_input = example_inputs
            return gm

        def fw_compiler(model: torch.fx.GraphModule, example_inputs):
            return inner_compile(
                model,
                example_inputs,
                is_backward=False
            )

        def bw_compiler(model: torch.fx.GraphModule, example_inputs):
            return inner_compile(
                model,
                example_inputs,
                is_backward=True,
            )

        with compile_fx.overrides.patch_functions():
            # TODO: can add logging before/after the call to create_aot_dispatcher_function
            # in torch._functorch/aot_autograd.py::aot_module_simplified::aot_function_simplified::new_func
            # once torchdynamo is merged into pytorch
            return compile_fx.aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                decompositions=self.decomposition,
                partition_fn=functools.partial(
                    compile_fx.min_cut_rematerialization_partition, compiler="inductor"
                ),
                keep_inference_input_mutations=True,
            )(model_, example_inputs_)


# capture graphs
def capture_forward_backward_graphs(model: nn.Module, model_args, reduce_fn, decomposition=None):
    backend = GraphCaptureBackend(decomposition=decomposition)
    torch._dynamo.reset()
    with graph_capture.set_graph() as graph:
        model_opt = torch.compile(model, backend=backend)
        output = model_opt(**model_args)
        loss = reduce_fn(output)
        loss.backward()

    return graph


class TestGraphCaptureBackend(unittest.TestCase):
    def test_softmax_layernorm(self):
        class Model(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.layer_norm = nn.LayerNorm(normalized_shape=[input_dim])

            def forward(self, x):
                x = self.layer_norm(x)
                x = torch.softmax(x, dim=-1)
                return x

        input_dim = 10
        model = Model(input_dim)
        x = torch.randn(1, input_dim)
        graph: AtenGraphs = capture_forward_backward_graphs(model, model_args={'x': x}, reduce_fn=torch.sum,
                                                            decomposition=None)
        # verify softmax and layernorm is not decomposed
        graph.forward_graph.graph.print_tabular()
        graph.backward_graph.graph.print_tabular()

    def test_embedding_decomposition(self):
        class Model(nn.Module):
            def __init__(self, num_embeddings, embedding_dim):
                super().__init__()
                self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

            def forward(self, x):
                x = self.embedding_layer(x)
                x = torch.softmax(x, dim=-1)
                return x

        num_embeddings = 10
        embedding_dim = 20
        model = Model(num_embeddings, embedding_dim)
        x = torch.randint(low=0, high=num_embeddings, size=(100,), dtype=torch.int64)
        decomposition = [torch.ops.aten.embedding_dense_backward]
        graph: AtenGraphs = capture_forward_backward_graphs(model, model_args={'x': x}, reduce_fn=torch.sum,
                                                            decomposition=decomposition)
        # verify embedding backward is decomposed
        graph.forward_graph.graph.print_tabular()
        graph.backward_graph.graph.print_tabular()



    @unittest.SkipTest
    def test_nanoGPT(self):
        from model import GPTConfig, GPT

        # data
        dataset = 'openwebtext'
        gradient_accumulation_steps = 5  # used to simulate larger batch sizes
        batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
        block_size = 1024
        # model
        n_layer = 12
        n_head = 12
        n_embd = 768
        dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
        bias = False  # do we use bias inside LayerNorm and Linear layers?

        # model init
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                          bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line

        data_dir = os.path.join('data', dataset)
        # attempt to derive vocab_size from the dataset
        meta_path = os.path.join(data_dir, 'meta.pkl')
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            meta_vocab_size = meta['vocab_size']
            print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        input = torch.randint(low=0, high=model_args['vocab_size'], size=(1, 1), device='cpu',
                              dtype=torch.int64)

        graph: AtenGraphs = capture_forward_backward_graphs(model, model_args={'idx': input}, reduce_fn=torch.sum,
                                                            decomposition=None)

        # do something with the graph


if __name__ == '__main__':
    unittest.main()
