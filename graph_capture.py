"""
Capture the forward and backward graph of nanoGPT
"""
import os
import pickle

from torch import nn

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

"""
Dump Graphs
"""

import torch._inductor.compile_fx
from typing import List

from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class AtenGraphs():
    forward_graph: torch.fx.GraphModule = None
    backward_graph: torch.fx.GraphModule = None


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


def inner_compiler(gm: torch.fx.GraphModule,
                   example_inputs: List[torch.Tensor],
                   cudagraphs=None,
                   num_fixed=0,
                   is_backward=False,
                   graph_id=None):
    if is_backward:
        graph_capture.graph.backward_graph = gm
    else:
        graph_capture.graph.forward_graph = gm

    return gm


# capture graphs
def capture_forward_backward_graphs(model: nn.Module):
    my_compiler = lambda gm, _: torch._inductor.compile_fx.compile_fx(gm, _, inner_compile=inner_compiler)
    with graph_capture.set_graph() as graph:
        model_opt = torch.compile(model, backend=my_compiler)
        model_opt(torch.randint(low=0, high=model_args['vocab_size'], size=(1, 1), device='cpu', dtype=torch.int64))

    return graph.forward_graph, graph.backward_graph


forward_graph, backward_graph = capture_forward_backward_graphs(model)

save_to_disk = False

if save_to_disk:
    torch.save(forward_graph, 'forward.pt')
    torch.save(backward_graph, 'backward.pt')
