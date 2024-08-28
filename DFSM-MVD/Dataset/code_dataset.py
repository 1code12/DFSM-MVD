
from torch.utils.data import Dataset, DataLoader
import json
import copy
from tqdm import tqdm
import warnings

from transformers import RobertaTokenizer
import torch
from dgl import DGLGraph
import dgl

import torch
from torch.utils.data import Dataset, DataLoader
import json
import copy
from tqdm import tqdm
import warnings
from dgl import DGLGraph
from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained('codebert/model')


def convert_examples_to_features(code, tokenizer=tokenizer):
    block_size = 512
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return source_ids

class DataEntry:
    def __init__(self, _id, datset, num_nodes, features, edges, target, code, ast):
        self._id = _id
        self.code = code  # code sequence
        self.ast = ast  # ast sequence
        self.num_nodes = num_nodes
        self.target = target
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(
            self.num_nodes,
            data={'features': self.features}
        )
        for s, _type, t in edges:
            etype_number = datset.get_edge_type_number(_type)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.graph.add_edge(s, t, data={'etype': torch.LongTensor([etype_number])})

			
class BatchGraph:
    def __init__(self):
        self.graph = DGLGraph()
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0

    def add_subgraph(self, _g):
        assert isinstance(_g, DGLGraph)

        num_new_nodes = _g.number_of_nodes()

        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes))).to(torch.device('cuda:0'))

        self.graph.add_nodes(num_new_nodes, data=_g.ndata)

        sources, dests = _g.all_edges()

        sources += self.number_of_nodes

        dests += self.number_of_nodes

        self.graph.add_edges(sources, dests, data=_g.edata)

        self.number_of_nodes += num_new_nodes

        self.num_of_subgraphs += 1

    def cuda(self, device=None):
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)

    def de_batchify_graphs(self, features=None):
        assert isinstance(features, torch.Tensor)

        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) for gid in
                   self.graphid_to_nodeids.keys()]
        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        for i, v in enumerate(vectors):
            vectors[i] = torch.cat((v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])),
                                                   requires_grad=v.requires_grad, device=v.device)), dim=0)
        output_vectors = torch.stack(vectors)

        return output_vectors  # , lengths

    def get_network_inputs(self, cuda=False):
        raise NotImplementedError('Must be implemented by subclasses.')


from scipy import sparse as sp


class GGNNBatchGraph(BatchGraph):
    def __init__(self):
        super(GGNNBatchGraph, self).__init__()

    def get_network_inputs(self, cuda=False, device=None):

        features = self.graph.ndata['features']
        edge_types = self.graph.edata['etype']

        if cuda:
            return self.graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return self.graph, features, edge_types
        pass


def de_batchify_graphs(graph, features=None):
    graph.ndata["features"] = features
    graphs = dgl.unbatch(graph)
    vectors = [g.ndata['features'] for g in graphs]
    lengths = [f.size(0) for f in vectors]
    max_len = max(lengths)
    for i, v in enumerate(vectors):
        vectors[i] = torch.cat(
            (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                            device=v.device)), dim=0)
    output_vectors = torch.stack(vectors)
    return output_vectors


class CodeDataset(Dataset):
    def __init__(self, src_file, n_ident, g_ident, l_ident):
        self.examples = []
        self.edge_types = {}
        self.max_etype = 0
        self.feature_size = 0
        self.n_ident = n_ident
        self.g_ident = g_ident
        self.l_ident = l_ident
        self.read_dataset(src_file)

    def read_dataset(self, src_file):
        with open(src_file, "r") as fp:
            for i, line in enumerate(tqdm(fp)):
                entry = json.loads(line)
                example = DataEntry(
                    _id=i,
                    datset=self,
                    num_nodes=len(entry['full_graph'][self.n_ident]),
                    features=entry['full_graph'][self.n_ident],
                    edges=entry['full_graph'][self.g_ident],
                    target=entry['full_graph'][self.l_ident][0][0],
                    code=convert_examples_to_features(entry['code']),
                    ast=convert_examples_to_features(entry['ast'])
                )
                if self.feature_size == 0:
                    self.feature_size = example.features.size(1)
                self.examples.append(example)

    def get_edge_type_number(self, _type):
        if (_type not in self.edge_types):
            self.edge_types[_type] = self.max_etype
            self.max_etype += 1
        return self.edge_types[_type]

    @property
    def max_edge_type(self):
        return self.max_etype

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return example.graph, torch.FloatTensor([example.target]), example.code, example.ast


def collate_fn(batch):
    graphs, labels, codes, asts = zip(*batch)
    batch_graph = GGNNBatchGraph()
    for graph in graphs:
        batch_graph.add_subgraph(graph)
    labels = torch.cat(labels)
    return batch_graph, labels, codes, asts


def main():
    #Test
    dataset = CodeDataset(src_file='Dataset/Devign/test.jsonl', n_ident='node_features', g_ident='graph', l_ident='targets')

    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    for batch_graph, labels, codes, asts in tqdm(data_loader, desc="Testing DataLoader"):
        print(
            f"Batch Graph Shape: {batch_graph.graph.number_of_nodes()} nodes, {batch_graph.graph.number_of_edges()} edges")
        print(f"Labels Shape: {labels.size()}")
        print(f"Codes Length: {len(codes)}, ASTs Length: {len(asts)}")
        print("=" * 50)
        print(batch_graph)
        print(labels)
        print(codes.shape)
        print(asts.shape)
        break


if __name__ == "__main__":
    main()




