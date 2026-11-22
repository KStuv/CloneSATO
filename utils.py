import torch
from torch_sparse import SparseTensor
from torch import Tensor
import torch_sparse
from typing import List, Tuple


class PermIterator:
    '''
    Iterator of a permutation
    '''
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret


def sparsesample(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > 0
    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand]

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask]

    ret = SparseTensor(row=samplerow.reshape(-1, 1).expand(-1, deg).flatten(),
                       col=samplecol.flatten(),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce().fill_value_(1.0)
    #print(ret.storage.value())
    return ret


def sparsesample2(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(
        row=torch.cat((samplerow, nosamplerow)),
        col=torch.cat((samplecol, nosamplecol)),
        sparse_sizes=adj.sparse_sizes()).to_device(
            adj.device()).fill_value_(1.0).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def sparsesample_reweight(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix. It will also scale the sampled elements.
    
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()
    samplevalue = (rowcount * (1/deg)).reshape(-1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(row=torch.cat((samplerow, nosamplerow)),
                       col=torch.cat((samplecol, nosamplecol)),
                       value=torch.cat((samplevalue,
                                        torch.ones_like(nosamplerow))),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def elem2spm(element: Tensor, sizes: List[int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem


def _element_membership(element1: Tensor, element2: Tensor):
    """Return whether each element in element2 occurs in sorted element1,
    together with its insertion index. Handles empty sparse tensors safely.
    """
    if element1.numel() == 0:
        idx = torch.zeros_like(element2, dtype=torch.long)
        return torch.zeros_like(element2, dtype=torch.bool), idx
    idx = torch.searchsorted(element1, element2)
    in_bounds = idx < element1.numel()
    matched = torch.zeros_like(element2, dtype=torch.bool)
    if in_bounds.any():
        matched[in_bounds] = element1[idx[in_bounds]] == element2[in_bounds]
    return matched, idx


def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    """Compute the element-wise overlap of two sparse adjacency matrices."""
    assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)
    matched, _ = _element_membership(element1, element2)
    return elem2spm(element2[matched], adj1.sizes())


def spmnotoverlap_(adj1: SparseTensor,
                   adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
    """Return elements in adj1 but not adj2, and in adj2 but not adj1."""
    assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)
    matched2, idx2 = _element_membership(element1, element2)
    keep1 = torch.ones_like(element1, dtype=torch.bool)
    if matched2.any():
        keep1[idx2[matched2]] = False
    return (elem2spm(element1[keep1], adj1.sizes()),
            elem2spm(element2[~matched2], adj2.sizes()))


def spmoverlap_notoverlap_(
        adj1: SparseTensor,
        adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
    """Return overlap, adj1-only elements, and adj2-only elements."""
    assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)
    matched2, idx2 = _element_membership(element1, element2)
    keep1 = torch.ones_like(element1, dtype=torch.bool)
    if matched2.any():
        keep1[idx2[matched2]] = False
    sizes = adj1.sizes()
    return (elem2spm(element2[matched2], sizes),
            elem2spm(element1[keep1], sizes),
            elem2spm(element2[~matched2], sizes))


def adjoverlap(adj1: SparseTensor,
               adj2: SparseTensor,
               tarei: Tensor,
               filled1: bool = False,
               calresadj: bool = False,
               cnsampledeg: int = -1,
               ressampledeg: int = -1):
    # a wrapper for functions above.
    adj1 = adj1[tarei[0]]
    adj2 = adj2[tarei[1]]
    if calresadj:
        adjoverlap, adjres1, adjres2 = spmoverlap_notoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
        if ressampledeg > 0:
            adjres1 = sparsesample_reweight(adjres1, ressampledeg)
            adjres2 = sparsesample_reweight(adjres2, ressampledeg)
        return adjoverlap, adjres1, adjres2
    else:
        adjoverlap = spmoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
    return adjoverlap


if __name__ == "__main__":
    adj1 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]]))
    adj2 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 3, 1, 2, 3], [0, 1, 1, 2, 3]]))
    adj3 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 1,  2, 2, 2,2, 3, 3, 3], [1, 0,  2,3,4, 5, 4, 5, 6]]))
    print(spmnotoverlap_(adj1, adj2))
    print(spmoverlap_(adj1, adj2))
    print(spmoverlap_notoverlap_(adj1, adj2))
    print(sparsesample2(adj3, 2))
    print(sparsesample_reweight(adj3, 2))