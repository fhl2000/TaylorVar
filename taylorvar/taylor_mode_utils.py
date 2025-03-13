from __future__ import annotations
import torch
from typing import Callable, Optional, List, Union

def to_slice(i):
    """Unify the index format"""
    if isinstance(i, int):
        return slice(i, i+1)
    if i is None:
        return slice(None)
    return i

def process_idx(idx):
    if_all_slice=True
    if idx is not None:
        assert isinstance(idx, tuple), f"idx should be a tuple, but got {type(idx)}"
        if any( not isinstance(i, slice) for i in idx):
            if_all_slice=False
        # unify the format to slice
        idx1 = tuple(to_slice(i) for i in idx)
        trans_fn = lambda x: 0 if isinstance(x,int) else slice(None)
        trans_idx = tuple( trans_fn(x) for x in idx)
        return if_all_slice, idx1, trans_idx
    
    return if_all_slice, None, None

def expand_val(val, ndim):
    """
    Expand the value tensor according to the derivative order
    """
    for _ in range(ndim):
        val = val.unsqueeze(-1)
    return val

def make_hashable_idx(idx):
    return tuple(((i.start, i.stop, i.step) if isinstance(i, slice) else i) for i in idx)

class DerivativeTensor:
    """Wrapper for derivative tensor, supporting delayed calculation"""
    def __init__(self, parent: TaylorVar, order: int, tensor: Optional[torch.Tensor] = None):
        self.parent = parent
        self.order = order
        self._tensor = tensor
        self._cache = {}
        
    def __getitem__(self, idx):
        """
        - [...] or [:,:,:]  indicates full computation
        - [i,j,k] indicates specific derivative component
        - [i1:i2, j1:j2, k1:k2] indicates a block of derivative components
        Not support fancy indexing (advanced indexing), boolean indexing, etc.
        """
        if self.order <= 1:
            # low order force full computation
            if self._tensor is None:
                self._tensor = self.parent._compute_fn(self.order, self.parent)
        
        if idx is Ellipsis or (
            isinstance(idx, tuple) and 
            all(isinstance(i, slice) and i == slice(None) for i in idx) and 
            len(idx) == self.order
        ):
            # full computation
            if self._tensor is None:
                self._tensor = self.parent._compute_fn(self.order, self.parent)
            return self._tensor
        
        # make sure the index is a tuple
        if not isinstance(idx, tuple):
            idx = (idx,)
        
        assert len(idx) == self.order, f"Expected {self.order} indices, got {len(idx)}"
        
        key = make_hashable_idx(idx)
        if self._tensor is None:
            if key not in self._cache:
                self._cache[key] = self.parent._compute_fn(self.order, self.parent, idx)
            return self._cache[key]
        
        # if already cached, directly return
        return self._tensor[(...,) + idx]

    def to_tensor(self):
        """Force calculation and return the full tensor"""
        if self._tensor is None:
            self._tensor = self.parent._compute_fn(self.order, self.parent)
        return self._tensor

class TaylorVar:
    """
    Taylor mode variable is a wrapper of a tensor that stores a tensor and 
    its derivatives (up to third order) respect to the input variable. and is
    used for explicitly construct the high-order forward propagation computational graph.
    """

    def __init__(
        self, 
        val: torch.Tensor, 
        first: Optional[torch.Tensor] = None, 
        second: Optional[torch.Tensor] = None,
        third: Optional[torch.Tensor] = None,
        order: int = 0,    # soft control the order of Taylor expansion, up to 3
        compute_fn: Optional[Callable] = None, #  used for delay computing the higher order derivatives
        input_dim: Optional[int] = None,
    ):
        """
        val:   value (...)
        first: first-order derivative (..., d)
        second: second-order derivative (..., d, d)
        third: third-order derivative (..., d, d, d)
        input_dim: `d`, the dimension of the independent variable, if None, infer from first
        """
        self.val = val
        
        self._first = DerivativeTensor(self, 1, first) if first is not None else None
        self._second = DerivativeTensor(self, 2, second) if second is not None else None
        self._third = DerivativeTensor(self, 3, third) if third is not None else None
        
        self._shape = val.shape
        self._dtype = val.dtype
        self._device = val.device

        self.order = order
        self._compute_fn = compute_fn
        
        # initialize input_dim
        if input_dim is not None:
            self.input_dim = input_dim
        elif first is not None:
            self.input_dim = first.shape[-1]
        else:
            self.input_dim = 1  # default to 1
        self.d = self.input_dim
    
    
    def __repr__(self):
        fn_str = f", fn={self._compute_fn.__name__}" if self._compute_fn else ''
        return f"TaylorVar(val={self.val}, order={self.order}{fn_str})"
    def to_tensor(self):
        return self.val
    @property
    def first(self):
        if self._first is None:
            if self._compute_fn is not None:
                self._first = DerivativeTensor(self, 1)
                return self._first
            else: # otherwise, default to zero
                return DerivativeTensor(self, 1, 
                        torch.zeros(self.val.shape + (self.d,), 
                                dtype=self.val.dtype, 
                                device=self.val.device))
        return self._first
    
    @property
    def second(self):
        if self._second is None:
            if self._compute_fn is not None:
                self._second = DerivativeTensor(self, 2)
                return self._second
            else: # otherwise, default to zero
                return DerivativeTensor(self, 2,
                        torch.zeros(self.val.shape + (self.d, self.d), 
                                dtype=self.val.dtype, 
                                device=self.val.device))
        return self._second
    
    @property
    def third(self):
        if self._third is None:
            if self._compute_fn is not None:
                self._third = DerivativeTensor(self, 3)
                return self._third
            else: # otherwise, default to zero
                return DerivativeTensor(self, 3,
                        torch.zeros(self.val.shape + (self.d, self.d, self.d), 
                                dtype=self.val.dtype, 
                                device=self.val.device))
        return self._third
    
    @property
    def shape(self):
        return self.val.shape
    
    @property
    def dtype(self):
        return self.val.dtype
    
    @property
    def device(self):
        return self.val.device
    
    def set_order(self, order: int):
        assert order >= 0 and order <= 3, f"order should be an integer >=0 and <=3, but got order=: {order}"
        self.order = order

    def compute_up_to(self, order: int):
        """Compute up to a specified order"""
        old_order = self.order
        self.order = order
        if order >= 1:
            self.first[...]
        if order >= 2:
            self.second[...]
        if order >= 3:
            self.third[...]
        return self

    def __add__(self, other):
        if isinstance(other, (int, float)):
            def compute_add(order, new_var, idx=None):
                if order == 1:
                    if idx is None:  # full computation
                        return self.first[...]
                    return self.first[idx]  # partial computation
                if order == 2:
                    if idx is None:  # full computation
                        return self.second[...]
                    return self.second[idx]  # partial computation
                if order == 3:
                    if idx is None:  # full computation
                        return self.third[...]
                    return self.third[idx]  # partial computation
                else:
                    raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
            return TaylorVar(self.val + other, order=self.order, compute_fn=compute_add, input_dim=self.d)
        
        if not isinstance(other, TaylorVar):  # constant tensor
            other = TaylorVar(other, input_dim=self.d)

        def compute_add(order, new_var, idx=None):
            if order == 1:
                if idx is None:  # full computation
                    return self.first[...] + other.first[...]
                return self.first[idx] + other.first[idx]  # partial computation
            if order == 2:
                if idx is None:  # full computation
                    return self.second[...] + other.second[...]
                return self.second[idx] + other.second[idx]  # partial computation
            if order == 3:
                if idx is None:  # full computation
                    return self.third[...] + other.third[...]
                return self.third[idx] + other.third[idx]  # partial computation
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
        
        return TaylorVar(self.val + other.val, order=max(self.order, other.order), 
                        compute_fn=compute_add, input_dim=self.d)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            # scalar multiplication
            def compute_mul_scalar(order, new_var, idx=None):
                if order == 1:
                    if idx is None:  # full computation
                        return self.first[...] * other
                    return self.first[idx] * other  # partial computation
                
                if order == 2:
                    if idx is None:  # full computation
                        return self.second[...] * other
                    return self.second[idx] * other  # partial computation
                
                if order == 3:
                    if idx is None:  # full computation
                        return self.third[...] * other
                    return self.third[idx] * other  # partial computation
                else:
                    raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
                
            return TaylorVar(
                self.val * other,
                order=self.order,
                compute_fn=compute_mul_scalar,
                input_dim=self.d
            )
        
        if not isinstance(other, TaylorVar):
            other = TaylorVar(other, input_dim=self.d)
        
        
        out_val = self.val * other.val
        
        def compute_mul(order, new_var, idx=None):
            all_slice, idx1, trans_idx = process_idx(idx)
            
            if order == 1:
                first_new = (self.first[... if idx1 is None else idx1] * expand_val(other.val, 1) + 
                        expand_val(self.val, 1) * other.first[... if idx1 is None else idx1])
                if not all_slice:
                    
                    first_new=first_new[(...,)+trans_idx]
                return first_new
                    
            if order == 2:
                second_new = (self.second[... if idx1 is None else idx1] * expand_val(other.val, 2) + 
                        expand_val(self.val, 2) * other.second[... if idx1 is None else idx1] +
                        self.first[... if idx1 is None else idx1[0]].unsqueeze(-1) * 
                        other.first[... if idx1 is None else idx1[1]].unsqueeze(-2) +
                        self.first[... if idx1 is None else idx1[1]].unsqueeze(-2) * 
                        other.first[... if idx1 is None else idx1[0]].unsqueeze(-1))
                        # torch.einsum('...i,...j->...ij', 
                        #              self.first[... if idx1 is None else idx1[0]],
                        #              other.first[... if idx1 is None else idx1[1]]) +
                        # torch.einsum('...j,...i->...ij', 
                        #              self.first[... if idx1 is None else idx1[1]],
                        #              other.first[... if idx1 is None else idx1[0]]))
                if not all_slice:
                    second_new=second_new[(...,)+trans_idx]
                return second_new
            if order == 3:
                third_new = (self.third[... if idx1 is None else idx1] * expand_val(other.val, 3) +
                        expand_val(self.val, 3) * other.third[... if idx1 is None else idx1] +
                        self.second[... if idx1 is None else idx1[0:2]].unsqueeze(-1) * 
                            other.first[... if idx1 is None else idx1[2]].unsqueeze(-2).unsqueeze(-2) +
                        self.second[... if idx1 is None else idx1[1:3]].unsqueeze(-3) * 
                            other.first[... if idx1 is None else idx1[0]].unsqueeze(-1).unsqueeze(-1) +
                        self.second[... if idx1 is None else (idx1[0],idx1[2])].unsqueeze(-2) * 
                            other.first[... if idx1 is None else idx1[1]].unsqueeze(-1).unsqueeze(-3) +
                        self.first[... if idx1 is None else idx1[0]].unsqueeze(-1).unsqueeze(-1) * 
                            other.second[... if idx1 is None else idx1[1:3]].unsqueeze(-3) +
                        self.first[... if idx1 is None else idx1[1]].unsqueeze(-1).unsqueeze(-3) * 
                            other.second[... if idx1 is None else (idx1[0],idx1[2])].unsqueeze(-2)+
                        self.first[... if idx1 is None else idx1[2]].unsqueeze(-2).unsqueeze(-2) * 
                            other.second[... if idx1 is None else idx1[0:2]].unsqueeze(-1)
                )
                        
                        # torch.einsum('...ij,...k->...ijk', 
                        #             self.second[... if idx1 is None else idx1[0:2]], 
                        #             other.first[... if idx1 is None else idx1[2]]) +
                        # torch.einsum('...jk,...i->...ijk', 
                        #             self.second[... if idx1 is None else idx1[1:3]], 
                        #             other.first[... if idx1 is None else idx1[0]]) +
                        # torch.einsum('...ik,...j->...ijk', 
                        #             self.second[... if idx1 is None else (idx1[0],idx1[2])], 
                        #             other.first[... if idx1 is None else idx1[1]]) +
                        # torch.einsum('...i,...jk->...ijk', 
                        #             self.first[... if idx1 is None else idx1[0]], 
                        #             other.second[... if idx1 is None else idx1[1:3]]) +
                        # torch.einsum('...j,...ik->...ijk', 
                        #             self.first[... if idx1 is None else idx1[1]], 
                        #             other.second[... if idx1 is None else (idx1[0],idx1[2])]) +
                        # torch.einsum('...k,...ij->...ijk', 
                        #             self.first[... if idx1 is None else idx1[2]], 
                        #             other.second[... if idx1 is None else idx1[0:2]]))
                
                if not all_slice:
                    third_new=third_new[(...,)+trans_idx]
                return third_new
            else:   
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")

        return TaylorVar(out_val, order=max(self.order, other.order), compute_fn=compute_mul, input_dim=self.d)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        """
        Taylor propagation rule for element-wise subtraction.
        Subtraction is equivalent to adding a negative number.
        """
        if isinstance(other, (int, float)):
            # scalar subtraction
            def compute_sub(order, new_var, idx=None):
                if order == 1:
                    if idx is None:  # full computation
                        return self.first[...]
                    return self.first[idx]  # partial computation
                if order == 2:
                    if idx is None:  # full computation
                        return self.second[...]
                    return self.second[idx]  # partial computation
                if order == 3:
                    if idx is None:  # full computation
                        return self.third[...]
                    return self.third[idx]  # partial computation
                else:
                    raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
            return TaylorVar(self.val - other, order=self.order, compute_fn=compute_sub, input_dim=self.d)
        
        if not isinstance(other, TaylorVar):
            other = TaylorVar(other, input_dim=self.d)
        return self + (-1) * other

    def __rsub__(self, other):
        """
        Right subtraction: other - self
        """
        return (-1) * self + other

    def __neg__(self):
        return (-1) * self

    def reshape(self, *shape):
        def compute_reshape(order, new_var, idx=None):
            if order == 1:
                if idx is None:  # full computation
                    return self.first[...].reshape(*shape, self.d)
                first_new = self.first[idx]
                deriv_shape = first_new.shape[self.val.dim():]
                return first_new.reshape(*shape, *deriv_shape)
            if order == 2:
                if idx is None:  # full computation
                    return self.second[...].reshape(*shape, self.d, self.d)
                second_new = self.second[idx]
                deriv_shape = second_new.shape[self.val.dim():]
                return second_new.reshape(*shape, *deriv_shape)
            if order == 3:
                if idx is None:  # full computation
                    return self.third[...].reshape(*shape, self.d, self.d, self.d)
                third_new = self.third[idx]
                deriv_shape = third_new.shape[self.val.dim():]
                return third_new.reshape(*shape, *deriv_shape)
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
        return TaylorVar(self.val.reshape(*shape), order=self.order, 
                        compute_fn=compute_reshape, input_dim=self.d)

    def view(self, *shape):
        def compute_view(order, new_var, idx=None):
            if order == 1:
                if idx is None:  # full computation
                    return self.first[...].view(*shape, self.d)
                first_new = self.first[idx]
                deriv_shape = first_new.shape[self.val.dim():]
                return first_new.view(*shape, *deriv_shape)
            if order == 2:
                if idx is None:  # full computation
                    return self.second[...].view(*shape, self.d, self.d)
                second_new = self.second[idx]
                deriv_shape = second_new.shape[self.val.dim():]
                return second_new.view(*shape, *deriv_shape)
            if order == 3:
                if idx is None:  # full computation
                    return self.third[...].view(*shape, self.d, self.d, self.d)
                third_new = self.third[idx]
                deriv_shape = third_new.shape[self.val.dim():]
                return third_new.view(*shape, *deriv_shape)
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
        return TaylorVar(self.val.view(*shape), order=self.order, compute_fn=compute_view, input_dim=self.d)

    @staticmethod
    def cat(tensors: List[TaylorVar], dim: int = -1) -> TaylorVar:
        """
        Concatenate multiple TaylorVar along a specified dimension.
        """
        if dim < 0:
            dim = tensors[0].val.dim() + dim
    
        max_order = max(t.order for t in tensors)
        def compute_cat(order, new_var, idx=None):
            if order == 1:
                if idx is None:
                    return torch.cat([t.first[...] for t in tensors], dim=dim)
                first_list = [t.first[idx] for t in tensors]
                return torch.cat(first_list, dim=dim)
            if order == 2:
                if idx is None:
                    return torch.cat([t.second[...] for t in tensors], dim=dim)
                second_list = [t.second[idx] for t in tensors]
                return torch.cat(second_list, dim=dim)
            if order == 3:
                if idx is None:
                    return torch.cat([t.third[...] for t in tensors], dim=dim)
                third_list = [t.third[idx] for t in tensors]
                return torch.cat(third_list, dim=dim)
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
        return TaylorVar(
            torch.cat([t.val for t in tensors], dim=dim),
            order=max_order,
            compute_fn=compute_cat,
            input_dim=tensors[0].d
        )

    @staticmethod
    def stack(tensors: List[TaylorVar], dim: int = 0) -> TaylorVar:
        """
        Stack multiple TaylorVar along a new dimension.
        """
        if dim < 0:
            dim = tensors[0].val.dim() + 1 + dim  # +1 because stack adds one dimension
    
        max_order = max(t.order for t in tensors)
        def compute_stack(order, new_var, idx=None):
            if order == 1:
                if idx is None:
                    return torch.stack([t.first[...] for t in tensors], dim=dim)
                first_list = [t.first[idx] for t in tensors]
                return torch.stack(first_list, dim=dim)
            if order == 2:
                if idx is None:
                    return torch.stack([t.second[...] for t in tensors], dim=dim)
                second_list = [t.second[idx] for t in tensors]
                return torch.stack(second_list, dim=dim)
            if order == 3:
                if idx is None:
                    return torch.stack([t.third[...] for t in tensors], dim=dim)
                third_list = [t.third[idx] for t in tensors]
                return torch.stack(third_list, dim=dim)
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
        return TaylorVar(
            torch.stack([t.val for t in tensors], dim=dim),
            order=max_order,
            compute_fn=compute_stack,
            input_dim=tensors[0].d
        )

    def squeeze(self, dim=None):
        """
        Remove size-1 dimensions.
        """
        if dim is not None and dim < 0:
            dim = self.val.dim() + dim
        
        def compute_squeeze(order, new_var, idx=None):
            if order == 1:
                if idx is None:
                    return self.first[...].squeeze(dim) if dim is not None else self.first[...].squeeze()
                first_new = self.first[idx]
                if dim is not None and dim < self.val.dim():
                    return first_new.squeeze(dim)
                return first_new
            if order == 2:
                if idx is None:
                    return self.second[...].squeeze(dim) if dim is not None else self.second[...].squeeze()
                second_new = self.second[idx]
                if dim is not None and dim < self.val.dim():
                    return second_new.squeeze(dim)
                return second_new
            if order == 3:
                if idx is None:
                    return self.third[...].squeeze(dim) if dim is not None else self.third[...].squeeze()
                third_new = self.third[idx]
                if dim is not None and dim < self.val.dim():
                    return third_new.squeeze(dim)
                return third_new
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
        return TaylorVar(
            self.val.squeeze(dim) if dim is not None else self.val.squeeze(),
            order=self.order,
            compute_fn=compute_squeeze,
            input_dim=self.d
        )

    def unsqueeze(self, dim):
        """
        Insert a size-1 dimension at the specified position.
        """
        if dim < 0:
            dim = self.val.dim() + 1 + dim
        
        def compute_unsqueeze(order, new_var, idx=None):
            if order == 1:
                if idx is None:
                    return self.first[...].unsqueeze(dim)
                first_new = self.first[idx]
                if dim < self.val.dim():
                    return first_new.unsqueeze(dim)
                return first_new
            if order == 2:
                if idx is None:
                    return self.second[...].unsqueeze(dim)
                second_new = self.second[idx]
                if dim < self.val.dim():
                    return second_new.unsqueeze(dim)
                return second_new
            if order == 3:
                if idx is None:
                    return self.third[...].unsqueeze(dim)
                third_new = self.third[idx]
                if dim < self.val.dim():
                    return third_new.unsqueeze(dim)
                return third_new
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
        return TaylorVar(
            self.val.unsqueeze(dim),
            order=self.order,
            compute_fn=compute_unsqueeze,
            input_dim=self.d
        )

    def flatten(self, start_dim=0, end_dim=-1):
        """
        Flatten the dimensions from start_dim to end_dim.
        """
        if end_dim < 0:
            end_dim = self.val.dim() + end_dim
        
        def compute_flatten(order, new_var, idx=None):
            if order == 1:
                if idx is None:
                    return self.first[...].flatten(start_dim, end_dim)
                first_new = self.first[idx]
                if end_dim < self.val.dim():
                    return first_new.flatten(start_dim, end_dim)
                return first_new
            if order == 2:
                if idx is None:
                    return self.second[...].flatten(start_dim, end_dim)
                second_new = self.second[idx]
                if end_dim < self.val.dim():
                    return second_new.flatten(start_dim, end_dim)
                return second_new
            if order == 3:
                if idx is None:
                    return self.third[...].flatten(start_dim, end_dim)
                third_new = self.third[idx]
                if end_dim < self.val.dim():
                    return third_new.flatten(start_dim, end_dim)
                return third_new
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")
        return TaylorVar(
            self.val.flatten(start_dim, end_dim),
            order=self.order,
            compute_fn=compute_flatten,
            input_dim=self.d
        )

    # linear transformation: z = Wz + b
    def linear(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        linear: R^{in_features} -> R^{out_features}.
        shape: weight: (out_features, in_features)
              bias:   (out_features,)
              self.val: (batch_size, in_features)
        
        For val, first, second, third, the following formulas are used:
        z_val = weight * val^T + bias
        z_first = weight * x.first
        z_second = weight * x.second
        z_third = weight * x.third
        """
        val_in = self.val
        out_val = val_in @ weight.T  # (..., out_features)
        if bias is not None:
            out_val = out_val + bias
        
        def compute_linear(order, new_var, idx=None):
            all_slice, idx1, trans_idx = process_idx(idx)
            
            if order == 1:
                # first_new = torch.einsum('...im,ji->...jm', 
                #                        self.first[... if idx1 is None else idx1], weight)
                A = self.first[... if idx1 is None else idx1]  # (..., in_features, d)
                first_new = torch.matmul(weight, A)  # (..., out_features, d)
                if not all_slice:
                    first_new = first_new[(...,) + trans_idx]
                return first_new
            
            if order == 2:
                # second_new = torch.einsum('...imn,ji->...jmn', 
                #                         self.second[... if idx1 is None else idx1], weight)
                A = self.second[... if idx1 is None else idx1]  # (..., in_features, d, d)
                sh = A.shape  # (*, in_features, d, d)
                A_reshaped = A.reshape(*sh[:-3], sh[-3], sh[-2]*sh[-1])  # reshape 为 (..., in_features, d*d)
                temp = torch.matmul(weight, A_reshaped)  # 结果 (..., out_features, d*d)
                second_new = temp.reshape(*sh[:-3], weight.shape[0], sh[-2], sh[-1])
                if not all_slice:
                    second_new = second_new[(...,) + trans_idx]
                return second_new
            
            if order == 3:
                
                # third_new = torch.einsum('...imnl,ji->...jmnl', 
                #                        self.third[... if idx1 is None else idx1], weight)
                A = self.third[... if idx1 is None else idx1]  # (..., in_features, d, d, d)
                sh = A.shape  # (*, in_features, d, d, d)
                A_reshaped = A.reshape(*sh[:-4], sh[-4], sh[-3]*sh[-2]*sh[-1])  # (..., in_features, d*d*d)
                temp = torch.matmul(weight, A_reshaped)  # (..., out_features, d*d*d)
                third_new = temp.reshape(*sh[:-4], weight.shape[0], sh[-3], sh[-2], sh[-1])
                
                if not all_slice:
                    third_new = third_new[(...,) + trans_idx]
                return third_new
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")

        return TaylorVar(out_val, order=self.order, compute_fn=compute_linear, input_dim=self.d)

    def elementwise_fn(
        self, 
        fn: Callable[[torch.Tensor], torch.Tensor],
        fn_prime: Callable[[torch.Tensor], torch.Tensor],
        fn_double_prime: Callable[[torch.Tensor], torch.Tensor],
        fn_triple_prime: Callable[[torch.Tensor], torch.Tensor]
    ):

        
        val_new = fn(self.val)
        
        def compute_elementwise_fn(order, new_var, idx=None):
            all_slice, idx1, trans_idx = process_idx(idx)
            
            if order == 1:
                first_new = fn_prime(self.val).unsqueeze(-1) * self.first[... if idx1 is None else idx1]
                if not all_slice:
                    first_new = first_new[(...,) + trans_idx]
                return first_new
            
            if order == 2:
                prime = fn_prime(self.val)
                double_prime = fn_double_prime(self.val)
                
                # second_new = (
                #     prime.unsqueeze(-1).unsqueeze(-1) * self.second[... if idx1 is None else idx1] +
                #     double_prime.unsqueeze(-1).unsqueeze(-1) * 
                #     torch.einsum('...i,...j->...ij', 
                #                self.first[... if idx1 is None else idx1[0]], 
                #                self.first[... if idx1 is None else idx1[1]])
                # )
                term_outer = self.first[... if idx1 is None else idx1[0]].unsqueeze(-1) * \
                             self.first[... if idx1 is None else idx1[1]].unsqueeze(-2)
                second_new = (prime.unsqueeze(-1).unsqueeze(-1) * self.second[... if idx1 is None else idx1] +
                              double_prime.unsqueeze(-1).unsqueeze(-1) * term_outer)
                
                if not all_slice:
                    second_new = second_new[(...,) + trans_idx]
                return second_new
            
            if order == 3:
                prime = fn_prime(self.val)
                double_prime = fn_double_prime(self.val)
                triple_prime = fn_triple_prime(self.val)
                
                # third_new = (
                #     prime.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 
                #     self.third[... if idx1 is None else idx1] +
                #     double_prime.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (
                #         torch.einsum('...ij,...k->...ijk', 
                #                    self.second[... if idx1 is None else idx1[0:2]], 
                #                    self.first[... if idx1 is None else idx1[2]]) +
                #         torch.einsum('...ik,...j->...ijk', 
                #                    self.second[... if idx1 is None else (idx1[0], idx1[2])], 
                #                    self.first[... if idx1 is None else idx1[1]]) +
                #         torch.einsum('...jk,...i->...ijk', 
                #                    self.second[... if idx1 is None else idx1[1:3]], 
                #                    self.first[... if idx1 is None else idx1[0]])
                #     ) +
                #     triple_prime.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * 
                #     torch.einsum('...i,...j,...k->...ijk', 
                #                self.first[... if idx1 is None else idx1[0]], 
                #                self.first[... if idx1 is None else idx1[1]], 
                #                self.first[... if idx1 is None else idx1[2]])
                # )
                term1 = self.second[... if idx1 is None else idx1[0:2]].unsqueeze(-1) * \
                        self.first[... if idx1 is None else idx1[2]].unsqueeze(-2).unsqueeze(-2)
                term2 = self.second[... if idx1 is None else idx1[1:3]].unsqueeze(-3) * \
                        self.first[... if idx1 is None else idx1[0]].unsqueeze(-1).unsqueeze(-1)
                term3 = self.second[... if idx1 is None else (idx1[0], idx1[2])].unsqueeze(-2) * \
                        self.first[... if idx1 is None else idx1[1]].unsqueeze(-1).unsqueeze(-3)
                
                outer_first = (self.first[... if idx1 is None else idx1[0]].unsqueeze(-1).unsqueeze(-1) *
                               self.first[... if idx1 is None else idx1[1]].unsqueeze(-2).unsqueeze(-1) *
                               self.first[... if idx1 is None else idx1[2]].unsqueeze(-2).unsqueeze(-2))
                
                third_new = (prime.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.third[... if idx1 is None else idx1] +
                             double_prime.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * (term1 + term2 + term3) +
                             triple_prime.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * outer_first)
                
                if not all_slice:
                    third_new = third_new[(...,) + trans_idx]
                return third_new
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")

        return TaylorVar(val_new, order=self.order, compute_fn=compute_elementwise_fn, input_dim=self.d)
        
    def __getitem__(self, idx):
        """
        Support index operation.
        Keep the derivative dimension unchanged, only index the dimension of the value tensor.
        Not support advanced indexing, boolean indexing, etc.
        """
        new_val = self.val[idx] 
        if not isinstance(idx, tuple):
            idx = (idx,)
        
        
        def normalize_idx(idx, ndim):
            if Ellipsis not in idx:
                missing_dims = self.val.ndim - len(idx)
                return idx + (slice(None),) * missing_dims
            ellipsis_idx = idx.index(Ellipsis)
            n_missing = ndim - (len(idx) - 1)  # -1 because ... is replaced
            return idx[:ellipsis_idx] + (slice(None),) * n_missing + idx[ellipsis_idx+1:]
        idx_normalized = normalize_idx(idx, self.val.ndim)
        def compute_getitem(order, new_var, deriv_idx=None):
            if order == 1:
                if deriv_idx is None:  # full computation
                    return self.first[...][idx_normalized + (...,)]  
                first_new = self.first[deriv_idx]
                return first_new[idx_normalized]  
            if order == 2:
                if deriv_idx is None:  # full computation
                    return self.second[...][idx_normalized + (...,)]
                second_new = self.second[deriv_idx]
                return second_new[idx_normalized]
            
            if order == 3:
                if deriv_idx is None:  # full computation
                    return self.third[...][idx_normalized + (...,)]
                third_new = self.third[deriv_idx]
                return third_new[idx_normalized]
            
            else:
                raise ValueError(f"order should be an integer >=1 and <=3, but got order=: {order}")

        return TaylorVar(
            new_val, 
            order=self.order,
            compute_fn=compute_getitem,
            input_dim=self.d
        )

# utils functions
def taylor_add(x: TaylorVar, y: TaylorVar) -> TaylorVar:
    return x + y

def taylor_mul(x: TaylorVar, y: TaylorVar) -> TaylorVar:
    return x * y

def taylor_linear(x: TaylorVar, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> TaylorVar:
    return x.linear(weight, bias)


def taylor_activation_wrapper(
    fn: Callable[[torch.Tensor], torch.Tensor],
    fn_prime: Callable[[torch.Tensor], torch.Tensor],
    fn_double_prime: Callable[[torch.Tensor], torch.Tensor],
    fn_triple_prime: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
) -> Callable[[TaylorVar], TaylorVar]:
    def wrapper(x: TaylorVar) -> TaylorVar:
        return x.elementwise_fn(fn, fn_prime, fn_double_prime, fn_triple_prime)
    return wrapper


def get_activation_with_derivatives(activation: Union[str, Callable], order: int = 3):
    """
    Get the activation function and its derivatives.
    """
    if isinstance(activation, str):
        if activation.lower() == 'relu':
            def fn(x): return torch.relu(x)
            def fn_prime(x): return (x > 0).float()
            def fn_double_prime(x): return torch.zeros_like(x)
            def fn_triple_prime(x): return torch.zeros_like(x)
            
        elif activation.lower() == 'tanh':
            def fn(x): return torch.tanh(x)
            def fn_prime(x): return 1 - torch.tanh(x)**2
            def fn_double_prime(x): 
                t = torch.tanh(x)
                return -2 * t * (1 - t**2)
            def fn_triple_prime(x):
                t = torch.tanh(x)
                return -2 * (1 - t**2) * (1 - 3*t**2)
            
        elif activation.lower() == 'sigmoid':
            def fn(x): return torch.sigmoid(x)
            def fn_prime(x): 
                s = torch.sigmoid(x)
                return s * (1 - s)
            def fn_double_prime(x):
                s = torch.sigmoid(x)
                return s * (1 - s) * (1 - 2*s)
            def fn_triple_prime(x):
                s = torch.sigmoid(x)
                return s * (1 - s) * (1 - 6*s + 6*s**2)
        elif activation.lower() == 'swish':
            def fn(x): 
                return x * torch.sigmoid(x)
            def fn_prime(x):
                sig = torch.sigmoid(x)
                return sig + x * sig * (1 - sig)
            def fn_double_prime(x):
                sig = torch.sigmoid(x)
                return sig * (1 - sig) * (2 + x * (1 - 2*sig))
            def fn_triple_prime(x):
                sig = torch.sigmoid(x)
                return sig * (1 - sig) * ( 6*x*sig**2 - 6*(x+1)*sig + x+3)
            
        elif activation.lower() == 'square':
            def fn(x): return x**2
            def fn_prime(x): return 2*x
            def fn_double_prime(x): return 2*torch.ones_like(x)
            def fn_triple_prime(x): return torch.zeros_like(x)
            
        elif activation.lower() == 'cube':
            def fn(x): return x**3
            def fn_prime(x): return 3*x**2
            def fn_double_prime(x): return 6*x
            def fn_triple_prime(x): return 6*torch.ones_like(x)
        
        elif activation.lower() == 'sin':
            def fn(x): return torch.sin(x)
            def fn_prime(x): return torch.cos(x)
            def fn_double_prime(x): return -torch.sin(x)
            def fn_triple_prime(x): return -torch.cos(x)
        
        elif activation.lower() == 'cos':
            def fn(x): return torch.cos(x)
            def fn_prime(x): return -torch.sin(x)
            def fn_double_prime(x): return -torch.cos(x)
            def fn_triple_prime(x): return torch.sin(x)
            
        else:
            raise ValueError(f"Unsupported string of activation: {activation}."
                             "hint: support providing a custom callable elementwise function.")
    else:
        # handle custom function
        fn = activation
        from torch.func import grad
        
        # compute derivatives
        def fn_prime(x):
            return grad(lambda y: fn(y).sum())(x)
            
        def fn_double_prime(x):
            return grad(lambda y: fn_prime(y).sum())(x)
            
        def fn_triple_prime(x):
            return grad(lambda y: fn_double_prime(y).sum())(x)

    return fn, fn_prime, fn_double_prime, fn_triple_prime
