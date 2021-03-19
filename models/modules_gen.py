#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import torch
import torch.nn as nn

from models.modules import SubspaceBN
from models.modules import SubspaceConv


class SimplexConv2(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)

    def get_weight(self):
        mult = 1 - self.t1
        w = mult * self.weight + self.t1 * self.weight1
        return w


class SimplexBN2(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)

    def get_weight(self):
        mult = 1 - self.t1
        w = mult * self.weight + self.t1 * self.weight1
        b = mult * self.bias + self.t1 * self.bias1
        return w, b


class SimplexConv3(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight2 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)
        initialize_fn(self.weight2)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2
        w = mult * self.weight + self.t1 * self.weight1 + self.t2 * self.weight2
        return w


class SimplexBN3(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2
        w = mult * self.weight + self.t1 * self.weight1 + self.t2 * self.weight2
        b = mult * self.bias + self.t1 * self.bias1 + self.t2 * self.bias2
        return w, b


class SimplexConv4(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight2 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight3 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)
        initialize_fn(self.weight2)
        initialize_fn(self.weight3)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2 - self.t3
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
        )
        return w


class SimplexBN4(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight3 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias3 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)
        torch.nn.init.ones_(self.weight3)
        torch.nn.init.zeros_(self.bias3)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2 - self.t3
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
        )
        b = (
            mult * self.bias
            + self.t1 * self.bias1
            + self.t2 * self.bias2
            + self.t3 * self.bias3
        )
        return w, b


class SimplexConv5(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight2 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight3 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight4 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)
        initialize_fn(self.weight2)
        initialize_fn(self.weight3)
        initialize_fn(self.weight4)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2 - self.t3 - self.t4
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
        )
        return w


class SimplexBN5(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight3 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias3 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight4 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias4 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)
        torch.nn.init.ones_(self.weight3)
        torch.nn.init.zeros_(self.bias3)
        torch.nn.init.ones_(self.weight4)
        torch.nn.init.zeros_(self.bias4)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2 - self.t3 - self.t4
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
        )
        b = (
            mult * self.bias
            + self.t1 * self.bias1
            + self.t2 * self.bias2
            + self.t3 * self.bias3
            + self.t4 * self.bias4
        )
        return w, b


class SimplexConv6(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight2 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight3 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight4 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight5 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)
        initialize_fn(self.weight2)
        initialize_fn(self.weight3)
        initialize_fn(self.weight4)
        initialize_fn(self.weight5)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2 - self.t3 - self.t4 - self.t5
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
            + self.t5 * self.weight5
        )
        return w


class SimplexBN6(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight3 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias3 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight4 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias4 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight5 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias5 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)
        torch.nn.init.ones_(self.weight3)
        torch.nn.init.zeros_(self.bias3)
        torch.nn.init.ones_(self.weight4)
        torch.nn.init.zeros_(self.bias4)
        torch.nn.init.ones_(self.weight5)
        torch.nn.init.zeros_(self.bias5)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2 - self.t3 - self.t4 - self.t5
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
            + self.t5 * self.weight5
        )
        b = (
            mult * self.bias
            + self.t1 * self.bias1
            + self.t2 * self.bias2
            + self.t3 * self.bias3
            + self.t4 * self.bias4
            + self.t5 * self.bias5
        )
        return w, b


class SimplexConv7(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight2 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight3 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight4 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight5 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight6 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)
        initialize_fn(self.weight2)
        initialize_fn(self.weight3)
        initialize_fn(self.weight4)
        initialize_fn(self.weight5)
        initialize_fn(self.weight6)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2 - self.t3 - self.t4 - self.t5 - self.t6
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
            + self.t5 * self.weight5
            + self.t6 * self.weight6
        )
        return w


class SimplexBN7(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight3 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias3 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight4 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias4 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight5 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias5 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight6 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias6 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)
        torch.nn.init.ones_(self.weight3)
        torch.nn.init.zeros_(self.bias3)
        torch.nn.init.ones_(self.weight4)
        torch.nn.init.zeros_(self.bias4)
        torch.nn.init.ones_(self.weight5)
        torch.nn.init.zeros_(self.bias5)
        torch.nn.init.ones_(self.weight6)
        torch.nn.init.zeros_(self.bias6)

    def get_weight(self):
        mult = 1 - self.t1 - self.t2 - self.t3 - self.t4 - self.t5 - self.t6
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
            + self.t5 * self.weight5
            + self.t6 * self.weight6
        )
        b = (
            mult * self.bias
            + self.t1 * self.bias1
            + self.t2 * self.bias2
            + self.t3 * self.bias3
            + self.t4 * self.bias4
            + self.t5 * self.bias5
            + self.t6 * self.bias6
        )
        return w, b


class SimplexConv8(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight2 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight3 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight4 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight5 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight6 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight7 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)
        initialize_fn(self.weight2)
        initialize_fn(self.weight3)
        initialize_fn(self.weight4)
        initialize_fn(self.weight5)
        initialize_fn(self.weight6)
        initialize_fn(self.weight7)

    def get_weight(self):
        mult = (
            1
            - self.t1
            - self.t2
            - self.t3
            - self.t4
            - self.t5
            - self.t6
            - self.t7
        )
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
            + self.t5 * self.weight5
            + self.t6 * self.weight6
            + self.t7 * self.weight7
        )
        return w


class SimplexBN8(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight3 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias3 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight4 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias4 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight5 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias5 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight6 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias6 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight7 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias7 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)
        torch.nn.init.ones_(self.weight3)
        torch.nn.init.zeros_(self.bias3)
        torch.nn.init.ones_(self.weight4)
        torch.nn.init.zeros_(self.bias4)
        torch.nn.init.ones_(self.weight5)
        torch.nn.init.zeros_(self.bias5)
        torch.nn.init.ones_(self.weight6)
        torch.nn.init.zeros_(self.bias6)
        torch.nn.init.ones_(self.weight7)
        torch.nn.init.zeros_(self.bias7)

    def get_weight(self):
        mult = (
            1
            - self.t1
            - self.t2
            - self.t3
            - self.t4
            - self.t5
            - self.t6
            - self.t7
        )
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
            + self.t5 * self.weight5
            + self.t6 * self.weight6
            + self.t7 * self.weight7
        )
        b = (
            mult * self.bias
            + self.t1 * self.bias1
            + self.t2 * self.bias2
            + self.t3 * self.bias3
            + self.t4 * self.bias4
            + self.t5 * self.bias5
            + self.t6 * self.bias6
            + self.t7 * self.bias7
        )
        return w, b


class SimplexConv9(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight2 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight3 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight4 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight5 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight6 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight7 = nn.Parameter(torch.zeros_like(self.weight))
        self.weight8 = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, initialize_fn):
        initialize_fn(self.weight1)
        initialize_fn(self.weight2)
        initialize_fn(self.weight3)
        initialize_fn(self.weight4)
        initialize_fn(self.weight5)
        initialize_fn(self.weight6)
        initialize_fn(self.weight7)
        initialize_fn(self.weight8)

    def get_weight(self):
        mult = (
            1
            - self.t1
            - self.t2
            - self.t3
            - self.t4
            - self.t5
            - self.t6
            - self.t7
            - self.t8
        )
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
            + self.t5 * self.weight5
            + self.t6 * self.weight6
            + self.t7 * self.weight7
            + self.t8 * self.weight8
        )
        return w


class SimplexBN9(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight2 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias2 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight3 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias3 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight4 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias4 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight5 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias5 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight6 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias6 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight7 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias7 = nn.Parameter(torch.Tensor(self.num_features))
        self.weight8 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias8 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        torch.nn.init.ones_(self.weight2)
        torch.nn.init.zeros_(self.bias2)
        torch.nn.init.ones_(self.weight3)
        torch.nn.init.zeros_(self.bias3)
        torch.nn.init.ones_(self.weight4)
        torch.nn.init.zeros_(self.bias4)
        torch.nn.init.ones_(self.weight5)
        torch.nn.init.zeros_(self.bias5)
        torch.nn.init.ones_(self.weight6)
        torch.nn.init.zeros_(self.bias6)
        torch.nn.init.ones_(self.weight7)
        torch.nn.init.zeros_(self.bias7)
        torch.nn.init.ones_(self.weight8)
        torch.nn.init.zeros_(self.bias8)

    def get_weight(self):
        mult = (
            1
            - self.t1
            - self.t2
            - self.t3
            - self.t4
            - self.t5
            - self.t6
            - self.t7
            - self.t8
        )
        w = (
            mult * self.weight
            + self.t1 * self.weight1
            + self.t2 * self.weight2
            + self.t3 * self.weight3
            + self.t4 * self.weight4
            + self.t5 * self.weight5
            + self.t6 * self.weight6
            + self.t7 * self.weight7
            + self.t8 * self.weight8
        )
        b = (
            mult * self.bias
            + self.t1 * self.bias1
            + self.t2 * self.bias2
            + self.t3 * self.bias3
            + self.t4 * self.bias4
            + self.t5 * self.bias5
            + self.t6 * self.bias6
            + self.t7 * self.bias7
            + self.t8 * self.bias8
        )
        return w, b
