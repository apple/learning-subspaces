#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

# This can be used to generate the code for a "SimplexConv{d} / SimplexBN{d}" which is simplex with d endpoints
# We generate the code because it results in a faster implementation than using a ParameterList...
# Although this will probably be fixed in future PyTorch versions.
if __name__ == "__main__":
    with open("modules_gen.py", "a") as f:

        for n in range(2, 10):

            f.write(f"class SimplexConv{n}(SubspaceConv):\n")
            f.write("    def __init__(self, *args, **kwargs):\n")
            f.write("        super().__init__(*args, **kwargs)\n")
            for i in range(1, n):
                f.write(
                    f"        self.weight{i} = nn.Parameter(torch.zeros_like(self.weight))\n"
                )
            f.write("\n")
            f.write("    def initialize(self, initialize_fn):\n")
            for i in range(1, n):
                f.write(f"        initialize_fn(self.weight{i})\n")
            f.write("\n")
            f.write("    def get_weight(self):\n")
            f.write(f"        mult = 1 - \\\n")
            for i in range(1, n - 1):
                f.write(f"            self.t{i} - \\\n")
            f.write(f"            self.t{n - 1} \n")
            f.write(f"        w = mult * self.weight + \\\n")
            for i in range(1, n - 1):
                f.write(f"            self.t{i} * self.weight{i} + \\\n")
            f.write(f"            self.t{n - 1} * self.weight{n - 1} \n")
            f.write(f"        return w\n")
            f.write("\n")

            f.write(f"class SimplexBN{n}(SubspaceBN):\n")
            f.write("    def __init__(self, *args, **kwargs):\n")
            f.write("        super().__init__(*args, **kwargs)\n")
            for i in range(1, n):
                f.write(
                    f"        self.weight{i} = nn.Parameter(torch.Tensor(self.num_features))\n"
                )
                f.write(
                    f"        self.bias{i} = nn.Parameter(torch.Tensor(self.num_features))\n"
                )
            for i in range(1, n):
                f.write(f"        torch.nn.init.ones_(self.weight{i})\n")
                f.write(f"        torch.nn.init.zeros_(self.bias{i})\n")
            f.write("\n")
            f.write("    def get_weight(self):\n")
            f.write(f"        mult = 1 - \\\n")
            for i in range(1, n - 1):
                f.write(f"            self.t{i} - \\\n")
            f.write(f"            self.t{n - 1} \n")
            f.write(f"        w = mult * self.weight + \\\n")
            for i in range(1, n - 1):
                f.write(f"            self.t{i} * self.weight{i} + \\\n")
            f.write(f"            self.t{n - 1} * self.weight{n - 1} \n")
            f.write(f"        b = mult * self.bias + \\\n")
            for i in range(1, n - 1):
                f.write(f"            self.t{i} * self.bias{i} + \\\n")
            f.write(f"            self.t{n - 1} * self.bias{n - 1} \n")
            f.write(f"        return w, b\n")
            f.write("\n")
