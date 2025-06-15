This is supplemental code for the poster "Derivatives of Spherical Harmonics" presented at High-Performance Graphics 2025:

https://momentsingraphics.de/HPGPoster2025.html

The class SHCodeGeneration in sh_code_generation.py can generate well-optimized code in various languages that evaluates the spherical harmonics basis and its derivatives of any order. Here is an example of its use:
```
from sh_code_generation import SHCodeGeneration
print(SHCodeGeneration("eval_sh_0_2", 2, 1, False, False, False, "sloan", "glsl").generate())
```
