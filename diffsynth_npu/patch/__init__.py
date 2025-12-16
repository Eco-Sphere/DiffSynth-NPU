from diffsynth_npu.patch.base_patch import (
    replace_patch_xfusers_imports,
    replace_patch_initialize_usp,
    replace_patch_torch_ones,
    replace_patch_torch_float64_to_float32,
    replace_patch_tensor_double_to_float32,
)

NPU_PATCH_MAP = {
    "patch_xfusers_imports": replace_patch_xfusers_imports,
    "patch_initialize_usp": replace_patch_initialize_usp,
    "patch_torch_ones": replace_patch_torch_ones,
    "patch_torch_float64_to_float32": replace_patch_torch_float64_to_float32,
    "patch_tensor_double_to_float32": replace_patch_tensor_double_to_float32,
}
