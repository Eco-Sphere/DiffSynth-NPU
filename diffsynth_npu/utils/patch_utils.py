from logger import init_logger
from device_utils import is_npu_available

logger = init_logger(__name__)


def log_replace_info(ori_module, npu_module):
    logger.info(f"Replaced '{ori_module}' with '{npu_module}' successfully")


def replace_npu_patch(npu_patch_map, optim_modules: dict = None):
    if is_npu_available():
        if optim_modules is None:
            optim_modules = list(npu_patch_map.keys())

        for module in optim_modules:
            try:
                if module in npu_patch_map:
                    npu_patch_map[module]()
                else:
                    logger.warning(f"Module {module} not found in npu_patch_map, will not apply npu optimizations")
            except:
                logger.warning(f"Apply {module} failed, will still use original module")
    else:
        logger.warning("NPU is unavailable, will not apply npu optimizations")
