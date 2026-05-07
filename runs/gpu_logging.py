from collections import defaultdict
from typing import Any, Iterable

import torch


def _bytes_to_mib(value: int) -> float:
    return value / (1024 * 1024)


def _module_device_bytes(module: torch.nn.Module) -> dict[str, int]:
    device_bytes: dict[str, int] = defaultdict(int)
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        device_bytes[str(tensor.device)] += tensor.numel() * tensor.element_size()
    return dict(device_bytes)


def _iter_named_modules(name: str, obj: Any) -> Iterable[tuple[str, torch.nn.Module]]:
    if isinstance(obj, torch.nn.Module):
        yield name, obj
        return

    components = getattr(obj, "components", None)
    if isinstance(components, dict):
        for component_name, component in components.items():
            if isinstance(component, torch.nn.Module):
                yield f"{name}.{component_name}", component
        return

    for attr_name in ("unet", "vae", "text_encoder", "text_encoder_2", "image_encoder"):
        component = getattr(obj, attr_name, None)
        if isinstance(component, torch.nn.Module):
            yield f"{name}.{attr_name}", component


def log_gpu_loads(label: str, *named_objects: tuple[str, Any]) -> None:
    if not torch.cuda.is_available():
        print(f"[GPU] {label}: CUDA is not available.")
        return

    print(f"[GPU] {label}")
    for index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        allocated = torch.cuda.memory_allocated(index)
        reserved = torch.cuda.memory_reserved(index)
        max_allocated = torch.cuda.max_memory_allocated(index)
        name = torch.cuda.get_device_name(index)
        print(
            f"  cuda:{index} {name}: "
            f"allocated={_bytes_to_mib(allocated):.1f} MiB "
            f"reserved={_bytes_to_mib(reserved):.1f} MiB "
            f"max_allocated={_bytes_to_mib(max_allocated):.1f} MiB "
            f"free={_bytes_to_mib(free_bytes):.1f}/{_bytes_to_mib(total_bytes):.1f} MiB"
        )

    rows: list[tuple[str, str, float]] = []
    for object_name, obj in named_objects:
        for module_name, module in _iter_named_modules(object_name, obj):
            for device_name, num_bytes in _module_device_bytes(module).items():
                if device_name.startswith("cuda"):
                    rows.append((module_name, device_name, _bytes_to_mib(num_bytes)))

    if not rows:
        print("  loaded CUDA modules: none detected")
        return

    print("  loaded CUDA modules:")
    for module_name, device_name, mib in sorted(rows):
        print(f"    {module_name}: {mib:.1f} MiB on {device_name}")
