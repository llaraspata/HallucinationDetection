import traceback
import torch.nn as nn
from functools import partial
from torch.utils.hooks import RemovableHandle


class InspectOutputContext:
    def __init__(self, model, module_names, move_to_cpu=False, last_position=False, save_generation=False, save_dir=None):
        self.model = model
        self.module_names = module_names
        self.move_to_cpu = move_to_cpu
        self.last_position = last_position
        self.save_generation = save_generation
        self.save_dir = save_dir
        self.handles = []
        self.catcher = dict()
        self.final_output = None

    def __enter__(self):
        for module_name, module in self.model.named_modules():
            if module_name in self.module_names:
                handle = self.inspect_output(module, self.catcher, module_name, move_to_cpu=self.move_to_cpu,
                                        last_position=self.last_position)
                self.handles.append(handle)
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()

        if exc_type is not None:
            print("An exception occurred:")
            print(f"Type: {exc_type}")
            print(f"Value: {exc_val}")
            print("Traceback:")
            traceback.print_tb(exc_tb)
            return False
        return True
    

    def inspect_output(self, module: nn.Module, catcher: dict, module_name, move_to_cpu, last_position=False) -> RemovableHandle:
        hook_instance = partial(self.inspect_hook, catcher=catcher, module_name=module_name, move_to_cpu=move_to_cpu,
                                last_position=last_position)
        handle = module.register_forward_hook(hook_instance)
        return handle
    

    def inspect_hook(self, module: nn.Module, inputs, outputs, catcher: dict, module_name, move_to_cpu, last_position=False):
        if last_position:
            if type(outputs) is tuple:
                catcher[module_name] = outputs[0][:, -1]
            else:
                catcher[module_name] = outputs[:, -1]
            if move_to_cpu:
                catcher[module_name] = catcher[module_name].cpu()
        else:
            if type(outputs) is tuple:
                catcher[module_name] = outputs[0]
            else:
                catcher[module_name] = outputs
            if move_to_cpu:
                catcher[module_name] = catcher[module_name].cpu()
        return outputs
