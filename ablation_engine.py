import torch

class CausalAblationEngine:
    def __init__(self, vit_engine):
        self.engine = vit_engine
        self.model = vit_engine.model
        
        # Directly target vit.layers without any 'encoder' checks
        if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'layers'):
            self.layers = self.model.vit.layers
        elif hasattr(self.model, 'layers'):
            self.layers = self.model.layers
        else:
            raise AttributeError("Could not locate 'layers' in model structure.")

    def run_ablation_experiment(self, pixel_values, target_layer, target_head):
        with torch.no_grad():
            base_out = self.model(pixel_values.to(self.engine.device))
            base_logits = base_out.logits if hasattr(base_out, 'logits') else base_out[0]

        # Direct access to the target layer's attention module
        layer_module = self.layers[target_layer].attention

        def hook_fn(module, input, output):
            start_dim = target_head * 64
            end_dim = (target_head + 1) * 64
            
            if isinstance(output, tuple):
                tensor_out = output[0].clone()
                tensor_out[:, :, start_dim:end_dim] = 0.0
                return (tensor_out,) + output[1:]
            else:
                tensor_out = output.clone()
                tensor_out[:, :, start_dim:end_dim] = 0.0
                return tensor_out

        handle = layer_module.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            ablated_out = self.model(pixel_values.to(self.engine.device))
            ablated_logits = ablated_out.logits if hasattr(ablated_out, 'logits') else ablated_out[0]
            
        handle.remove()
        
        logit_diff = torch.norm(base_logits - ablated_logits).item()
        
        return {
            "baseline_logits": base_logits,
            "ablated_logits": ablated_logits,
            "logit_difference": logit_diff
        }
