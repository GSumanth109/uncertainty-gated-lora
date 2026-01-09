import torch
from transformers import AutoImageProcessor, ResNetForImageClassification
from peft import LoraConfig, get_peft_model, TaskType

class AdaptiveVisionModel:
    """
    The Core Backbone. 
    Auto-detects hardware (NVIDIA CUDA vs Apple MPS vs CPU).
    """
    def __init__(self, model_id: str = "microsoft/resnet-50", device: str = None):
        # 1. HARDWARE AUTO-DETECTION
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
                print("⚡️ HARDWARE DETECTED: NVIDIA GPU (CUDA)")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                print("🍎 HARDWARE DETECTED: Apple Silicon GPU (MPS)")
            else:
                self.device = "cpu"
                print("⚠️ HARDWARE DETECTED: CPU Only (Slow)")
        else:
            self.device = device
            
        # 2. Load Base Model
        print(f"Loading Backbone: {model_id}...")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.base_model = ResNetForImageClassification.from_pretrained(model_id)
        
        # 3. Define LoRA Config
        self.peft_config = LoraConfig(
            task_type=TaskType.IMAGE_CLASSIFICATION, 
            inference_mode=True, 
            r=16,
            lora_alpha=32, 
            lora_dropout=0.1,
            target_modules=["convolution"]
        )
        
        # 4. Inject LoRA Wrappers
        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.to(self.device)
        print("Model Loaded & Ready.")

    def add_adapter(self, adapter_name: str):
        self.model.add_adapter(adapter_name, self.peft_config)
    
    def switch_adapter(self, adapter_name: str):
        self.model.set_adapter(adapter_name)

    def predict(self, image_tensor: torch.Tensor):
        with torch.no_grad():
            return self.model(image_tensor.to(self.device))
