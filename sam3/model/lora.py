import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层
    
    参考论文: https://arxiv.org/abs/2106.09685
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        freeze_original: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.linear = None
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._freeze_original = freeze_original
        self._lora_enabled = True
        
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def set_original_linear(self, linear: nn.Linear):
        """设置原始的线性层"""
        self.linear = linear
        if self._freeze_original:
            for param in self.linear.parameters():
                param.requires_grad_(False)
    
    def enable_lora(self, enabled: bool = True):
        """启用或禁用 LoRA"""
        self._lora_enabled = enabled
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            raise ValueError("LoRALayer: original linear layer not set. Call set_original_linear() first.")
        
        result = self.linear(x)
        
        if self._lora_enabled:
            lora_out = self.lora_B(self.dropout(self.lora_A(x)))
            result = result + lora_out * self.scaling
        
        return result
    
    @property
    def weight(self):
        """代理原始线性层的 weight 属性"""
        return self.linear.weight
    
    @weight.setter
    def weight(self, value):
        """代理原始线性层的 weight 设置"""
        self.linear.weight = value
    
    @property
    def bias(self):
        """代理原始线性层的 bias 属性"""
        return self.linear.bias
    
    @bias.setter
    def bias(self, value):
        """代理原始线性层的 bias 设置"""
        self.linear.bias = value
    
    def __getattr__(self, name):
        """代理原始线性层的其他属性"""
        if name in ['linear', 'lora_A', 'lora_B', 'dropout', '_freeze_original', '_lora_enabled', 
                   'in_features', 'out_features', 'rank', 'alpha', 'scaling']:
            return super().__getattr__(name)
        try:
            return getattr(self.linear, name)
        except AttributeError:
            return super().__getattr__(name)


class LoRAWrapper(nn.Module):
    """
    LoRA 包装器，用于给模型添加 LoRA 适配器
    继承自 nn.Module 以确保所有参数都能被正确处理
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        freeze_original: bool = True,
    ):
        """
        Args:
            model: 原始模型
            rank: LoRA 的秩
            alpha: LoRA 的缩放因子
            dropout: LoRA 的 dropout 率
            target_modules: 要添加 LoRA 的模块名称列表，如果为 None 则自动检测所有 Linear 层
            freeze_original: 是否冻结原始模型的参数
        """
        super().__init__()
        self.model = model
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.freeze_original = freeze_original
        
        self._lora_layers: nn.ModuleDict = nn.ModuleDict()
        self._original_layers: Dict[str, nn.Linear] = {}
        
        self._apply_lora()
    
    def _apply_lora(self):
        """给模型应用 LoRA"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if self.target_modules is None or any(t in name for t in self.target_modules):
                    self._wrap_linear(name, module)
        
        if self.freeze_original:
            for name, param in self.model.named_parameters():
                param.requires_grad_(False)
    
    def _wrap_linear(self, name: str, linear: nn.Linear):
        """包装一个线性层"""
        lora_layer = LoRALayer(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
            freeze_original=self.freeze_original,
        )
        lora_layer.set_original_linear(linear)
        
        self._original_layers[name] = linear
        self._lora_layers[name.replace('.', '_')] = lora_layer
        
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        
        if parent_name:
            parent = self.model.get_submodule(parent_name)
        else:
            parent = self.model
        
        setattr(parent, child_name, lora_layer)
    
    def enable_lora(self, enabled: bool = True):
        """启用或禁用所有 LoRA 层"""
        for lora_layer in self._lora_layers.values():
            lora_layer.enable_lora(enabled)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练的参数（仅 LoRA 参数）"""
        params = []
        for lora_layer in self._lora_layers.values():
            params.extend(lora_layer.lora_A.parameters())
            params.extend(lora_layer.lora_B.parameters())
        return params
    
    def save_lora_weights(self, path: str):
        """保存 LoRA 权重"""
        state_dict = {}
        for name, lora_layer in self._lora_layers.items():
            original_name = name.replace('_', '.')
            state_dict[f"{original_name}.lora_A"] = lora_layer.lora_A.weight.data.cpu()
            state_dict[f"{original_name}.lora_B"] = lora_layer.lora_B.weight.data.cpu()
        torch.save(state_dict, path)
    
    def load_lora_weights(self, path: str):
        """加载 LoRA 权重"""
        state_dict = torch.load(path, map_location="cpu")
        for name, lora_layer in self._lora_layers.items():
            original_name = name.replace('_', '.')
            lora_layer.lora_A.weight.data.copy_(state_dict[f"{original_name}.lora_A"])
            lora_layer.lora_B.weight.data.copy_(state_dict[f"{original_name}.lora_B"])
    
    def merge_lora(self):
        """合并 LoRA 权重到原始线性层（用于推理加速）"""
        for name, lora_layer in self._lora_layers.items():
            original_name = name.replace('_', '.')
            if lora_layer.linear is not None:
                merged_weight = (
                    lora_layer.linear.weight.data +
                    (lora_layer.lora_B.weight.data @ lora_layer.lora_A.weight.data) * lora_layer.scaling
                )
                lora_layer.linear.weight.data.copy_(merged_weight)
                if lora_layer.linear.bias is not None:
                    lora_layer.linear.bias.data.copy_(lora_layer.linear.bias.data)
                
                parent_name = ".".join(original_name.split(".")[:-1])
                child_name = original_name.split(".")[-1]
                
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                
                setattr(parent, child_name, lora_layer.linear)
    
    def forward(self, *args, **kwargs):
        """前向传播，代理到原始模型"""
        return self.model(*args, **kwargs)
