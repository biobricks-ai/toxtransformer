"""
Batch size finder utility for optimal GPU memory usage.

This module provides functionality to automatically determine the optimal batch size
that uses a target percentage of available GPU memory, with safety margins for training.
"""

import gc
import logging
import math
import time
from typing import Optional, Tuple, Union, Any, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast


def get_gpu_memory_info(device: Union[int, torch.device] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage information.
    
    Args:
        device: GPU device (default: current device)
        
    Returns:
        Dictionary with memory info in GB
    """
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    
    # Get memory info
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    cached_memory = torch.cuda.memory_reserved(device)
    free_memory = total_memory - cached_memory
    
    # Convert to GB
    gb = 1024**3
    return {
        'total_gb': total_memory / gb,
        'allocated_gb': allocated_memory / gb,
        'cached_gb': cached_memory / gb,
        'free_gb': free_memory / gb,
        'available_gb': (total_memory - allocated_memory) / gb
    }


def clear_gpu_cache():
    """Clear GPU cache and run garbage collection."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_batch_size(model: nn.Module, 
                   sample_batch: Any,
                   batch_size: int,
                   device: torch.device,
                   use_amp: bool = True,
                   gradient_accumulation: bool = True) -> Tuple[bool, float, Dict[str, float]]:
    """
    Test if a specific batch size fits in GPU memory.
    
    Args:
        model: PyTorch model
        sample_batch: Sample batch from dataloader (will be replicated to target batch size)
        batch_size: Batch size to test
        device: GPU device
        use_amp: Whether to use automatic mixed precision
        gradient_accumulation: Whether to test with gradient computation
        
    Returns:
        Tuple of (success, peak_memory_gb, memory_info)
    """
    model.train() if gradient_accumulation else model.eval()
    initial_memory = get_gpu_memory_info(device)
    peak_memory = initial_memory['allocated_gb']
    
    try:
        # Scale the sample batch to target batch size
        scaled_batch = scale_batch_to_size(sample_batch, batch_size, device)
        
        # Clear cache before test
        clear_gpu_cache()
        
        # Test forward pass
        if gradient_accumulation:
            # Test full training step (forward + backward)
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                if hasattr(model, '__call__') and len(scaled_batch) >= 4:
                    # Assuming SELFIES-style model with (selfies, properties, values, mask)
                    if len(scaled_batch) == 4:
                        selfies, properties, values, mask = scaled_batch
                        logits = model(selfies, properties, values, mask)
                        
                        # Compute loss for backward pass
                        loss = compute_sample_loss(logits, values, mask)
                    else:
                        # Generic case
                        logits = model(*scaled_batch[:-1])
                        loss = compute_sample_loss(logits, scaled_batch[-1])
                else:
                    # Fallback for other model types
                    if isinstance(scaled_batch, (list, tuple)):
                        output = model(*scaled_batch[:-1])
                        loss = compute_sample_loss(output, scaled_batch[-1])
                    else:
                        output = model(scaled_batch)
                        loss = output.mean()  # Dummy loss
                
                # Track peak memory during forward
                current_memory = get_gpu_memory_info(device)
                peak_memory = max(peak_memory, current_memory['allocated_gb'])
                
                # Backward pass
                loss.backward()
                
                # Track peak memory after backward
                current_memory = get_gpu_memory_info(device)
                peak_memory = max(peak_memory, current_memory['allocated_gb'])
                
        else:
            # Test inference only
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_amp):
                    if hasattr(model, '__call__') and len(scaled_batch) >= 4:
                        if len(scaled_batch) == 4:
                            selfies, properties, values, mask = scaled_batch
                            _ = model(selfies, properties, values, mask)
                        else:
                            _ = model(*scaled_batch[:-1])
                    else:
                        if isinstance(scaled_batch, (list, tuple)):
                            _ = model(*scaled_batch[:-1])
                        else:
                            _ = model(scaled_batch)
                    
                    current_memory = get_gpu_memory_info(device)
                    peak_memory = max(peak_memory, current_memory['allocated_gb'])
        
        final_memory = get_gpu_memory_info(device)
        success = True
        
    except torch.cuda.OutOfMemoryError:
        final_memory = get_gpu_memory_info(device)
        success = False
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            final_memory = get_gpu_memory_info(device)
            success = False
        else:
            raise e
    finally:
        # Cleanup
        clear_gpu_cache()
    
    return success, peak_memory, final_memory


def scale_batch_to_size(sample_batch: Any, target_batch_size: int, device: torch.device) -> Any:
    """
    Scale a sample batch to the target batch size.
    
    Args:
        sample_batch: Original batch (tuple/list of tensors)
        target_batch_size: Desired batch size
        device: Target device
        
    Returns:
        Scaled batch
    """
    if isinstance(sample_batch, (list, tuple)):
        current_batch_size = sample_batch[0].size(0)
        scale_factor = max(1, target_batch_size // current_batch_size)
        
        scaled_tensors = []
        for tensor in sample_batch:
            if isinstance(tensor, torch.Tensor):
                # Repeat the tensor to approximate larger batch size
                if scale_factor > 1:
                    tensor = tensor.repeat([scale_factor] + [1] * (tensor.dim() - 1))
                
                # Trim or pad to exact target size
                if tensor.size(0) > target_batch_size:
                    tensor = tensor[:target_batch_size]
                elif tensor.size(0) < target_batch_size:
                    # Pad by repeating last elements
                    needed = target_batch_size - tensor.size(0)
                    last_elements = tensor[-1:].repeat([needed] + [1] * (tensor.dim() - 1))
                    tensor = torch.cat([tensor, last_elements], dim=0)
                
                tensor = tensor.to(device, non_blocking=True)
                scaled_tensors.append(tensor)
            else:
                scaled_tensors.append(tensor)
        
        return type(sample_batch)(scaled_tensors)
    else:
        # Single tensor case
        current_batch_size = sample_batch.size(0)
        scale_factor = max(1, target_batch_size // current_batch_size)
        
        if scale_factor > 1:
            scaled = sample_batch.repeat([scale_factor] + [1] * (sample_batch.dim() - 1))
        else:
            scaled = sample_batch
        
        if scaled.size(0) > target_batch_size:
            scaled = scaled[:target_batch_size]
        elif scaled.size(0) < target_batch_size:
            needed = target_batch_size - scaled.size(0)
            last_elements = scaled[-1:].repeat([needed] + [1] * (scaled.dim() - 1))
            scaled = torch.cat([scaled, last_elements], dim=0)
        
        return scaled.to(device, non_blocking=True)


def compute_sample_loss(logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute a sample loss for testing purposes.
    
    Args:
        logits: Model logits
        targets: Target values
        mask: Optional mask for masked positions
        
    Returns:
        Loss tensor
    """
    if mask is not None:
        # Masked loss computation (like in SELFIES models)
        B, T, C = logits.shape
        logits_f = logits.reshape(-1, C)
        targets_f = targets.reshape(-1)
        mask_f = mask.reshape(-1)
        
        if mask_f.any():
            loss = torch.nn.functional.cross_entropy(logits_f[mask_f], targets_f[mask_f])
        else:
            loss = logits_f.new_zeros(())
    else:
        # Standard loss
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
        
        loss = torch.nn.functional.cross_entropy(logits, targets)
    
    return loss


def find_optimal_batch_size(model: nn.Module,
                          dataset: Union[Dataset, DataLoader],
                          target_memory_percent: float = 75.0,
                          device: Optional[Union[int, torch.device]] = None,
                          min_batch_size: int = 1,
                          max_batch_size: int = 8192,
                          use_amp: bool = True,
                          for_training: bool = True,
                          safety_margin: float = 0.9,
                          log_results: bool = True) -> Tuple[int, Dict[str, Any]]:
    """
    Find optimal batch size that uses target percentage of GPU memory.
    
    Args:
        model: PyTorch model
        dataset: Dataset or DataLoader to sample from
        target_memory_percent: Target GPU memory usage percentage (default: 75%)
        device: GPU device (default: current device)
        min_batch_size: Minimum batch size to test
        max_batch_size: Maximum batch size to test
        use_amp: Whether to use automatic mixed precision
        for_training: Whether to test for training (includes backward pass) or inference
        safety_margin: Safety factor to apply to found batch size (default: 0.9)
        log_results: Whether to log progress and results
        
    Returns:
        Tuple of (optimal_batch_size, info_dict)
        
    Recommended target_memory_percent values:
        - Training: 70-80% (leaves room for gradient computation)
        - Inference: 85-90% (can use more memory safely)
        - Fine-tuning: 65-75% (more conservative for stability)
    """
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, int):
        device = torch.device(f'cuda:{device}')
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    # Get initial memory state
    initial_memory = get_gpu_memory_info(device)
    target_memory_gb = initial_memory['total_gb'] * (target_memory_percent / 100.0)
    
    if log_results:
        mode = "training" if for_training else "inference"
        logging.info(f"🔍 Finding optimal batch size for {mode}")
        logging.info(f"   Target: {target_memory_percent:.1f}% of {initial_memory['total_gb']:.1f}GB = {target_memory_gb:.1f}GB")
        logging.info(f"   Range: {min_batch_size} - {max_batch_size}")
        logging.info(f"   Safety margin: {safety_margin:.1f}")
    
    # Get sample batch
    if isinstance(dataset, DataLoader):
        sample_batch = next(iter(dataset))
        original_batch_size = sample_batch[0].size(0) if isinstance(sample_batch, (list, tuple)) else sample_batch.size(0)
    else:
        # Create temporary dataloader
        temp_loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=False)
        sample_batch = next(iter(temp_loader))
        original_batch_size = sample_batch[0].size(0) if isinstance(sample_batch, (list, tuple)) else sample_batch.size(0)
    
    # Move model to device
    model = model.to(device)
    
    # Binary search for optimal batch size
    left, right = min_batch_size, max_batch_size
    best_batch_size = min_batch_size
    results = []
    
    # Test a few sample points first to get rough estimates
    test_points = [min_batch_size, max_batch_size // 4, max_batch_size // 2, max_batch_size]
    test_points = [bs for bs in test_points if min_batch_size <= bs <= max_batch_size]
    
    if log_results:
        logging.info(f"📊 Testing sample points: {test_points}")
    
    for batch_size in test_points:
        try:
            success, peak_memory, final_memory = test_batch_size(
                model, sample_batch, batch_size, device, use_amp, for_training
            )
            results.append({
                'batch_size': batch_size,
                'success': success,
                'peak_memory_gb': peak_memory,
                'memory_percent': (peak_memory / initial_memory['total_gb']) * 100
            })
            
            if log_results:
                status = "✅" if success else "❌"
                logging.info(f"   {status} BS {batch_size}: {peak_memory:.1f}GB ({results[-1]['memory_percent']:.1f}%)")
            
            if success and peak_memory <= target_memory_gb:
                best_batch_size = max(best_batch_size, batch_size)
            elif not success:
                right = min(right, batch_size - 1)
        
        except Exception as e:
            if log_results:
                logging.warning(f"   ❌ BS {batch_size}: Error - {e}")
            results.append({
                'batch_size': batch_size,
                'success': False,
                'peak_memory_gb': 0,
                'memory_percent': 0,
                'error': str(e)
            })
    
    # Refine with binary search
    if log_results:
        logging.info(f"🎯 Refining search in range [{left}, {right}]")
    
    while left <= right and right - left > 8:  # Stop when range is small
        mid = (left + right) // 2
        
        try:
            success, peak_memory, final_memory = test_batch_size(
                model, sample_batch, mid, device, use_amp, for_training
            )
            
            memory_percent = (peak_memory / initial_memory['total_gb']) * 100
            
            if log_results and len(results) % 3 == 0:  # Log every few tests
                status = "✅" if success else "❌"
                logging.info(f"   {status} BS {mid}: {peak_memory:.1f}GB ({memory_percent:.1f}%)")
            
            if success and peak_memory <= target_memory_gb:
                best_batch_size = max(best_batch_size, mid)
                left = mid + 1
            else:
                right = mid - 1
                
            results.append({
                'batch_size': mid,
                'success': success,
                'peak_memory_gb': peak_memory,
                'memory_percent': memory_percent
            })
            
        except Exception as e:
            if log_results:
                logging.warning(f"   ❌ BS {mid}: Error - {e}")
            right = mid - 1
    
    # Apply safety margin
    safe_batch_size = max(min_batch_size, int(best_batch_size * safety_margin))
    
    # Final validation of recommended batch size
    try:
        success, peak_memory, final_memory = test_batch_size(
            model, sample_batch, safe_batch_size, device, use_amp, for_training
        )
        final_memory_percent = (peak_memory / initial_memory['total_gb']) * 100
    except Exception:
        success = False
        peak_memory = 0
        final_memory_percent = 0
    
    # Compile results
    info = {
        'target_memory_percent': target_memory_percent,
        'target_memory_gb': target_memory_gb,
        'found_batch_size': best_batch_size,
        'recommended_batch_size': safe_batch_size,
        'safety_margin': safety_margin,
        'final_memory_gb': peak_memory,
        'final_memory_percent': final_memory_percent,
        'original_batch_size': original_batch_size,
        'device': str(device),
        'total_gpu_memory_gb': initial_memory['total_gb'],
        'for_training': for_training,
        'use_amp': use_amp,
        'all_tests': results
    }
    
    if log_results:
        logging.info(f"🎉 Batch size optimization complete:")
        logging.info(f"   Found max batch size: {best_batch_size}")
        logging.info(f"   Recommended (with {safety_margin:.1f} safety): {safe_batch_size}")
        logging.info(f"   Final memory usage: {final_memory_percent:.1f}%")
        logging.info(f"   Improvement over original: {safe_batch_size / original_batch_size:.1f}x")
    
    return safe_batch_size, info


# Convenience functions for common use cases
def find_training_batch_size(model: nn.Module, 
                           dataset: Union[Dataset, DataLoader],
                           target_memory_percent: float = 75.0,
                           **kwargs) -> Tuple[int, Dict[str, Any]]:
    """Find optimal batch size for training (includes backward pass)."""
    return find_optimal_batch_size(
        model=model,
        dataset=dataset,
        target_memory_percent=target_memory_percent,
        for_training=True,
        safety_margin=0.9,
        **kwargs
    )


def find_inference_batch_size(model: nn.Module,
                            dataset: Union[Dataset, DataLoader], 
                            target_memory_percent: float = 85.0,
                            **kwargs) -> Tuple[int, Dict[str, Any]]:
    """Find optimal batch size for inference (no backward pass)."""
    return find_optimal_batch_size(
        model=model,
        dataset=dataset,
        target_memory_percent=target_memory_percent,
        for_training=False,
        safety_margin=0.95,
        **kwargs
    )
