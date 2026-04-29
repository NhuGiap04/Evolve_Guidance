# Checklist: Why No Logging/Visualization?

## Required Parameters for Logging to Work

Để logging được kích hoạt, **BẮT BUỘC** phải có:

```python
output = pipe(
    prompt=prompt,
    
    # ✓ MUST HAVE: Enable intermediate rewards logging
    intermediate_rewards=True,      # ⚠️ DEFAULT: False
    
    # ✓ MUST HAVE: Reward function (otherwise no steering happens)
    reward_fn=my_reward_function,   # ⚠️ DEFAULT: None
    
    # ✓ SHOULD HAVE: Stein parameters > 0
    num_particles=2,                 # Default: 4
    stein_step=0.05,                 # ⚠️ MUST BE > 0
    stein_loop=1,                    # ⚠️ MUST BE > 0
    stein_kernel="rbf",
    
    # ✓ SHOULD HAVE: Steering range
    steer_start=0,
    steer_end=49,                    # or your num_inference_steps-1
)
```

## Debug Checklist

### Step 1: Check Parameters
```python
# ✓ This should be True
intermediate_rewards=True

# ✓ This should NOT be None
reward_fn is not None

# ✓ These should be > 0
stein_step > 0
stein_loop > 0
```

### Step 2: Check Output Type
```python
# ✓ Should be dictionary
isinstance(output, dict)

# ✓ Should have this key
"intermediate_rewards" in output

# ✓ Should not be empty
len(output["intermediate_rewards"]["prior_score_norm_mean"]) > 0
```

### Step 3: Check Data Keys
```python
# ✓ These keys should exist:
required_keys = [
    "step_indices",
    "timesteps",
    "prior_score_norm_mean",
    "reward_grad_norm_mean",
    "cosine_similarity_mean",
]

for key in required_keys:
    assert key in output["intermediate_rewards"]
```

## Common Issues

### Issue 1: "intermediate_rewards" not in output
**Cause**: `intermediate_rewards=False` (default)
**Fix**: 
```python
output = pipe(..., intermediate_rewards=True)
```

### Issue 2: Empty intermediate_rewards
**Cause**: No steering steps executed
**Reasons**:
- `reward_fn=None` → `use_stein=False`
- `stein_step=0` or `stein_loop=0` → `use_stein=False`
- `steer_start > steer_end` → no steering steps in range

**Fix**: Check all are set:
```python
reward_fn is not None           # ✓
stein_step > 0                  # ✓
stein_loop > 0                  # ✓
steer_start <= steer_end        # ✓
```

### Issue 3: Only some data logged
**Cause**: Logging depends on conditions
- `prior_score_norm_*` → logged if `should_log_rewards and loop_idx == 0`
- `cosine_similarity_*` → logged if `should_log_rewards and loop_idx == 0`
- `reward_scale_*` → logged if `should_log_rewards and loop_idx == 0`

**Fix**: Make sure `loop_idx=0` on first iteration by having `stein_loop >= 1`

## Quick Test

```python
import json
from pathlib import Path

# Run pipeline with logging
output = pipe(
    prompt="a cat",
    intermediate_rewards=True,
    reward_fn=my_reward_fn,
    stein_step=0.05,
    stein_loop=1,
    num_particles=2,
)

# Check if data exists
if "intermediate_rewards" in output:
    data = output["intermediate_rewards"]
    
    # Save for visualization
    with open("test_rewards.json", "w") as f:
        json.dump(data, f)
    
    print(f"✓ Logged {len(data['step_indices'])} steps")
    print(f"  Keys: {list(data.keys())}")
else:
    print("❌ No intermediate_rewards in output!")
    print("   Check: intermediate_rewards=True")
    print("   Check: reward_fn is not None")
    print("   Check: stein_step > 0")
```

## Visualization

```bash
# After saving intermediate_rewards to JSON:
python notebooks/plot_guidance_norms.py test_rewards.json output_dir/
```

This generates:
- `guidance_norms.png` - 6 plots of norms
- `norm_ratio.png` - ratio analysis
- `cosine_similarity.png` - alignment analysis
- `alignment_effectiveness.png` - impact analysis
