# Why No Logging? - Full Debugging Guide

## TL;DR - Quick Fix

**Logging không hoạt động vì thiếu 3 điều này:**

```python
output = pipe(
    prompt="a photo",
    intermediate_rewards=True,      # ← ENABLE LOGGING (default: False)
    reward_fn=your_reward_function, # ← YOUR REWARD MODEL
    stein_step=0.05,               # ← MUST BE > 0
    stein_loop=1,                  # ← MUST BE > 0
)
```

---

## Full Explanation

### How Logging Works

```
Pipeline runs...
   ↓
for each denoising step:
   ↓
   if is_steered_step (reward guidance is active):
      ↓
      if should_log_rewards (intermediate_rewards=True):
         ↓
         Log: prior_score_norm, reward_grad_norm, cosine_similarity, etc.
         ↓
         Collect in intermediate_rewards_data
   ↓
At end:
   ↓
   if intermediate_rewards=True:
      return (images, intermediate_rewards_data)
   else:
      return (images,)
```

### Conditions for Steering (and Logging)

```python
use_stein = (reward_fn is not None) and (stein_loop > 0) and (stein_step > 0)
is_steered_step = use_stein and (steer_start <= step_idx <= steer_end)
should_log = (intermediate_rewards=True) and is_steered_step
```

**If ANY of these is False, NO LOGGING happens:**
- `reward_fn is None` → steering disabled
- `stein_loop == 0` → steering disabled
- `stein_step == 0` → steering disabled
- `intermediate_rewards == False` → logging disabled
- Outside steer_start/end range → no steering at this step

---

## Added Debug Features

### 1. Debug Output in Pipeline

When you run the pipeline, you'll now see:

```
[DEBUG] Setup: use_stein=False, intermediate_rewards=True, 
        reward_fn_exists=True, stein_step=0.0, stein_loop=1
                                         ↑ problem here
```

If you see `use_stein=False`, that's why nothing is logged!

```
[DEBUG] Logging rewards at step 5/49 (t=900)
[DEBUG] Logging rewards at step 10/49 (t=800)
...
[DEBUG] Returning intermediate_rewards with 20 logged steps
```

### 2. Debug Scripts

**File**: `test_logging.py`
```bash
python test_logging.py
```
This tests if everything is set up correctly and saves `test_intermediate_rewards.json`

**File**: `example_with_logging.py`
```bash
python example_with_logging.py
```
This shows the minimal correct setup and explains each parameter

---

## Common Issues and Solutions

### Issue: `use_stein=False` but I set reward_fn

**Check**:
```python
# These must ALL be true:
reward_fn is not None         # ✓ You provided a function
stein_step > 0                # ✓ Usually 0.05
stein_loop > 0                # ✓ Usually 1-3
```

**Fix if not**:
```python
output = pipe(
    ...,
    reward_fn=your_reward_function,  # Make sure not None!
    stein_step=0.05,                 # Not 0!
    stein_loop=1,                    # Not 0!
)
```

### Issue: Got output but no "intermediate_rewards"

**Cause**: `intermediate_rewards=False` (default)

**Fix**:
```python
output = pipe(..., intermediate_rewards=True)
```

### Issue: Got "intermediate_rewards" but empty

**Check**:
```python
len(output["intermediate_rewards"]["step_indices"])  # Should be > 0
```

**Cause**: No steering happened (check use_stein conditions above)

**Fix**: Use the debug output to see what's False:
```
[DEBUG] Setup: use_stein=?, intermediate_rewards=?, reward_fn_exists=?, ...
```

### Issue: Output is just images, not a dict

**Check**:
```python
# Are you returning a dict?
isinstance(output, dict)
```

**Cause**: Either:
1. `intermediate_rewards=False` → returns just images
2. `return_dict=False` → returns tuple instead of dict

**Fix**:
```python
# Make sure both are True:
output = pipe(..., intermediate_rewards=True, return_dict=True)

# Output will be: {"images": [...], "intermediate_rewards": {...}}
```

---

## Complete Working Example

```python
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from my_project.scorers import MyRewardModel

# 1. Load pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

# 2. Create reward function
scorer = MyRewardModel(device="cuda")
def reward_fn(images, prompts):
    scores = scorer(images, prompts)
    return torch.tensor(scores)

# 3. Run with ALL required settings
output = pipe(
    prompt="a beautiful photo",
    
    # ✓ Logging enabled
    intermediate_rewards=True,
    
    # ✓ Reward function provided
    reward_fn=reward_fn,
    
    # ✓ Steering enabled
    num_particles=2,
    stein_step=0.05,         # Not 0!
    stein_loop=1,            # Not 0!
    reward_guidance_rho=0.05,
    steer_start=0,
    steer_end=49,
    
    # Standard settings
    num_inference_steps=50,
    guidance_scale=7.5,
)

# 4. Extract data
if "intermediate_rewards" in output:
    rewards = output["intermediate_rewards"]
    print(f"Logged {len(rewards['step_indices'])} steps ✓")
    
    # Save
    with open("rewards.json", "w") as f:
        json.dump(rewards, f)
    
    # Visualize
    # python plot_guidance_norms.py rewards.json
else:
    print("ERROR: No intermediate_rewards!")
```

---

## Verification Checklist

✓ Run pipeline with:
```python
intermediate_rewards=True
reward_fn=<not None>
stein_step > 0
stein_loop > 0
```

✓ Check debug output:
```
[DEBUG] use_stein=True
[DEBUG] Logging rewards at step X/Y
[DEBUG] Returning intermediate_rewards with N logged steps
```

✓ Check output:
```python
"intermediate_rewards" in output  # True
len(output["intermediate_rewards"]["step_indices"]) > 0  # True
```

✓ Save and visualize:
```bash
python plot_guidance_norms.py rewards.json
```

✓ Should generate 4 PNG files:
- guidance_norms.png
- norm_ratio.png
- cosine_similarity.png
- alignment_effectiveness.png

---

## Still Not Working?

1. **Run the debug script**:
   ```bash
   python test_logging.py
   ```
   This will tell you exactly what's wrong

2. **Check the debug output**:
   ```
   [DEBUG] Setup: ...
   [DEBUG] Logging rewards at step ...
   [DEBUG] Returning ...
   ```
   Look for what's False/0/None

3. **Compare with example_with_logging.py** - make sure your parameters match

4. **Check the error log** - what exactly does it say is missing?
