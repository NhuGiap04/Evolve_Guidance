"""
Minimal example: How to use guidance with logging and visualization.
"""

import json
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline

# Example reward function (use your own)
def my_reward_fn(images, prompts):
    """Example: Could be CLIP score, ImageReward, PickScore, etc."""
    # Return scores for each image (batch)
    scores = [0.5] * len(images)  # Placeholder
    return torch.tensor(scores, dtype=torch.float32)


def main():
    print("=" * 70)
    print("EXAMPLE: Guidance with Logging and Visualization")
    print("=" * 70)
    
    # 1. Load pipeline
    print("\n[1/3] Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    
    # 2. Run with logging
    print("\n[2/3] Running pipeline with logging...")
    prompt = "a beautiful landscape with mountains and lake"
    
    output = pipe(
        prompt=prompt,
        
        # ========== KEY SETTINGS FOR LOGGING ==========
        intermediate_rewards=True,      # ⚠️ ENABLE LOGGING
        reward_fn=my_reward_fn,         # ⚠️ YOUR REWARD FUNCTION
        
        # Stein particle parameters
        num_particles=2,
        stein_step=0.05,                # ⚠️ MUST BE > 0
        stein_loop=1,                   # ⚠️ MUST BE > 0
        reward_guidance_rho=0.05,
        
        # Steering steps (when to apply guidance)
        steer_start=0,
        steer_end=49,
        
        # Other parameters
        num_inference_steps=50,
        guidance_scale=7.5,
        height=512,
        width=512,
    )
    
    # 3. Extract and save logs
    print("\n[3/3] Saving logs...")
    
    if isinstance(output, dict) and "intermediate_rewards" in output:
        rewards_data = output["intermediate_rewards"]
        
        # Save to JSON
        output_file = Path("my_run_rewards.json")
        with open(output_file, "w") as f:
            json.dump(rewards_data, f, indent=2)
        
        print(f"\n✓ Saved {len(rewards_data['step_indices'])} logged steps to {output_file}")
        print(f"  Keys: {list(rewards_data.keys())}")
        
        # Show some values
        if "cosine_similarity_mean" in rewards_data:
            print(f"\n✓ Cosine similarity values:")
            print(f"  {[f'{v:.4f}' for v in rewards_data['cosine_similarity_mean'][:5]]}")
        
        if "prior_score_norm_mean" in rewards_data:
            print(f"\n✓ Prior score norm values:")
            print(f"  {[f'{v:.4f}' for v in rewards_data['prior_score_norm_mean'][:5]]}")
        
        # Visualize
        print(f"\n═══════════════════════════════════════════════════════════════════")
        print("Now visualize with:")
        print(f"  python notebooks/plot_guidance_norms.py {output_file}")
        print("═══════════════════════════════════════════════════════════════════")
        
    else:
        print("\n❌ ERROR: intermediate_rewards not in output!")
        print("Check your parameters:")
        print("  - intermediate_rewards=True ✓")
        print("  - reward_fn is not None ✓")
        print("  - stein_step > 0 ✓")
        print("  - stein_loop > 0 ✓")


if __name__ == "__main__":
    main()
