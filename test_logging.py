"""
Test script to verify logging and visualization are working.
Checks if intermediate_rewards_data is properly collected and saved.
"""

import json
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from diffusers import StableDiffusionPipeline
from seg.scorers.clip_scorer import ClipScorer

def test_intermediate_rewards_logging():
    """Test that intermediate rewards are being logged correctly."""
    
    print("=" * 70)
    print("TESTING INTERMEDIATE REWARDS LOGGING")
    print("=" * 70)
    
    # Load pipeline
    print("\n1. Loading pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    # Create simple reward function
    print("2. Creating reward function...")
    scorer = ClipScorer(
        model_name="openai/clip-vit-base-patch32",
        device="cuda"
    )
    
    def reward_fn(images, prompts):
        """Simple CLIP score reward."""
        scores = scorer(images, prompts)
        return torch.tensor(scores, dtype=torch.float32)
    
    # Test parameters
    prompt = "a beautiful landscape photo"
    test_params = {
        "prompt": prompt,
        "height": 512,
        "width": 512,
        "num_inference_steps": 20,
        "guidance_scale": 7.5,
        # STEIN parameters
        "num_particles": 2,
        "reward_fn": reward_fn,
        "stein_step": 0.05,
        "stein_loop": 1,
        "reward_guidance_rho": 0.05,
        "steer_start": 0,
        "steer_end": 19,
        "intermediate_rewards": True,  # IMPORTANT: Enable logging
    }
    
    print("\n3. Test Parameters:")
    print(f"   - intermediate_rewards: {test_params['intermediate_rewards']}")
    print(f"   - num_particles: {test_params['num_particles']}")
    print(f"   - stein_step: {test_params['stein_step']}")
    print(f"   - stein_loop: {test_params['stein_loop']}")
    print(f"   - reward_fn: {test_params['reward_fn'] is not None}")
    
    print("\n4. Running pipeline...")
    try:
        output = pipe(**test_params)
        
        if isinstance(output, dict) and "intermediate_rewards" in output:
            rewards = output["intermediate_rewards"]
            print("\n✓ Intermediate rewards returned!")
            print(f"   Keys: {list(rewards.keys())}")
            
            if "cosine_similarity_mean" in rewards and len(rewards["cosine_similarity_mean"]) > 0:
                print(f"\n✓ Cosine similarity logged: {len(rewards['cosine_similarity_mean'])} steps")
                print(f"   Values: {rewards['cosine_similarity_mean'][:3]}...")
            else:
                print("\n❌ Cosine similarity NOT found or empty!")
                
            if "prior_score_norm_mean" in rewards and len(rewards["prior_score_norm_mean"]) > 0:
                print(f"\n✓ Prior score norm logged: {len(rewards['prior_score_norm_mean'])} steps")
                print(f"   Values: {rewards['prior_score_norm_mean'][:3]}...")
            else:
                print("\n❌ Prior score norm NOT found or empty!")
            
            if "reward_grad_norm_mean" in rewards and len(rewards["reward_grad_norm_mean"]) > 0:
                print(f"\n✓ Reward grad norm logged: {len(rewards['reward_grad_norm_mean'])} steps")
                print(f"   Values: {rewards['reward_grad_norm_mean'][:3]}...")
            else:
                print("\n❌ Reward grad norm NOT found or empty!")
            
            # Save to JSON
            output_path = Path("test_intermediate_rewards.json")
            with open(output_path, "w") as f:
                json.dump(rewards, f, indent=2)
            print(f"\n✓ Saved to {output_path}")
            
            return True
        else:
            print("\n❌ intermediate_rewards NOT in output!")
            print(f"   Output type: {type(output)}")
            print(f"   Output keys: {output.keys() if isinstance(output, dict) else 'N/A'}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during pipeline run: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_intermediate_rewards_logging()
    
    if success:
        print("\n" + "=" * 70)
        print("✓ LOGGING IS WORKING!")
        print("=" * 70)
        print("\nNow you can visualize with:")
        print("  python plot_guidance_norms.py test_intermediate_rewards.json")
    else:
        print("\n" + "=" * 70)
        print("❌ LOGGING NOT WORKING - CHECK THE ERRORS ABOVE")
        print("=" * 70)
        print("\nPossible fixes:")
        print("1. Make sure intermediate_rewards=True")
        print("2. Make sure reward_fn is not None")
        print("3. Make sure stein_step > 0 and stein_loop > 0")
        print("4. Make sure steer_start/end are valid")
