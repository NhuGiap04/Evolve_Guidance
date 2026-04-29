"""
Visualization utility for guidance norm analysis.
Plots prior score norms, reward gradient norms, reward scaling, and cosine similarity over denoising steps.
"""

import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np


def plot_guidance_norms(
    intermediate_rewards: Dict[str, List[float]],
    figsize: tuple = (16, 10),
    save_path: Optional[Union[str, Path]] = None,
    title_suffix: str = "",
):
    """
    Plot guidance norm statistics from intermediate rewards data.
    
    Parameters:
    -----------
    intermediate_rewards : dict
        Dictionary containing logged norm and reward data with keys like:
        - step_indices, timesteps
        - prior_score_norm_mean, prior_score_norm_max
        - reward_grad_norm_mean, reward_grad_norm_max
        - reward_scale_mean, reward_scale_max
        - pre_steer_mean, pre_steer_max, post_steer_mean, post_steer_max
    figsize : tuple
        Figure size for matplotlib
    save_path : str or Path, optional
        If provided, save the figure to this path
    title_suffix : str
        Additional suffix for the plot title
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.suptitle(f"Guidance Norm Analysis{title_suffix}", fontsize=16, fontweight='bold')
    
    steps = intermediate_rewards.get("step_indices", list(range(len(intermediate_rewards.get("prior_score_norm_mean", [])))))
    
    # 1. Prior Score Norm vs Reward Gradient Norm
    ax = axes[0, 0]
    if "prior_score_norm_mean" in intermediate_rewards:
        ax.plot(steps, intermediate_rewards["prior_score_norm_mean"], 'b-o', label='Prior Score', linewidth=2, markersize=4)
        ax.fill_between(steps, intermediate_rewards["prior_score_norm_max"], alpha=0.2, color='b')
    if "reward_grad_norm_mean" in intermediate_rewards:
        ax.plot(steps, intermediate_rewards["reward_grad_norm_mean"], 'r-s', label='Reward Grad', linewidth=2, markersize=4)
        ax.fill_between(steps, intermediate_rewards["reward_grad_norm_max"], alpha=0.2, color='r')
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Norm (Mean)")
    ax.set_title("Prior Score vs Reward Gradient Norm (Mean)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Max Norms
    ax = axes[0, 1]
    if "prior_score_norm_max" in intermediate_rewards:
        ax.plot(steps, intermediate_rewards["prior_score_norm_max"], 'b-o', label='Prior Score Max', linewidth=2, markersize=4)
    if "reward_grad_norm_max" in intermediate_rewards:
        ax.plot(steps, intermediate_rewards["reward_grad_norm_max"], 'r-s', label='Reward Grad Max', linewidth=2, markersize=4)
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Norm (Max)")
    ax.set_title("Prior Score vs Reward Gradient Norm (Max)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Reward Scale Evolution
    ax = axes[1, 0]
    if "reward_scale_mean" in intermediate_rewards:
        ax.plot(steps, intermediate_rewards["reward_scale_mean"], 'g-o', label='Mean Scale', linewidth=2, markersize=4)
        if "reward_scale_max" in intermediate_rewards:
            ax.fill_between(steps, 0, intermediate_rewards["reward_scale_max"], alpha=0.2, color='g')
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Scale Factor")
    ax.set_title("Reward Scale (rho * ||grad|| / ||prior||)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Reward Scale vs Gradient Norm
    ax = axes[1, 1]
    if "reward_scale_mean" in intermediate_rewards and "reward_grad_norm_mean" in intermediate_rewards:
        ax2 = ax.twinx()
        line1 = ax.plot(steps, intermediate_rewards["reward_scale_mean"], 'g-o', label='Scale Mean', linewidth=2, markersize=4)
        line2 = ax2.plot(steps, intermediate_rewards["reward_grad_norm_mean"], 'r--s', label='Grad Norm Mean', linewidth=2, markersize=4)
        ax.set_xlabel("Denoising Step")
        ax.set_ylabel("Scale Factor", color='g')
        ax2.set_ylabel("Grad Norm", color='r')
        ax.tick_params(axis='y', labelcolor='g')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_title("Reward Scale vs Gradient Norm")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # 5. Reward Change (pre vs post steering)
    ax = axes[2, 0]
    if "pre_steer_mean" in intermediate_rewards and "post_steer_mean" in intermediate_rewards:
        ax.plot(steps, intermediate_rewards["pre_steer_mean"], 'b-o', label='Pre-Steering', linewidth=2, markersize=4)
        ax.plot(steps, intermediate_rewards["post_steer_mean"], 'g-s', label='Post-Steering', linewidth=2, markersize=4)
        ax.fill_between(steps, intermediate_rewards["pre_steer_mean"], intermediate_rewards["post_steer_mean"], alpha=0.2, color='green')
    ax.set_xlabel("Denoising Step")
    ax.set_ylabel("Reward (Mean)")
    ax.set_title("Reward Improvement from Steering")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Score Composition (prior + scaled_reward)
    ax = axes[2, 1]
    if all(k in intermediate_rewards for k in ["pre_score_norm_mean", "reward_grad_norm_mean", "reward_scale_mean"]):
        # Approximate the scaled reward magnitude
        prior_norms = np.array(intermediate_rewards["prior_score_norm_mean"])
        reward_grad_norms = np.array(intermediate_rewards["reward_grad_norm_mean"])
        scales = np.array(intermediate_rewards["reward_scale_mean"])
        scaled_reward_norms = scales * reward_grad_norms
        
        ax.bar(steps, prior_norms, label='Prior Score Norm', alpha=0.7, width=0.4, align='edge')
        ax.bar(np.array(steps) + 0.4, scaled_reward_norms, label='Scaled Reward Norm', alpha=0.7, width=0.4, align='edge')
        ax.set_xlabel("Denoising Step")
        ax.set_ylabel("Norm Magnitude")
        ax.set_title("Score Component Magnitudes")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_norm_ratio(
    intermediate_rewards: Dict[str, List[float]],
    figsize: tuple = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot the ratio of reward gradient norm to prior score norm.
    This helps understand the relative influence of reward guidance.
    
    Parameters:
    -----------
    intermediate_rewards : dict
        Dictionary containing logged norm data
    figsize : tuple
        Figure size for matplotlib
    save_path : str or Path, optional
        If provided, save the figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Reward Guidance Influence Analysis", fontsize=14, fontweight='bold')
    
    steps = intermediate_rewards.get("step_indices", list(range(len(intermediate_rewards.get("reward_grad_norm_mean", [])))))
    
    if "prior_score_norm_mean" in intermediate_rewards and "reward_grad_norm_mean" in intermediate_rewards:
        prior_norms = np.array(intermediate_rewards["prior_score_norm_mean"])
        grad_norms = np.array(intermediate_rewards["reward_grad_norm_mean"])
        
        # Avoid division by zero
        ratio_mean = np.divide(grad_norms, np.maximum(prior_norms, 1e-8))
        
        ax1.plot(steps, ratio_mean, 'mo-', linewidth=2, markersize=6)
        ax1.set_xlabel("Denoising Step")
        ax1.set_ylabel("Ratio (||grad|| / ||prior||)")
        ax1.set_title("Gradient to Prior Norm Ratio")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal magnitude')
        ax1.legend()
        
        # Log scale view
        ax2.semilogy(steps, ratio_mean, 'mo-', linewidth=2, markersize=6)
        ax2.set_xlabel("Denoising Step")
        ax2.set_ylabel("Ratio (||grad|| / ||prior||) [log scale]")
        ax2.set_title("Gradient to Prior Norm Ratio (Log Scale)")
        ax2.grid(True, alpha=0.3, which='both')
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, (ax1, ax2)


def plot_cosine_similarity(
    intermediate_rewards: Dict[str, List[float]],
    figsize: tuple = (15, 10),
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot cosine similarity between prior score and reward gradient.
    
    Parameters:
    -----------
    intermediate_rewards : dict
        Dictionary containing logged cosine similarity data
    figsize : tuple
        Figure size for matplotlib
    save_path : str or Path, optional
        If provided, save the figure to this path
    """
    if "cosine_similarity_mean" not in intermediate_rewards:
        print("Cosine similarity data not available in intermediate_rewards")
        return None, None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Prior Score vs Reward Gradient Alignment (Cosine Similarity)", fontsize=14, fontweight='bold')
    
    steps = intermediate_rewards.get("step_indices", list(range(len(intermediate_rewards.get("cosine_similarity_mean", [])))))
    cos_sim_mean = np.array(intermediate_rewards.get("cosine_similarity_mean", []))
    cos_sim_min = np.array(intermediate_rewards.get("cosine_similarity_min", []))
    cos_sim_max = np.array(intermediate_rewards.get("cosine_similarity_max", []))
    
    # 1. Cosine similarity over time
    ax = axes[0, 0]
    ax.plot(steps, cos_sim_mean, 'mo-', linewidth=2.5, markersize=6, label='Mean')
    ax.fill_between(steps, cos_sim_min, cos_sim_max, alpha=0.2, color='m', label='Min-Max range')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='Orthogonal')
    ax.axhline(y=1, color='g', linestyle='--', linewidth=1, alpha=0.5, label='Aligned')
    ax.axhline(y=-1, color='r', linestyle='--', linewidth=1, alpha=0.5, label='Opposite')
    ax.set_xlabel("Denoising Step", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Alignment Over Time", fontsize=12, fontweight='bold')
    ax.set_ylim([-1.2, 1.2])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Distribution
    ax = axes[0, 1]
    ax.hist(cos_sim_mean, bins=10, alpha=0.7, color='m', edgecolor='black')
    ax.axvline(x=np.mean(cos_sim_mean), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cos_sim_mean):.3f}')
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Distribution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Min, Mean, Max
    ax = axes[1, 0]
    ax.plot(steps, cos_sim_mean, 'mo-', linewidth=2.5, markersize=6, label='Mean')
    ax.plot(steps, cos_sim_max, 'm^--', linewidth=2, markersize=5, label='Max', alpha=0.7)
    ax.plot(steps, cos_sim_min, 'mv--', linewidth=2, markersize=5, label='Min', alpha=0.7)
    ax.fill_between(steps, cos_sim_min, cos_sim_max, alpha=0.15, color='m')
    ax.set_xlabel("Denoising Step", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Range (Min, Mean, Max)", fontsize=12, fontweight='bold')
    ax.set_ylim([-1.1, 1.1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 4. Interpretation zones
    ax = axes[1, 1]
    ax.axhspan(-1, -0.3, alpha=0.1, color='red', label='Strongly opposed')
    ax.axhspan(-0.3, 0, alpha=0.1, color='orange', label='Misaligned')
    ax.axhspan(0, 0.7, alpha=0.1, color='yellow', label='Partially aligned')
    ax.axhspan(0.7, 1, alpha=0.1, color='green', label='Well-aligned')
    
    ax.plot(steps, cos_sim_mean, 'mo-', linewidth=2.5, markersize=6, label='Actual', zorder=5)
    ax.set_xlabel("Denoising Step", fontsize=11)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Alignment Interpretation Zones", fontsize=12, fontweight='bold')
    ax.set_ylim([-1.1, 1.1])
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, zorder=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, axes


def plot_alignment_effectiveness(
    intermediate_rewards: Dict[str, List[float]],
    figsize: tuple = (14, 5),
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot relationship between cosine similarity and actual reward improvement.
    
    Parameters:
    -----------
    intermediate_rewards : dict
        Dictionary containing logged data
    figsize : tuple
        Figure size for matplotlib
    save_path : str or Path, optional
        If provided, save the figure to this path
    """
    if not all(k in intermediate_rewards for k in ["cosine_similarity_mean", "pre_steer_mean", "post_steer_mean"]):
        print("Insufficient data for alignment effectiveness plot")
        return None, None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Alignment vs Effectiveness", fontsize=14, fontweight='bold')
    
    steps = intermediate_rewards.get("step_indices", list(range(len(intermediate_rewards.get("cosine_similarity_mean", [])))))
    cos_sim = np.array(intermediate_rewards["cosine_similarity_mean"])
    pre_reward = np.array(intermediate_rewards.get("pre_steer_mean", []))
    post_reward = np.array(intermediate_rewards.get("post_steer_mean", []))
    reward_improvement = post_reward - pre_reward
    
    # Cosine similarity vs reward improvement
    ax1.scatter(cos_sim, reward_improvement, s=100, alpha=0.6, c=steps, cmap='viridis')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Cosine Similarity (alignment)", fontsize=11)
    ax1.set_ylabel("Reward Improvement", fontsize=11)
    ax1.set_title("Alignment vs Reward Gain", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Overlaid trends
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(steps, cos_sim, 'mo-', linewidth=2.5, markersize=6, label='Cosine Similarity')
    line2 = ax2_twin.bar(steps, reward_improvement, alpha=0.3, color='c', label='Reward Improvement')
    
    ax2.set_xlabel("Denoising Step", fontsize=11)
    ax2.set_ylabel("Cosine Similarity", color='m', fontsize=11)
    ax2_twin.set_ylabel("Reward Improvement", color='c', fontsize=11)
    ax2.tick_params(axis='y', labelcolor='m')
    ax2_twin.tick_params(axis='y', labelcolor='c')
    ax2.set_title("Alignment and Reward Over Time", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(line1, [l.get_label() for l in line1], loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig, (ax1, ax2)



    json_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    show: bool = True,
):
    """
    Load intermediate rewards from JSON and create visualizations.
    
    Parameters:
    -----------
    json_path : str or Path
        Path to the intermediate_rewards JSON file
    output_dir : str or Path, optional
        Directory to save plots. If None, uses same directory as JSON
    show : bool
        Whether to display plots
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    
    with open(json_path) as f:
        intermediate_rewards = json.load(f)
    
    if output_dir is None:
        output_dir = json_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    fig1, _ = plot_guidance_norms(
        intermediate_rewards,
        save_path=output_dir / "guidance_norms.png",
        title_suffix=f" - {json_path.stem}"
    )
    
    fig2, _ = plot_norm_ratio(
        intermediate_rewards,
        save_path=output_dir / "norm_ratio.png",
    )
    
    fig3, _ = plot_cosine_similarity(
        intermediate_rewards,
        save_path=output_dir / "cosine_similarity.png",
    )
    
    fig4, _ = plot_alignment_effectiveness(
        intermediate_rewards,
        save_path=output_dir / "alignment_effectiveness.png",
    )
    
    if show:
        plt.show()
    
    print(f"Plots saved to {output_dir}")
    return intermediate_rewards


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        load_and_plot(json_path, output_dir)
    else:
        print("Usage: python plot_guidance_norms.py <json_path> [output_dir]")
        print("\nExample:")
        print("  python plot_guidance_norms.py rewards.json ./plots/")
