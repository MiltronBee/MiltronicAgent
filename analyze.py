# analyze.py
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results():
    """
    Fetches run data from W&B, aggregates results across seeds,
    and generates a comparative analysis plot.
    """
    api = wandb.Api()
    project = "miltronic-pacman-v5-stable-warmup"
    print(f"Fetching runs from project: {project}")

    runs = api.runs(project, filters={"job_type": "full_run"})
    
    summary_list = []
    for run in runs:
        summary_list.append({
            "name": run.name,
            "agent_type": "Miltronic" if "miltronic" in run.name else "Baseline",
            "seed": run.config.get("seed"),
            "final_reward": run.summary.get("rollout/ep_rew_mean"),
            "history": run.history(keys=["rollout/ep_rew_mean", "global_step"])
        })

    if not summary_list:
        print("No runs found. Please run train.py first.")
        return

    all_history_df = pd.concat([run['history'].assign(agent_type=run['agent_type']) for run in summary_list])
    
    # Clean data
    all_history_df.dropna(inplace=True)
    all_history_df = all_history_df.rename(columns={"global_step": "Timesteps", "rollout/ep_rew_mean": "Mean Episode Reward"})

    # --- Generate and save plot ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=all_history_df, x="Timesteps", y="Mean Episode Reward", hue="agent_type", errorbar="sd")
    plt.title("Performance Comparison: Miltronic vs. Baseline (5-Seed Average)")
    plt.tight_layout()
    plot_filename = "reward_comparison.png"
    plt.savefig(plot_filename)
    print(f"Comparison plot saved to {plot_filename}")

    # --- Calculate and print final stats ---
    final_rewards_df = pd.DataFrame(summary_list)
    mean_final_rewards = final_rewards_df.groupby("agent_type")["final_reward"].mean()
    std_final_rewards = final_rewards_df.groupby("agent_type")["final_reward"].std()
    
    print("\n--- Final Performance Summary ---")
    for agent_type in mean_final_rewards.index:
        print(f"{agent_type}: Mean Reward = {mean_final_rewards[agent_type]:.2f} ± {std_final_rewards[agent_type]:.2f}")

    # --- Log analysis to W&B ---
    try:
        analysis_run = wandb.init(project=project, job_type="analysis", name="final_analysis_summary")
        analysis_run.log({
            "reward_comparison_plot": wandb.Image(plot_filename),
            "final_mean_rewards": mean_final_rewards.to_dict()
        })
        print("Analysis results logged to W&B.")
    finally:
        if analysis_run:
            analysis_run.finish()

if __name__ == '__main__':
    analyze_results()