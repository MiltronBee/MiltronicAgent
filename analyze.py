# analyze.py
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration: Set these to match your W&B details ---
ENTITY = "dosdepastor69-saptiva"
PROJECT = "miltronic-pacman-v5-multi-seed"
# -------------------------------------------------------------

def analyze_results():
    """
    Fetches run data from W&B, aggregates results across seeds,
    and generates a comparative analysis plot.
    """
    api = wandb.Api()
    
    # Construct the full path required by the API
    full_project_path = f"{ENTITY}/{PROJECT}"
    print(f"Fetching runs from project: {full_project_path}")

    try:
        # Fetch all runs from the specified project
        runs = api.runs(full_project_path)
    except Exception as e:
        print(f"Error fetching runs from W&B. Please check your ENTITY and PROJECT settings.")
        print(f"Details: {e}")
        return
    summary_list = []
    miltronic_runs_found = 0
    baseline_runs_found = 0

    if len(runs) == 0:
        print(f"No runs found in project '{full_project_path}'. Exiting.")
        return

    for run in runs:
        # Skip unfinished or crashed runs
        if run.state != "finished":
            print(f"Skipping run '{run.name}' because it is not finished (state: {run.state}).")
            continue

        # Categorize runs based on their name
        agent_type = "Baseline PPO"
        if "miltronic" in run.name.lower():
            agent_type = "Miltronic PPO"
            miltronic_runs_found += 1
        else:
            baseline_runs_found += 1
            
        summary_list.append({
            "name": run.name,
            "agent_type": agent_type,
            "seed": run.config.get("seed"),
            "run_id": run.config.get("run_id"),
            "final_reward": run.summary.get("rollout/ep_rew_mean"),
            "history": run.history(keys=["rollout/ep_rew_mean", "_step"])
        })

    if not summary_list:
        print("No finished runs found to analyze.")
        return

    # --- Data Processing and Plotting ---
    all_history_df = pd.concat([run['history'].assign(agent_type=run['agent_type']) for run in summary_list])
    
    all_history_df.dropna(inplace=True)
    all_history_df = all_history_df.rename(columns={"_step": "Timesteps", "rollout/ep_rew_mean": "Mean Episode Reward"})

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="darkgrid")
    sns.lineplot(data=all_history_df, x="Timesteps", y="Mean Episode Reward", hue="agent_type", errorbar="sd", legend="full")
    plt.title(f"Multi-Seed Performance Comparison: Miltronic vs Baseline PPO")
    plt.xlabel("Training Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.tight_layout()
    
    plot_dir = "analysis_results"
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = os.path.join(plot_dir, "multi_seed_reward_comparison.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to {plot_filename}")

    # --- Final Stats Calculation ---
    final_rewards_df = pd.DataFrame(summary_list)
    print("\n--- Final Performance Summary ---")
    if miltronic_runs_found > 0:
        miltronic_rewards = final_rewards_df[final_rewards_df['agent_type'] == 'Miltronic PPO']['final_reward']
        print(f"Miltronic PPO: Mean Reward = {miltronic_rewards.mean():.2f} ± {miltronic_rewards.std():.2f} (from {miltronic_runs_found} runs)")
        print(f"Miltronic PPO individual results:")
        for _, row in final_rewards_df[final_rewards_df['agent_type'] == 'Miltronic PPO'].iterrows():
            print(f"  - Seed {row['seed']}, Run {row['run_id']}: {row['final_reward']:.2f}")
    
    if baseline_runs_found > 0:
        baseline_rewards = final_rewards_df[final_rewards_df['agent_type'] == 'Baseline PPO']['final_reward']
        print(f"Baseline PPO: Mean Reward = {baseline_rewards.mean():.2f} ± {baseline_rewards.std():.2f} (from {baseline_runs_found} runs)")
        print(f"Baseline PPO individual results:")
        for _, row in final_rewards_df[final_rewards_df['agent_type'] == 'Baseline PPO'].iterrows():
            print(f"  - Seed {row['seed']}, Run {row['run_id']}: {row['final_reward']:.2f}")

    # --- Log analysis back to W&B ---
    try:
        analysis_run = wandb.init(project=PROJECT, entity=ENTITY, job_type="analysis", name="summary_analysis_report", reinit=True)
        analysis_run.log({"reward_comparison_plot": wandb.Image(plot_filename)})
        print("\nAnalysis results logged to a new W&B run.")
    except Exception as e:
        print(f"\nCould not log analysis to W&B. Error: {e}")
    finally:
        if 'analysis_run' in locals() and analysis_run:
            analysis_run.finish()

if __name__ == '__main__':
    analyze_results()