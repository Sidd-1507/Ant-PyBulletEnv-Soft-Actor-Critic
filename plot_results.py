"""
Plot training results for AntBulletEnv SAC.
Generates a learning curve from saved score data.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys


def plot_scores(scores_file='dir_ant_lr0001-sc2500/scores.npy',
                avg_scores_file='dir_ant_lr0001-sc2500/avg_scores.npy',
                output_file='images/plot_Ant_SAC_lr0001.png',
                target_score=2500):
    """Plot the training scores and average scores."""

    if not os.path.exists(scores_file):
        print(f"Scores file not found: {scores_file}")
        print("Run training first: python train.py --mode train")
        sys.exit(1)

    scores = np.load(scores_file)
    avg_scores = np.load(avg_scores_file)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    episodes = np.arange(1, len(scores) + 1)

    ax.plot(episodes, scores, alpha=0.3, color='blue', label='Score')
    ax.plot(episodes, avg_scores, color='red', linewidth=2, label='Avg Score (100 ep)')
    ax.axhline(y=target_score, color='green', linestyle='--', linewidth=1.5, label=f'Target ({target_score})')

    ax.set_xlabel('Episode', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('AntBulletEnv-v0 SAC Training (lr=0.0001)', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")
    plt.close()


if __name__ == '__main__':
    plot_scores()
