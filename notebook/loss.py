# pip install watchdog
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

sns.set(style="darkgrid")

# --- Args (same as before) ---
args = sys.argv[1:]
batch_skip = int(args[0]) if len(args) > 0 else 0
sched_interval = int(args[1]) if len(args) > 1 else 20
update_interval = int(args[2]) if len(args) > 2 else 3  # seconds between updates

# metrics_file = 'cache/train_multitask_transformer/metrics/multitask_loss.tsv'
# metrics_file = 'cache/train_adapters/metrics/multitask_loss.tsv'
# metrics_file = 'cache/finetune_benchmarks/metrics/multitask_loss.tsv'

# metrics_file = 'cache/train_multitask_transformer_parallel/metrics/multitask_loss.tsv'
metrics_file = 'cache/train_multitask_transformer_parallel/metrics/fold_0/multitask_loss.tsv'
# metrics_file = 'cache/full_train/metrics/multitask_loss.tsv'
# metrics_file = 'cache/train_adapters_all_props/metrics/fold_0/adapter_loss.tsv'

# --- Core plotting function ---
def draw_plot(last_scheduler_length=0):
    try:
        data = pd.read_csv(metrics_file, sep='\t')
    except FileNotFoundError:
        print(colored(f"[plotter] metrics file not found: {metrics_file}", "yellow"))
        return last_scheduler_length
    except pd.errors.EmptyDataError:
        print(colored("[plotter] metrics file is empty", "yellow"))
        return last_scheduler_length
    except Exception as e:
        print(colored(f"[plotter] error reading metrics file: {e}", "red"))
        return last_scheduler_length
    
    cols = list(data.columns)
    if len(cols) >= 4 and cols[0] != 'type':
        data = data.iloc[:, :4]
        data.columns = ['type', 'batch', 'loss', 'lr']
    else:
        needed = {'type', 'batch', 'loss', 'lr'}
        missing = needed - set(cols)
        if missing:
            print(colored(f"[plotter] metrics file missing columns: {missing}", "red"))
            return last_scheduler_length
        data = data[['type', 'batch', 'loss', 'lr']]

    scheduler_length = len(data[data['type'] == 'train']['loss'])

    data = data.reset_index(drop=True)
    data['batch'] = data.index
    data = data[data['batch'] > batch_skip]

    train_data = data[data['type'] == 'train'].copy()
    if not train_data.empty:
        train_data = train_data.assign(
            sched_group=lambda x: (x['batch'] // sched_interval) * sched_interval
        )
        sched_data = train_data.groupby('sched_group')['loss'].median().reset_index()
        counts = train_data.groupby('sched_group').size().reset_index(name='count')
        sched_data = sched_data.merge(counts, on='sched_group', how='left')
        sched_data = sched_data[sched_data['count'] >= (sched_interval - 5)]
        sched_data = sched_data.rename(columns={'sched_group': 'batch'})
        sched_data['type'] = 'sched'
    else:
        sched_data = pd.DataFrame(columns=['batch', 'loss', 'count', 'type'])

    eval_data = data[data['type'] == 'eval'][['batch', 'loss']].copy()
    eval_data['type'] = 'eval'

    lr_data = train_data[['batch', 'lr']].copy()
    lr_data = lr_data.rename(columns={'lr': 'value'})
    lr_data['type'] = 'learning_rate'

    sched_plot = sched_data[['batch', 'loss', 'type']].rename(columns={'loss': 'value'})
    eval_plot = eval_data[['batch', 'loss', 'type']].rename(columns={'loss': 'value'})

    plot_data = pd.concat([sched_plot, eval_plot, lr_data], ignore_index=True)

    if scheduler_length == last_scheduler_length:
        return last_scheduler_length  # nothing new

    def print_losses(loss_type):
        if loss_type not in ['train', 'sched', 'eval']:
            return
        if loss_type == 'train':
            dt = data[data['type'] == 'train']['loss'].round(6)
        elif loss_type == 'sched':
            dt = sched_data['loss'].round(6) if not sched_data.empty else pd.Series(dtype=float)
        else:
            dt = data[data['type'] == 'eval']['loss'].round(6)
        if dt.empty:
            return
        mn = dt.min()
        tl = dt.tail()
        msg = [colored(x, 'green') if x <= mn else colored(x, 'yellow') for x in tl]
        print(colored(f"{loss_type}\t {' '.join(msg)}", 'white'))

    if not data[data['type'] == 'eval'].empty:
        min_eval_loss = data[data['type'] == 'eval']['loss'].min()
        print(colored(f"Minimum eval loss: {min_eval_loss:.6f}", 'cyan'))
    print_losses('train')
    print_losses('sched')
    print_losses('eval')

    palette = {
        'eval': '#1f77b4',
        'sched': '#ff7f0e',
        'learning_rate': '#2ca02c'
    }

    g = sns.FacetGrid(
        plot_data,
        row="type",
        hue="type",
        palette=palette,
        sharex=True,
        sharey=False,
        height=5,
        aspect=2
    )
    g.map(sns.scatterplot, "batch", "value", alpha=1.0, s=50, edgecolor=None)
    g.set_titles(row_template="{row_name}", color='white')
    g.set_axis_labels("Iteration", "")

    next_eval_batch = None
    if not eval_data.empty:
        eval_batches = eval_data['batch'].to_numpy()
        if len(eval_batches) >= 2:
            interval = int(eval_batches[-1] - eval_batches[-2])
        else:
            interval = int(sched_interval)  # fallback if only one eval logged
        next_eval_batch = int(eval_batches[-1] + interval)

    for ax, (row_type, _) in zip(g.axes.flatten(), plot_data.groupby('type', sort=False)):
        ax.set_facecolor('black')
        ax.figure.set_facecolor('black')
        for side in ['top', 'bottom', 'left', 'right']:
            ax.spines[side].set_color('white')
        ax.tick_params(axis='x', colors='white', which='both')
        ax.tick_params(axis='y', colors='white', which='both')
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, which='both', axis='both', color='#333333', linestyle='-', linewidth=0.5)
        if row_type == 'learning_rate':
            ax.set_ylabel('Learning Rate', color='white')
        elif row_type == 'eval':
            ax.set_ylabel('Eval Loss', color='white')
        elif row_type == 'sched':
            ax.set_ylabel('Sched Loss', color='white')

        if next_eval_batch is not None:
            ax.axvline(next_eval_batch, color='red', linestyle='--', linewidth=1)
            if row_type == 'eval':
                ymax = ax.get_ylim()[1]
                ax.text(
                    next_eval_batch, ymax,
                    f'Next eval ~{next_eval_batch}',
                    color='red', rotation=90, va='top', ha='right'
                )

    g.add_legend(title='Metric', labelcolor='white', facecolor='black', edgecolor='black')

    os.makedirs('notebook/plots', exist_ok=True)
    g.savefig('notebook/plots/loss2.png', dpi=300, bbox_inches='tight')
    plt.close()

    return scheduler_length

def main():
    # Track the last scheduler length between redraws
    last_scheduler_length = 0
    
    print(colored(f"[plotter] updating plot every {update_interval} seconds...", "cyan"))
    print(colored(f"[plotter] monitoring {metrics_file}", "cyan"))
    
    # Initial draw if file already exists
    last_scheduler_length = draw_plot(last_scheduler_length)
    
    try:
        while True:
            time.sleep(update_interval)
            last_scheduler_length = draw_plot(last_scheduler_length)
    except KeyboardInterrupt:
        print(colored("[plotter] stopping...", "yellow"))

if __name__ == "__main__":
    main()