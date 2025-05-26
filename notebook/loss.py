import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
from termcolor import colored

sns.set(style="darkgrid")

args = sys.argv[1:]
batch_skip = int(args[0]) if len(args) > 0 else 0
sched_interval = int(args[1]) if len(args) > 1 else 20

# metrics_file = 'cache/train_multitask_transformer/metrics/multitask_loss.tsv'
# metrics_file = 'cache/train_multitask_transformer_parallel/metrics/multitask_loss.tsv'
metrics_file = 'cache/finetune_benchmarks/metrics/finetune_benchmarks_loss.tsv'

def draw_plot(last_scheduler_length=0):
    # Read the first three columns of the metrics file type, batch, loss, lr
    data = pd.read_csv(metrics_file, sep='\t')
    
    if data.columns[0] != 'type':
        data.columns = ['type', 'batch', 'loss', 'lr',"tmp","tmp2"]
    
    scheduler_length = len(data[data['type'] == 'train']['loss'])
    
    # data['batch'] = data.index + 1
    # set data['batch'] to be the index right now it is the batch identifier
    data['batch'] = data.index
    data = data[data['batch'] > batch_skip]

    # create and append a new dataframe that batches the 'train' type and finds the average loss
    max_batch = data[data['type'] == 'train']['batch'].max()
    sched_data = data[data['type'] == 'train'].assign(
        batch=lambda x: max_batch - ((max_batch - x['batch']) // sched_interval) * sched_interval
    )
    sched_data = sched_data.groupby('batch')['loss'].mean().reset_index()
    sched_data['type'] = 'sched'
    data = pd.concat([data, sched_data])

    if scheduler_length == last_scheduler_length:
        return last_scheduler_length
    
    

    def print_losses(loss_type):
        dt = data[data['type'] == loss_type]['loss'].round(6)
        mn = dt.min()
        tl = data[data['type'] == loss_type]['loss'].round(6).tail()
        msg = [colored(x, 'green') if x <= mn else colored(x, 'yellow') for x in tl]
        print(colored(f"{loss_type}\t {' '.join(msg)}", 'white'))
    
    # epoch = data['batch'].iloc[-1]
    # last_lr = data['lr'].iloc[-1]

    # print(colored(f"Epoch: {epoch} Last learning rate: {last_lr}", 'white'))
    # Print the current minimum eval loss if available
    if len(data[data['type'] == 'eval']) > 0:
        min_eval_loss = data[data['type'] == 'eval']['loss'].min()
        print(colored(f"Minimum eval loss: {min_eval_loss:.6f}", 'cyan'))
    print_losses('train')
    print_losses('sched')
    if len(data[data['type'] == 'eval']) > 0:
        print_losses('eval')

    # Create the plot with a black theme and log scale for the y-axis
    g = sns.FacetGrid(data, row="type", hue="type", palette=['#1f77b4', '#ff7f0e', '#2ca02c'], sharex=True, sharey=False, height=5, aspect=2)
    g.map(sns.scatterplot, "batch", "loss", alpha=1.0, s=50, edgecolor=None)
    g.set_titles(row_template="{row_name}", color='white')
    g.set_axis_labels("Iteration", "Loss")
    
    for ax in g.axes.flatten():
        ax.set_facecolor('black')
        ax.figure.set_facecolor('black')
        ax.spines['top'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.tick_params(axis='x', colors='white', which='both')  # Ensure x-axis tick labels and minor ticks are white
        ax.tick_params(axis='y', colors='white', which='both')  # Ensure y-axis tick labels and minor ticks are white
        ax.yaxis.label.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, which='both', axis='both', color='#333333', linestyle='-', linewidth=0.5)
    
    g.add_legend(title='Metric', labelcolor='white', facecolor='black', edgecolor='black')

    # Create the directory and save the plot
    os.makedirs('notebook/plots', exist_ok=True)
    g.savefig('notebook/plots/loss.png', dpi=300, bbox_inches='tight')
    plt.close()

    return scheduler_length

last_run = 0
last_scheduler_length = 0
while True:
    last_scheduler_length = draw_plot(last_scheduler_length)
    time.sleep(15)
