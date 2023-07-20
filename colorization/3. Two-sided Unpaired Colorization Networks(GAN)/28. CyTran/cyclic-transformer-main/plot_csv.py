import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the CSV file
df = pd.read_csv('test_csv/eval_metrics_56.csv')

# Create a line plot with epoch on the x-axis and each of the three metrics as a separate line
sns.lineplot(x='epoch', y='mae_post', data=df, label='MAE Post')
sns.lineplot(x='epoch', y='rmse_post', data=df, label='RMSE Post')
sns.lineplot(x='epoch', y='ssim_post', data=df, label='SSIM Post')

# Set the title and axis labels
plt.title('Evaluation Metrics for Epochs 2-56')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')

# Display the plot
plt.show()
