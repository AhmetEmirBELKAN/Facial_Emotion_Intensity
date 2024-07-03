import pandas as pd
import matplotlib.pyplot as plt

csv_path = 'image_labels_with_clusters.csv'
df = pd.read_csv(csv_path)

cluster_counts = df['cluster'].value_counts()

total_count = len(df)
percentages = (cluster_counts / total_count) * 100

labels = [f'Cluster {i}: {count} ({percentage:.1f}%)' for i, count, percentage in zip(cluster_counts.index, cluster_counts.values, percentages)]

plt.figure(figsize=(8, 8))
plt.pie(cluster_counts, labels=labels, autopct='%1.1f%%', colors=['blue', 'green', 'red'])
plt.title('Distribution of Images per Cluster')

plt.savefig('cluster_distributi.png')
plt.show()
