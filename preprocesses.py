import os
import csv
import matplotlib.pyplot as plt
import pandas as pd

def createANnotation():
    dataset_dir = '/home/basilisk/Desktop/2209-b/2209b_dataset/udarshanvaidya - fer_ckplus_kdef/fer_ckplus_kdef'

    output_csv = 'emotion_dataset.csv'
    print(f"os.listdir(dataset_dir) : {os.listdir(dataset_dir)}")
    with open(output_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image Name', 'Emotion'])

        for emotion_dir in os.listdir(dataset_dir):
            emotion_path = os.path.join(dataset_dir, emotion_dir)
            if os.path.isdir(emotion_path):
                emotion_name = emotion_dir
                
                for image_name in os.listdir(emotion_path):
                    image_path = os.path.join(emotion_path, image_name)
                    if os.path.isfile(image_path):
                        csv_writer.writerow([image_name, emotion_name])

def visulationdatasetInfo():
    csv_file = 'emotion_dataset.csv'
    df = pd.read_csv(csv_file)

    emotion_counts = df['Emotion'].value_counts()

    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val:d})'
        return my_format

    plt.figure(figsize=(10, 6))
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct=autopct_format(emotion_counts), startangle=140)
    plt.title('Emotion Distribution in Dataset')
    plt.axis('equal') 

    plt.savefig("emotion_distribution_pie_chart.png")
    plt.show()

def image_labels_with_clusters_visulation():
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


image_labels_with_clusters_visulation()