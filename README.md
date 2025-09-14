# ML-Mini-Project

Student Friendship Group Clustering Analysis
This project uses unsupervised machine learning to cluster students into "friendship groups" based on their shared interests, including hobbies, club participation, and other lifestyle preferences. The analysis uses K-Means clustering and evaluates the quality of the clusters using the Silhouette Score and the Davies–Bouldin Index.

Project Highlights
Data Loading & Preprocessing: The project loads a CSV file named snu_friendship.csv, handles categorical features through one-hot encoding, and prepares the data for clustering.

Optimal Cluster Determination: It uses the Silhouette Score and Davies–Bouldin Index to determine the optimal number of clusters, evaluating a range of k from 2 to 10.

Clustering: The K-Means algorithm is applied to the processed data to group the students.

Visualization: Principal Component Analysis (PCA) is used to reduce the dimensionality of the data to 2D, allowing for a visual representation of the clusters.

Evaluation: The quality of the final clustering is quantified using the Silhouette Score (0.50) and the Davies–Bouldin Index (0.31).

Cluster Analysis: The project analyzes the number of students in each cluster and provides insights into the distinguishing characteristics of each group based on their hobbies, club affiliations, and other survey data.

Repository Structure
ML Mini Project smp.ipynb: The main Jupyter Notebook containing all the code for data processing, clustering, visualization, and analysis.

snu_friendship.csv: The input dataset used for the project.

Quick Start
Prerequisites
Make sure you have Python and Jupyter installed. The necessary libraries include pandas, matplotlib, scikit-learn, and seaborn.

How to Run
Place the snu_friendship.csv file in the same directory as the notebook.

Open the ML Mini Project smp.ipynb notebook in Jupyter.

Run all the cells in the notebook.

The notebook will automatically perform the following steps:

Load and inspect the data.

Preprocess categorical features using one-hot encoding.

Plot the Silhouette Scores and Davies–Bouldin Indices to help identify an optimal number of clusters.

Apply K-Means clustering with the chosen number of clusters.

Visualize the resulting clusters in a 2D PCA plot.

Print the final Silhouette Score and Davies–Bouldin Index.

Provide a detailed summary of the number of students and key characteristics for each identified cluster.

Key Findings
The analysis identified 10 distinct clusters, though some are very small (single-student clusters).

The two largest clusters (Cluster 0 and Cluster 7) account for a significant portion of the students (41 and 35 students respectively).

The Silhouette Score of 0.50 indicates that the clusters are reasonably well-separated and distinct.

The Davies–Bouldin Index of 0.31 suggests that the clusters are relatively compact and well-separated from each other, as lower values are better.

Next Steps
Refine Cluster Count: The plots for the Silhouette Score and Davies-Bouldin index can be used to further investigate the optimal number of clusters for a potentially better grouping.

Deeper Feature Analysis: A more in-depth analysis of the features within each cluster could be conducted to create more descriptive profiles of each friendship group, which could be used for activities and student engagement planning.



By- Pratik Kumar , Sohel Khan ,Manish Raj
