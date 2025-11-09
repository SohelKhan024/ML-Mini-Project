import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import time
import numpy as np


# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('snu_friendship.csv')
    columns = ['Hobby_top1', 'Hobby top2', 'Club top1', 'Club top 2', 'Teamwork preference  \n  Like: ( 1=Prefer Solo, 5=Prefer Teams)']
    df_selected = df[columns].dropna()
    return df_selected

# Train and save the model
@st.cache_data
def train_model(df_selected):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df_selected)

    # Fit K-Means with optimal k=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(encoded_data)

    # Calculate evaluation metrics
    silhouette = silhouette_score(encoded_data, clusters)
    davies_bouldin = davies_bouldin_score(encoded_data, clusters)

    return encoder, kmeans, encoded_data, clusters, silhouette, davies_bouldin

# Function to predict cluster
def predict_cluster(encoder, kmeans, hobby_top1, hobby_top2, club_top1, club_top2, teamwork_pref):
    new_data = pd.DataFrame({
        'Hobby_top1': [hobby_top1],
        'Hobby top2': [hobby_top2],
        'Club top1': [club_top1],
        'Club top 2': [club_top2],
        'Teamwork preference  \n  Like: ( 1=Prefer Solo, 5=Prefer Teams)': [teamwork_pref]
    })
    encoded_new = encoder.transform(new_data)
    predicted_cluster = kmeans.predict(encoded_new)[0]
    return predicted_cluster

# Streamlit app
def main():
    st.title("🎓 The Friendship Blueprint of SNU - Find the club best suited to you")
    st.markdown("Discover which friendship group you belong to based on your hobbies and interests!")

    # Load data and train model
    df_selected = load_data()
    encoder, kmeans, encoded_data, clusters, silhouette, davies_bouldin = train_model(df_selected)

    # Sidebar for inputs
    st.sidebar.header("📝 Enter Your Details")
    hobby_top1 = st.sidebar.selectbox("Top Hobby 1", df_selected['Hobby_top1'].unique())
    hobby_top2 = st.sidebar.selectbox("Top Hobby 2", df_selected['Hobby top2'].unique())
    club_top1 = st.sidebar.selectbox("Top Club 1", df_selected['Club top1'].unique())
    club_top2 = st.sidebar.selectbox("Top Club 2", df_selected['Club top 2'].unique())
    teamwork_pref = st.sidebar.slider("Teamwork Preference (1-5)", 1, 5, 3)

    if st.sidebar.button("🔮 Predict My Club"):
        predicted_cluster = predict_cluster(encoder, kmeans, hobby_top1, hobby_top2, club_top1, club_top2, teamwork_pref)

        # Club nicknames and descriptions
        club_nicknames = {
            0: "🦋 Social Butterflies",
            1: "🎨 Creative Minds",
            2: "⚽ Active Enthusiasts",
            3: "📚 Thoughtful Individuals"
        }

        club_descriptions = {
            0: "🌟 **Social Butterflies**: You love group activities and teamwork. Great for collaborative projects!",
            1: "🎨 **Creative Minds**: Artistic and hobby-focused. Perfect for creative clubs and events!",
            2: "⚽ **Active Enthusiasts**: Sports and outdoor hobbies. Ideal for sports teams and adventure groups!",
            3: "📚 **Thoughtful Individuals**: Prefer independent activities with some teamwork. Suited for study groups and quiet gatherings!"
        }

        nickname = club_nicknames.get(predicted_cluster, f"Club {predicted_cluster}")
        st.success(f"🎉 You should join **{nickname} Club**!")

        st.markdown(club_descriptions.get(predicted_cluster, "A unique club just for you!"))

        # Display predicted club with icon and message
        st.header("🎯 Your Predicted Club")

        # Club icons and names
        club_icons = {
            0: "🦋",
            1: "🎨",
            2: "⚽",
            3: "📚"
        }

        club_full_names = {
            0: "Social Butterflies",
            1: "Creative Minds",
            2: "Active Enthusiasts",
            3: "Thoughtful Individuals"
        }

        icon = club_icons.get(predicted_cluster, "🎉")
        full_name = club_full_names.get(predicted_cluster, f"Club {predicted_cluster}")

        # Large icon display
        st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{icon}</h1>", unsafe_allow_html=True)

        # Club name and message
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{full_name}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 18px;'>🎉 Congratulations! Based on your hobbies and interests, you are a perfect fit for the <strong>{full_name}</strong> club!</p>", unsafe_allow_html=True)

    # Add visualizations section
    st.header("📊 Cluster Analysis & Visualizations")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Cluster Distribution", "🎯 PCA Visualization", "📊 Evaluation Metrics", "📋 Cluster Details"])

    with tab1:
        st.subheader("Cluster Distribution")
        cluster_counts = Counter(clusters)
        cluster_df = pd.DataFrame({
            'Cluster': list(cluster_counts.keys()),
            'Number of Students': list(cluster_counts.values()),
            'Club Name': ['🦋 Social Butterflies', '🎨 Creative Minds', '⚽ Active Enthusiasts', '📚 Thoughtful Individuals']
        })

        # Animated bar chart
        fig = px.bar(cluster_df, x='Club Name', y='Number of Students',
                     title='Student Distribution Across Friendship Groups',
                     color='Club Name', color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Pie chart
        fig_pie = px.pie(cluster_df, values='Number of Students', names='Club Name',
                        title='Proportion of Students in Each Group')
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        st.subheader("PCA Cluster Visualization")
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(encoded_data)
        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = clusters
        pca_df['Club Name'] = pca_df['Cluster'].map({
            0: '🦋 Social Butterflies',
            1: '🎨 Creative Minds',
            2: '⚽ Active Enthusiasts',
            3: '📚 Thoughtful Individuals'
        })

        # Interactive scatter plot
        fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='Club Name',
                           title='2D PCA Visualization of Student Clusters',
                           hover_data=['Club Name'])
        st.plotly_chart(fig_pca, use_container_width=True)

    with tab3:
        st.subheader("Clustering Evaluation Metrics")

        # Create metrics display
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Silhouette Score", f"{silhouette:.3f}",
                     help="Higher values (closer to 1) indicate better cluster separation")

        with col2:
            st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}",
                     help="Lower values indicate better clustering quality")

        # Progress bars for metrics
        st.subheader("Metric Quality Indicators")
        st.progress(min(1.0, max(0.0, (silhouette + 1) / 2)))  # Normalize to 0-1
        st.caption("Silhouette Score Quality (higher is better)")

        st.progress(max(0.0, 1.0 - davies_bouldin))  # Invert DB index
        st.caption("Davies-Bouldin Index Quality (lower is better)")

    with tab4:
        st.subheader("Detailed Cluster Analysis")
        df_selected['Cluster'] = clusters

        # Club sizes and summaries
        cluster_counts = Counter(clusters)

    # Define club_nicknames here as well
    club_nicknames = {
        0: "🦋 Social Butterflies",
        1: "🎨 Creative Minds",
        2: "⚽ Active Enthusiasts",
        3: "📚 Thoughtful Individuals"
    }

    for cluster in sorted(cluster_counts.keys()):
        nickname = club_nicknames.get(cluster, f"Club {cluster}")
        st.subheader(f"{nickname} Club")

        # Get cluster data
        cluster_data = df_selected[df_selected['Cluster'] == cluster]

        # Most common hobbies
        top_hobby1 = cluster_data['Hobby_top1'].mode().iloc[0] if not cluster_data['Hobby_top1'].mode().empty else "N/A"
        top_hobby2 = cluster_data['Hobby top2'].mode().iloc[0] if not cluster_data['Hobby top2'].mode().empty else "N/A"

        # Most common clubs
        top_club1 = cluster_data['Club top1'].mode().iloc[0] if not cluster_data['Club top1'].mode().empty else "N/A"
        top_club2 = cluster_data['Club top 2'].mode().iloc[0] if not cluster_data['Club top 2'].mode().empty else "N/A"

        # Average teamwork preference
        avg_teamwork = cluster_data['Teamwork preference  \n  Like: ( 1=Prefer Solo, 5=Prefer Teams)'].mean()

        st.write(f"**Top Hobbies:** {top_hobby1}, {top_hobby2}")
        st.write(f"**Top Clubs:** {top_club1}, {top_club2}")
        st.write(f"**Average Teamwork Preference:** {avg_teamwork:.2f}")
        st.write("---")

    # Add animations and interactive elements
    st.header("🎉 Fun Facts & Animations")

    # Animated counter for total students
    total_students = len(df_selected)
    st.subheader("📊 Total Students Analyzed")
    st.markdown(f"<h1 style='text-align: center; color: #FF6B6B;'>{total_students}</h1>", unsafe_allow_html=True)

    # Fun facts
    col1, col2 = st.columns(2)
    with col1:
        st.info("🤝 **Friendship Insight**: Students with similar hobbies tend to form stronger bonds!")
    with col2:
        st.success("🎯 **Clustering Magic**: Our AI found 4 distinct friendship groups using advanced machine learning!")

    # Add some emojis and animations
    st.markdown("---")
    st.markdown("✨ **Built with ❤️ using Streamlit and Machine Learning** ✨")
    st.markdown("🚀 **Powered by K-Means Clustering & PCA Visualization** 🚀")

if __name__ == "__main__":
    main()
