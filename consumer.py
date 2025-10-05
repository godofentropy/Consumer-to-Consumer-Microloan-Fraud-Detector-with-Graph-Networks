import networkx as nx
import pandas as pd
import numpy as np
import streamlit as st
from pyvis.network import Network
import tempfile
import random

# Step 1: Generate synthetic P2P lending data

def generate_data(num_users=50, num_loans=150, fraud_ratio=0.1):
    users = [f'User_{i}' for i in range(num_users)]
    loans = []
    
    # Normal transactions
    for _ in range(int(num_loans * (1 - fraud_ratio))):
        lender, borrower = random.sample(users, 2)
        amount = np.round(np.random.uniform(100, 5000), 2)
        loans.append([lender, borrower, amount, "Legit"])
    
    # Fraudulent circular loans
    fraud_users = random.sample(users, max(3, int(num_users * fraud_ratio)))  # At least 3 to form cycles
    for i in range(len(fraud_users)):
        lender = fraud_users[i]
        borrower = fraud_users[(i+1)%len(fraud_users)]
        amount = np.round(np.random.uniform(1000, 7000), 2)
        loans.append([lender, borrower, amount, "Fraud"])
    
    df = pd.DataFrame(loans, columns=["Lender", "Borrower", "Amount", "Label"])
    return df

# Step 2: Build the graph

def build_graph(df):
    G = nx.DiGraph()
    for idx, row in df.iterrows():
        G.add_edge(row['Lender'], row['Borrower'], weight=row['Amount'], label=row['Label'])
    return G

# Step 3: Detect fraud with cycle detection and centrality

def detect_fraud(G):
    cycles = list(nx.simple_cycles(G))
    fraud_scores = nx.betweenness_centrality(G)
    suspicious_cycles = [cycle for cycle in cycles if len(cycle) <= 5]
    return fraud_scores, suspicious_cycles

# Step 4: Visualize with PyVis

def visualize_graph(G, fraud_scores, suspicious_cycles):
    net = Network(height="700px", width="100%", notebook=False, directed=True)

    for node in G.nodes():
        score = fraud_scores[node]
        color = "red" if score > 0.1 else "green"
        size = 15 + score * 50  # Bigger node for high fraud score
        net.add_node(node, label=node, title=f"Fraud Score: {score:.2f}", color=color, size=size)
    
    for source, target, data in G.edges(data=True):
        color = "orange" if data['label'] == "Fraud" else "gray"
        net.add_edge(source, target, title=f"${data['weight']} | {data['label']}", color=color)
    
    # Highlight suspicious cycles
    for cycle in suspicious_cycles:
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i+1)%len(cycle)]
            for edge in net.edges:
                if edge['from'] == source and edge['to'] == target:
                    edge['color'] = "purple"
                    edge['title'] += " (Cycle)"

    return net

# Step 5: Streamlit App

def main():
    st.title("P2P Microloan Fraud Detection using Graph Networks")
    st.markdown("Detect circular lending fraud using graph centrality and cycle detection.")
    
    num_users = st.slider("Number of Users", 20, 100, 50)
    num_loans = st.slider("Number of Loans", 50, 300, 150)
    fraud_ratio = st.slider("Fraud Ratio", 0.05, 0.5, 0.1)
    
    df = generate_data(num_users, num_loans, fraud_ratio)
    st.subheader("Synthetic Transaction Data")
    st.dataframe(df)
    
    G = build_graph(df)
    fraud_scores, suspicious_cycles = detect_fraud(G)
    
    st.subheader("Detected Suspicious Cycles (Circular Lending)")
    if suspicious_cycles:
        for cycle in suspicious_cycles:
            st.write(" → ".join(cycle) + " → " + cycle[0])
    else:
        st.write("No suspicious cycles detected.")
    
    st.subheader("Interactive Graph Visualization")
    net = visualize_graph(G, fraud_scores, suspicious_cycles)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        st.components.v1.html(open(tmp_file.name, 'r', encoding='utf-8').read(), height=750, scrolling=True)

if __name__ == "__main__":
    main()
