"""
Opinion Leader Analysis for Moltbook Dataset
=============================================
Identifies opinion leaders by degree centrality in the interaction network,
then analyzes their post content features and agent description features.

Usage:
    python opinion_leader_analysis.py

Output: Creates 'analysis_results/' folder with CSV files and summary stats.
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import os
import sys
from collections import Counter, defaultdict
import re

# ─── Configuration ───────────────────────────────────────────────────────
# Paths relative to macs-thesis/ repo root for reproducibility.
# macs-thesis/
# ├── Data/moltbook_upload.db          (gitignored)
# └── Proposals/Exploratory/
#     ├── OpinionLeaderAnalysis.py      (this script)
#     └── analysis_results/             (output)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))  # macs-thesis/

DB_PATH = os.path.join(REPO_ROOT, "Data", "moltbook_upload.db")
TOP_PERCENTILE = 5  # Top 5% by degree centrality = "opinion leaders"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "analysis_results")

def main(db_path):
    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path)

    print("=" * 60)
    print("STEP 1: Loading data...")
    print("=" * 60)

    # Load agents
    agents = pd.read_sql("SELECT * FROM agents", conn)
    print(f"  Agents: {len(agents):,}")

    # Load posts (select key columns to save memory)
    posts = pd.read_sql("""
        SELECT id, title, content, submolt, author_id, author_name,
               upvotes, downvotes, comment_count, created_at
        FROM posts
    """, conn)
    print(f"  Posts: {len(posts):,}")

    # Load comment edges (author_id commenting on posts by post author)
    # We need: commenter_id -> post_author_id edges
    print("  Loading comment interactions (this may take a moment)...")
    interactions = pd.read_sql("""
        SELECT c.author_id AS commenter_id,
               p.author_id AS post_author_id,
               COUNT(*) AS weight
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.author_id != p.author_id
        GROUP BY c.author_id, p.author_id
    """, conn)
    print(f"  Unique interaction edges: {len(interactions):,}")

    # Also load comment-to-comment interactions (reply network)
    comment_replies = pd.read_sql("""
        SELECT c1.author_id AS replier_id,
               c2.author_id AS parent_author_id,
               COUNT(*) AS weight
        FROM comments c1
        JOIN comments c2 ON c1.parent_id = c2.id
        WHERE c1.author_id != c2.author_id
        GROUP BY c1.author_id, c2.author_id
    """, conn)
    print(f"  Comment reply edges: {len(comment_replies):,}")

    print("\n" + "=" * 60)
    print("STEP 2: Building interaction network & computing centrality...")
    print("=" * 60)

    # Combine both interaction types
    # Edge: A -> B means A directed something at B (commented on B's post or replied to B's comment)
    all_edges = pd.concat([
        interactions.rename(columns={"commenter_id": "source", "post_author_id": "target"}),
        comment_replies.rename(columns={"replier_id": "source", "parent_author_id": "target"})
    ])
    all_edges = all_edges.groupby(["source", "target"])["weight"].sum().reset_index()
    print(f"  Combined edges: {len(all_edges):,}")

    # Compute degree centrality manually (avoid networkx memory issues for large graphs)
    # In-degree: how many unique agents interact WITH this agent (= popularity/influence)
    # Out-degree: how many unique agents this agent interacts with
    in_degree = all_edges.groupby("target")["source"].nunique().reset_index()
    in_degree.columns = ["agent_id", "in_degree"]

    out_degree = all_edges.groupby("source")["target"].nunique().reset_index()
    out_degree.columns = ["agent_id", "out_degree"]

    # Total weighted interactions received
    in_weight = all_edges.groupby("target")["weight"].sum().reset_index()
    in_weight.columns = ["agent_id", "in_weight"]

    # Merge
    degree_df = pd.merge(in_degree, out_degree, on="agent_id", how="outer").fillna(0)
    degree_df = pd.merge(degree_df, in_weight, on="agent_id", how="left").fillna(0)
    degree_df["total_degree"] = degree_df["in_degree"] + degree_df["out_degree"]

    # Total number of nodes for centrality normalization
    all_nodes = set(all_edges["source"].unique()) | set(all_edges["target"].unique())
    n_nodes = len(all_nodes)
    print(f"  Nodes in network: {n_nodes:,}")

    degree_df["in_degree_centrality"] = degree_df["in_degree"] / (n_nodes - 1)
    degree_df["out_degree_centrality"] = degree_df["out_degree"] / (n_nodes - 1)
    degree_df["total_degree_centrality"] = degree_df["total_degree"] / (2 * (n_nodes - 1))

    # Identify opinion leaders: top 5% by in-degree centrality
    threshold = degree_df["in_degree"].quantile(1 - TOP_PERCENTILE / 100)
    degree_df["is_opinion_leader"] = degree_df["in_degree"] >= threshold
    n_leaders = degree_df["is_opinion_leader"].sum()
    print(f"  In-degree threshold (top {TOP_PERCENTILE}%): >= {threshold}")
    print(f"  Opinion leaders identified: {n_leaders:,}")

    leader_ids = set(degree_df[degree_df["is_opinion_leader"]]["agent_id"])
    non_leader_ids = set(degree_df[~degree_df["is_opinion_leader"]]["agent_id"])

    print("\n" + "=" * 60)
    print("STEP 3: Analyzing post content features...")
    print("=" * 60)

    posts["is_leader_post"] = posts["author_id"].isin(leader_ids)

    # Compute text features
    posts["content_length"] = posts["content"].fillna("").str.len()
    posts["title_length"] = posts["title"].fillna("").str.len()
    posts["word_count"] = posts["content"].fillna("").str.split().str.len()
    posts["has_url_in_content"] = posts["content"].fillna("").str.contains(r'https?://', regex=True)
    posts["has_question_mark"] = posts["content"].fillna("").str.contains(r'\?')
    posts["has_exclamation"] = posts["content"].fillna("").str.contains(r'!')
    posts["num_sentences"] = posts["content"].fillna("").str.count(r'[.!?]+')
    posts["avg_word_length"] = posts["content"].fillna("").apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.strip() else 0
    )
    posts["net_upvotes"] = posts["upvotes"] - posts["downvotes"]

    # Compare leader vs non-leader posts
    feature_cols = [
        "content_length", "title_length", "word_count", "upvotes", "downvotes",
        "net_upvotes", "comment_count", "has_url_in_content", "has_question_mark",
        "has_exclamation", "num_sentences", "avg_word_length"
    ]

    comparison = posts.groupby("is_leader_post")[feature_cols].agg(["mean", "median", "std"])
    comparison.columns = ["_".join(col) for col in comparison.columns]
    comparison.index = ["Non-Leader Posts", "Leader Posts"]
    print("\n  Post Feature Comparison (Mean):")
    for col in feature_cols:
        leader_val = comparison.loc["Leader Posts", f"{col}_mean"]
        non_leader_val = comparison.loc["Non-Leader Posts", f"{col}_mean"]
        ratio = leader_val / non_leader_val if non_leader_val != 0 else float("inf")
        print(f"    {col:25s}  Leader: {leader_val:10.2f}  Non-Leader: {non_leader_val:10.2f}  Ratio: {ratio:.2f}x")

    # Leader posting frequency
    leader_post_counts = posts[posts["is_leader_post"]].groupby("author_id").size()
    non_leader_post_counts = posts[~posts["is_leader_post"]].groupby("author_id").size()
    print(f"\n  Posts per leader (mean): {leader_post_counts.mean():.1f}")
    print(f"  Posts per non-leader (mean): {non_leader_post_counts.mean():.1f}")

    # Top submolts for leaders vs non-leaders
    leader_submolts = posts[posts["is_leader_post"]]["submolt"].value_counts().head(20)
    non_leader_submolts = posts[~posts["is_leader_post"]]["submolt"].value_counts().head(20)

    print("\n  Top 10 submolts for leader posts:")
    for s, c in leader_submolts.head(10).items():
        print(f"    {s}: {c:,}")

    print("\n  Top 10 submolts for non-leader posts:")
    for s, c in non_leader_submolts.head(10).items():
        print(f"    {s}: {c:,}")

    print("\n" + "=" * 60)
    print("STEP 4: Analyzing agent description features...")
    print("=" * 60)

    agents["is_opinion_leader"] = agents["id"].isin(leader_ids)

    # Description features
    agents["desc_length"] = agents["description"].fillna("").str.len()
    agents["desc_word_count"] = agents["description"].fillna("").str.split().str.len()
    agents["has_description"] = agents["description"].fillna("").str.len() > 0
    agents["desc_has_url"] = agents["description"].fillna("").str.contains(r'https?://', regex=True)
    agents["desc_num_sentences"] = agents["description"].fillna("").str.count(r'[.!?]+')
    agents["has_x_handle"] = agents["x_handle"].notna() & (agents["x_handle"] != "")

    # Keyword analysis in descriptions
    keywords_to_check = [
        "AI", "artificial intelligence", "machine learning", "bot", "agent",
        "research", "science", "technology", "creative", "art",
        "help", "assist", "expert", "professional", "specialist",
        "community", "social", "friend", "chat", "conversation",
        "news", "update", "information", "knowledge",
        "fun", "humor", "joke", "meme", "entertainment",
        "finance", "crypto", "trading", "invest",
        "code", "programming", "developer", "software",
        "health", "fitness", "wellness",
        "music", "gaming", "game",
        "philosophy", "ethics", "debate"
    ]

    for kw in keywords_to_check:
        agents[f"desc_has_{kw.lower().replace(' ', '_')}"] = (
            agents["description"].fillna("").str.lower().str.contains(kw.lower(), regex=False)
        )

    # Compare leader vs non-leader agent profiles
    desc_feature_cols = [
        "desc_length", "desc_word_count", "has_description", "desc_has_url",
        "desc_num_sentences", "has_x_handle", "karma", "follower_count", "following_count"
    ]

    desc_comparison = agents.groupby("is_opinion_leader")[desc_feature_cols].agg(["mean", "median"])
    desc_comparison.columns = ["_".join(col) for col in desc_comparison.columns]
    desc_comparison.index = ["Non-Leader", "Leader"]
    print("\n  Agent Profile Comparison (Mean):")
    for col in desc_feature_cols:
        leader_val = desc_comparison.loc["Leader", f"{col}_mean"]
        non_leader_val = desc_comparison.loc["Non-Leader", f"{col}_mean"]
        ratio = leader_val / non_leader_val if non_leader_val != 0 else float("inf")
        print(f"    {col:25s}  Leader: {leader_val:10.2f}  Non-Leader: {non_leader_val:10.2f}  Ratio: {ratio:.2f}x")

    # Keyword prevalence comparison
    kw_cols = [c for c in agents.columns if c.startswith("desc_has_") and c != "desc_has_url"]
    print("\n  Keyword Prevalence in Descriptions (Leader vs Non-Leader):")
    kw_results = []
    for col in kw_cols:
        kw_name = col.replace("desc_has_", "")
        leader_pct = agents[agents["is_opinion_leader"]][col].mean() * 100
        non_leader_pct = agents[~agents["is_opinion_leader"]][col].mean() * 100
        diff = leader_pct - non_leader_pct
        kw_results.append({"keyword": kw_name, "leader_pct": leader_pct,
                           "non_leader_pct": non_leader_pct, "diff": diff})

    kw_results = sorted(kw_results, key=lambda x: abs(x["diff"]), reverse=True)
    for r in kw_results[:20]:
        print(f"    {r['keyword']:25s}  Leader: {r['leader_pct']:5.1f}%  Non-Leader: {r['non_leader_pct']:5.1f}%  Diff: {r['diff']:+.1f}pp")

    print("\n" + "=" * 60)
    print("STEP 5: Saving results...")
    print("=" * 60)

    # Save degree centrality data
    degree_df_merged = pd.merge(degree_df, agents[["id", "name", "description", "karma",
                                                     "follower_count", "following_count"]],
                                 left_on="agent_id", right_on="id", how="left")
    degree_df_merged.to_csv(os.path.join(OUTPUT_DIR, "degree_centrality.csv"), index=False)
    print(f"  Saved: {OUTPUT_DIR}/degree_centrality.csv")

    # Save top leaders details
    top_leaders = degree_df_merged[degree_df_merged["is_opinion_leader"]].sort_values(
        "in_degree", ascending=False
    ).head(100)
    top_leaders.to_csv(os.path.join(OUTPUT_DIR, "top_100_leaders.csv"), index=False)
    print(f"  Saved: {OUTPUT_DIR}/top_100_leaders.csv")

    # Save leader posts
    leader_posts = posts[posts["is_leader_post"]].copy()
    leader_posts.to_csv(os.path.join(OUTPUT_DIR, "leader_posts.csv"), index=False)
    print(f"  Saved: {OUTPUT_DIR}/leader_posts.csv ({len(leader_posts):,} posts)")

    # Save post feature comparison
    comparison.to_csv(os.path.join(OUTPUT_DIR, "post_feature_comparison.csv"))
    print(f"  Saved: {OUTPUT_DIR}/post_feature_comparison.csv")

    # Save agent description comparison
    desc_comparison.to_csv(os.path.join(OUTPUT_DIR, "description_feature_comparison.csv"))
    print(f"  Saved: {OUTPUT_DIR}/description_feature_comparison.csv")

    # Save keyword analysis
    kw_df = pd.DataFrame(kw_results)
    kw_df.to_csv(os.path.join(OUTPUT_DIR, "keyword_analysis.csv"), index=False)
    print(f"  Saved: {OUTPUT_DIR}/keyword_analysis.csv")

    # Save leader descriptions (for qualitative analysis)
    leader_agents = agents[agents["is_opinion_leader"]][
        ["id", "name", "description", "karma", "follower_count", "following_count"]
    ].sort_values("karma", ascending=False)
    leader_agents.to_csv(os.path.join(OUTPUT_DIR, "leader_agent_profiles.csv"), index=False)
    print(f"  Saved: {OUTPUT_DIR}/leader_agent_profiles.csv ({len(leader_agents):,} agents)")

    # Save submolt distribution comparison
    submolt_comp = pd.DataFrame({
        "leader_count": leader_submolts,
        "non_leader_count": non_leader_submolts
    }).fillna(0).astype(int)
    submolt_comp.to_csv(os.path.join(OUTPUT_DIR, "submolt_distribution.csv"))
    print(f"  Saved: {OUTPUT_DIR}/submolt_distribution.csv")

    # Save summary stats as JSON
    summary = {
        "total_agents": len(agents),
        "agents_in_network": n_nodes,
        "n_opinion_leaders": int(n_leaders),
        "top_percentile": TOP_PERCENTILE,
        "in_degree_threshold": float(threshold),
        "leader_posts_count": int(leader_posts.shape[0]),
        "non_leader_posts_count": int(posts[~posts["is_leader_post"]].shape[0]),
        "total_edges": int(len(all_edges)),
        "mean_in_degree_leaders": float(degree_df[degree_df["is_opinion_leader"]]["in_degree"].mean()),
        "mean_in_degree_non_leaders": float(degree_df[~degree_df["is_opinion_leader"]]["in_degree"].mean()),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR}/summary.json")

    conn.close()
    print("\n" + "=" * 60)
    print("DONE! All results saved to:", OUTPUT_DIR)
    print("=" * 60)

if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    main(db)