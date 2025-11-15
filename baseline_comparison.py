"""
Comparison script: Baseline RAG vs Multi-Signal RAG
Shows improvement from each component
"""

import numpy as np
from typing import List, Dict
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# BASELINE RETRIEVER (Standard RAG)
# ============================================================================

class BaselineRAG:
    """
    Standard RAG: Pure semantic similarity, fixed top-k
    No multi-signal scoring, no budget awareness, no structure
    """
    
    def __init__(self, indexer):
        self.indexer = indexer
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """Simple cosine similarity retrieval"""
        
        # Encode query
        query_embedding = self.indexer.encoder.encode([query])[0]
        
        # Compute cosine similarity with all chunks
        similarities = np.dot(
            self.indexer.chunk_embeddings,
            query_embedding
        ) / (
            np.linalg.norm(self.indexer.chunk_embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )
        
        # Return top-k
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        all_chunk_ids = list(self.indexer.chunks.keys())
        
        return [all_chunk_ids[idx] for idx in top_k_indices]


# ============================================================================
# ABLATION VARIANTS
# ============================================================================

class SemanticPlusLexicalRetriever:
    """Baseline + BM25 (ablation 1)"""
    
    def __init__(self, indexer, scorer):
        self.indexer = indexer
        self.scorer = scorer
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        query_embedding = self.indexer.encoder.encode([query])[0]
        
        # Get candidates
        similarities = np.dot(self.indexer.chunk_embeddings, query_embedding) / (
            np.linalg.norm(self.indexer.chunk_embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )
        top_50_indices = np.argsort(similarities)[::-1][:50]
        all_chunk_ids = list(self.indexer.chunks.keys())
        candidates = [all_chunk_ids[idx] for idx in top_50_indices]
        
        # Score with semantic + lexical only
        tokenized_query = query.lower().split()
        all_bm25_scores = self.scorer.bm25.get_scores(tokenized_query)
        
        combined_scores = {}
        for cid in candidates:
            idx = self.indexer.chunk_id_to_idx[cid]
            chunk_emb = self.indexer.chunk_embeddings[idx]
            
            semantic = np.dot(query_embedding, chunk_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
            )
            lexical = all_bm25_scores[idx]
            
            # Normalize lexical
            max_lex = max(all_bm25_scores[top_50_indices])
            lexical = lexical / max_lex if max_lex > 0 else 0
            
            combined_scores[cid] = 0.6 * semantic + 0.4 * lexical
        
        # Return top-k
        sorted_chunks = sorted(combined_scores.keys(),
                              key=lambda x: combined_scores[x],
                              reverse=True)
        return sorted_chunks[:top_k]


class FullMultiSignalRetriever:
    """All signals but no budget awareness (ablation 2)"""
    
    def __init__(self, indexer, scorer):
        self.indexer = indexer
        self.scorer = scorer
    
    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        query_embedding = self.indexer.encoder.encode([query])[0]
        
        # Get candidates
        similarities = np.dot(self.indexer.chunk_embeddings, query_embedding) / (
            np.linalg.norm(self.indexer.chunk_embeddings, axis=1) *
            np.linalg.norm(query_embedding)
        )
        top_50_indices = np.argsort(similarities)[::-1][:50]
        all_chunk_ids = list(self.indexer.chunks.keys())
        candidates = [all_chunk_ids[idx] for idx in top_50_indices]
        
        # Full multi-signal scoring
        scores = self.scorer.score_chunks(query, query_embedding, candidates)
        
        # Return top-k (no budget awareness)
        sorted_chunks = sorted(scores.keys(),
                              key=lambda x: scores[x],
                              reverse=True)
        return sorted_chunks[:top_k]


# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================

def run_comprehensive_comparison(qasper_path: str, num_papers: int = 10):
    """
    Compare multiple retrieval strategies
    """
    
    print("="*70)
    print("COMPREHENSIVE RETRIEVAL COMPARISON")
    print("="*70)
    
    # Setup
    from qasper_implementation import QASPERLoader, SimpleIndexer, SimpleScorer, QASPEREvaluator, SimpleTier1Retriever
    
    print("\n[Setup] Loading and indexing...")
    loader = QASPERLoader(qasper_path)
    paper_ids = loader.get_paper_ids()[:num_papers]
    
    indexer = SimpleIndexer()
    for paper_id in tqdm(paper_ids, desc="Indexing"):
        sections = loader.get_paper_sections(paper_id)
        indexer.index_paper(paper_id, sections, loader)
    
    scorer = SimpleScorer(indexer)
    evaluator = QASPEREvaluator(indexer, loader)
    
    # Initialize all retrievers
    retrievers = {
        'Baseline (Semantic Only)': BaselineRAG(indexer),
        'Semantic + Lexical': SemanticPlusLexicalRetriever(indexer, scorer),
        'Multi-Signal (No Budget)': FullMultiSignalRetriever(indexer, scorer),
        'Multi-Signal + Budget (Ours)': SimpleTier1Retriever(indexer, scorer, token_budget=4000)
    }
    
    # Evaluate each retriever
    all_results = {}
    
    for name, retriever in retrievers.items():
        print(f"\n[Evaluating] {name}...")
        
        results = {
            'recall@5': [],
            'recall@10': [],
            'precision@5': [],
            'precision@10': [],
            'mrr': [],
            'answer_coverage': [],
            'avg_tokens': []
        }
        
        for paper_id in tqdm(paper_ids, desc=name, leave=False):
            questions = loader.get_questions(paper_id)
            
            for q_data in questions:
                query = q_data['question']
                evidence = q_data['evidence']
                answer_spans = q_data['extractive_spans']
                
                # Retrieve
                retrieved = retriever.retrieve(query, top_k=10)
                
                # Compute metrics
                results['recall@5'].append(
                    evaluator.compute_recall_at_k(retrieved, evidence, k=5)
                )
                results['recall@10'].append(
                    evaluator.compute_recall_at_k(retrieved, evidence, k=10)
                )
                results['precision@5'].append(
                    evaluator.compute_precision_at_k(retrieved, evidence, k=5)
                )
                results['precision@10'].append(
                    evaluator.compute_precision_at_k(retrieved, evidence, k=10)
                )
                results['mrr'].append(
                    evaluator.compute_mrr(retrieved, evidence)
                )
                results['answer_coverage'].append(
                    evaluator.compute_answer_coverage(retrieved, answer_spans)
                )
                
                # Token count
                tokens = sum(indexer.chunks[cid].token_count for cid in retrieved)
                results['avg_tokens'].append(tokens)
        
        # Aggregate
        all_results[name] = {
            metric: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            for metric, values in results.items()
        }
    
    # Print comparison table
    print("\n" + "="*70)
    print("RESULTS COMPARISON")
    print("="*70)
    
    metrics_to_show = ['recall@5', 'recall@10', 'precision@5', 'mrr', 'answer_coverage']
    
    # Header
    print(f"\n{'Metric':<20} | " + " | ".join(f"{name[:15]:>15}" for name in retrievers.keys()))
    print("-" * 100)
    
    # Rows
    for metric in metrics_to_show:
        row = f"{metric:<20} | "
        values = []
        for name in retrievers.keys():
            mean = all_results[name][metric]['mean']
            values.append(mean)
            row += f"{mean:>15.3f} | "
        
        print(row)
        
        # Show improvement over baseline
        if metric != 'avg_tokens':
            baseline = values[0]
            improvements = [(v - baseline) / baseline * 100 if baseline > 0 else 0 
                          for v in values[1:]]
            imp_row = f"{'  (vs baseline)':<20} | {'':<17} | "
            imp_row += " | ".join(f"{imp:>+14.1f}%" for imp in improvements)
            print(imp_row)
    
    # Token efficiency
    print("\n" + "-" * 100)
    print(f"\n{'Avg Tokens Used':<20} | " + 
          " | ".join(f"{all_results[name]['avg_tokens']['mean']:>15.1f}" 
                     for name in retrievers.keys()))
    
    # Calculate efficiency (quality per token)
    print(f"\n{'Efficiency (R@10/1K tokens)':<20} | ", end="")
    for name in retrievers.keys():
        recall = all_results[name]['recall@10']['mean']
        tokens = all_results[name]['avg_tokens']['mean']
        efficiency = (recall / tokens) * 1000 if tokens > 0 else 0
        print(f"{efficiency:>15.3f} | ", end="")
    print()
    
    print("\n" + "="*70)
    
    # Save results
    output = {
        'comparison': all_results,
        'num_papers': num_papers,
        'num_questions': sum(len(loader.get_questions(pid)) for pid in paper_ids)
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nResults saved to: comparison_results.json")
    
    return all_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(results: Dict):
    """Create visualization of comparison results"""
    
    metrics = ['recall@5', 'recall@10', 'precision@5', 'mrr', 'answer_coverage']
    retrievers = list(results.keys())
    
    # Extract means
    data = []
    for metric in metrics:
        for retriever in retrievers:
            data.append({
                'Metric': metric,
                'Retriever': retriever,
                'Score': results[retriever][metric]['mean']
            })
    
    # Create grouped bar chart
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Retrieval Performance Comparison', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        metric_data = [results[r][metric]['mean'] for r in retrievers]
        metric_std = [results[r][metric]['std'] for r in retrievers]
        
        x = np.arange(len(retrievers))
        bars = ax.bar(x, metric_data, yerr=metric_std, capsize=5, alpha=0.8)
        
        # Color bars (baseline in gray, ours in green)
        colors = ['gray', 'orange', 'blue', 'green']
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([r.split()[0] for r in retrievers], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('retrieval_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to: retrieval_comparison.png")
    plt.close()


# ============================================================================
# STATISTICAL SIGNIFICANCE TEST
# ============================================================================

def compute_statistical_significance(qasper_path: str, num_papers: int = 10):
    """
    Perform paired t-test to check if improvements are statistically significant
    """
    from scipy import stats
    from qasper_implementation import QASPERLoader, SimpleIndexer, SimpleScorer, QASPEREvaluator, SimpleTier1Retriever
    
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("="*70)
    
    # Setup
    loader = QASPERLoader(qasper_path)
    paper_ids = loader.get_paper_ids()[:num_papers]
    
    indexer = SimpleIndexer()
    for paper_id in tqdm(paper_ids, desc="Indexing"):
        sections = loader.get_paper_sections(paper_id)
        indexer.index_paper(paper_id, sections, loader)
    
    scorer = SimpleScorer(indexer)
    evaluator = QASPEREvaluator(indexer, loader)
    
    # Compare baseline vs ours
    baseline = BaselineRAG(indexer)
    ours = SimpleTier1Retriever(indexer, scorer, token_budget=4000)
    
    # Collect paired samples
    baseline_recalls = []
    ours_recalls = []
    
    for paper_id in paper_ids:
        questions = loader.get_questions(paper_id)
        
        for q_data in questions:
            query = q_data['question']
            evidence = q_data['evidence']
            
            # Baseline
            baseline_retrieved = baseline.retrieve(query, top_k=10)
            baseline_recall = evaluator.compute_recall_at_k(baseline_retrieved, evidence, k=10)
            baseline_recalls.append(baseline_recall)
            
            # Ours
            ours_retrieved = ours.retrieve(query, top_k=10)
            ours_recall = evaluator.compute_recall_at_k(ours_retrieved, evidence, k=10)
            ours_recalls.append(ours_recall)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(ours_recalls, baseline_recalls)
    
    print(f"\nPaired t-test (Recall@10):")
    print(f"  Baseline mean: {np.mean(baseline_recalls):.3f}")
    print(f"  Ours mean: {np.mean(ours_recalls):.3f}")
    print(f"  Improvement: {(np.mean(ours_recalls) - np.mean(baseline_recalls)) / np.mean(baseline_recalls) * 100:+.1f}%")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  ✓ Statistically significant at α=0.05")
    else:
        print(f"  ✗ Not statistically significant at α=0.05")
    
    print("="*70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    QASPER_PATH = "qasper-dev-v0.3.json"
    
    # Run comprehensive comparison
    results = run_comprehensive_comparison(QASPER_PATH, num_papers=100)
    
    # Visualize
    plot_comparison(results)
    
    # Statistical test
    compute_statistical_significance(QASPER_PATH, num_papers=100)