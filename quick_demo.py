"""
Quick Demo: End-to-end example of the RAG system
Run this to see the system in action on a single paper
"""

import json
import numpy as np
from typing import List, Dict

# Import from our implementation
# (In practice, these would be: from qasper_implementation import ...)

def quick_demo(qasper_path: str = "qasper-dev-v0.3.json"):
    """
    Demonstrates the complete system on ONE paper
    Shows retrieval for 3 different query types
    """
    
    print("="*70)
    print("MULTI-SIGNAL RAG SYSTEM - QUICK DEMO")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    
    print("\n[STEP 1] Loading QASPER dataset...")
    
    with open(qasper_path, 'r') as f:
        data = json.load(f)
    
    # Get first paper
    paper_id = list(data.keys())[0]
    paper = data[paper_id]
    
    print(f"Paper ID: {paper_id}")
    print(f"Title: {paper['title']}")
    print(f"Sections: {len(paper['full_text'])}")
    print(f"Questions available: {len(paper['qas'])}")
    
    # ========================================================================
    # STEP 2: Initialize System
    # ========================================================================
    
    print("\n[STEP 2] Initializing retrieval system...")
    
    from qasper_implementation import SimpleIndexer, SimpleScorer, SimpleTier1Retriever, QASPERLoader
    
    loader = QASPERLoader(qasper_path)
    indexer = SimpleIndexer()
    
    # Index this paper
    sections = loader.get_paper_sections(paper_id)
    indexer.index_paper(paper_id, sections, loader)
    
    print(f"Indexed {len(indexer.chunks)} chunks")
    
    # Initialize scorer and retriever
    scorer = SimpleScorer(indexer)
    retriever = SimpleTier1Retriever(indexer, scorer, token_budget=3000)
    
    print("✓ System ready")
    
    # ========================================================================
    # STEP 3: Demo Different Query Types
    # ========================================================================
    
    print("\n[STEP 3] Running retrieval examples...")
    
    # Get actual questions from paper
    questions = loader.get_questions(paper_id)
    
    if len(questions) >= 3:
        demo_questions = [
            questions[0]['question'],  # Actual question 1
            questions[1]['question'],  # Actual question 2
            questions[2]['question'] if len(questions) > 2 else questions[1]['question']
        ]
    else:
        # Fallback generic questions
        demo_questions = [
            "What is the main contribution of this paper?",
            "What datasets were used in the experiments?",
            "What were the main results and findings?"
        ]
    
    for i, query in enumerate(demo_questions, 1):
        print(f"\n{'─'*70}")
        print(f"Query {i}: {query}")
        print(f"{'─'*70}")
        
        # Retrieve
        retrieved = retriever.retrieve(query, top_k=5)
        
        print(f"\n✓ Retrieved {len(retrieved)} chunks")
        
        # Show results
        for rank, chunk_id in enumerate(retrieved, 1):
            chunk = indexer.chunks[chunk_id]
            
            print(f"\n[Rank {rank}]")
            print(f"  Section: {chunk.section_name} ({chunk.section_type})")
            print(f"  Position: {chunk.position:.2f}")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Content preview: {chunk.content[:150]}...")
        
        # Show token usage
        total_tokens = sum(indexer.chunks[cid].token_count for cid in retrieved)
        print(f"\n  Total tokens: {total_tokens} / 3000 budget")
        print(f"  Budget utilization: {total_tokens/3000*100:.1f}%")
    
    # ========================================================================
    # STEP 4: Compare with Baseline
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("[STEP 4] Comparing with baseline RAG...")
    print(f"{'='*70}")
    
    from baseline_comparison import BaselineRAG
    
    baseline = BaselineRAG(indexer)
    
    # Use first question
    query = demo_questions[0]
    print(f"\nQuery: {query}")
    
    # Baseline retrieval
    baseline_chunks = baseline.retrieve(query, top_k=5)
    
    # Our retrieval
    our_chunks = retriever.retrieve(query, top_k=5)
    
    print("\n[Baseline Retrieval]")
    for rank, chunk_id in enumerate(baseline_chunks, 1):
        chunk = indexer.chunks[chunk_id]
        print(f"  {rank}. {chunk.section_name} (pos={chunk.position:.2f})")
    
    print("\n[Our Multi-Signal Retrieval]")
    for rank, chunk_id in enumerate(our_chunks, 1):
        chunk = indexer.chunks[chunk_id]
        print(f"  {rank}. {chunk.section_name} (pos={chunk.position:.2f})")
    
    # Show differences
    baseline_set = set(baseline_chunks)
    our_set = set(our_chunks)
    
    only_baseline = baseline_set - our_set
    only_ours = our_set - baseline_set
    
    if only_ours:
        print(f"\n✓ Our method found {len(only_ours)} different chunks:")
        for chunk_id in only_ours:
            chunk = indexer.chunks[chunk_id]
            print(f"    • {chunk.section_name}")
    
    # ========================================================================
    # STEP 5: Show Signal Breakdown
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("[STEP 5] Signal breakdown for top result...")
    print(f"{'='*70}")
    
    # Get top chunk
    top_chunk_id = our_chunks[0]
    query_embedding = indexer.encoder.encode([query])[0]
    
    # Compute individual signals
    chunk = indexer.chunks[top_chunk_id]
    chunk_idx = indexer.chunk_id_to_idx[top_chunk_id]
    chunk_emb = indexer.chunk_embeddings[chunk_idx]
    
    # Semantic
    semantic = np.dot(query_embedding, chunk_emb) / (
        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
    )
    
    # Lexical
    tokenized_query = query.lower().split()
    bm25_scores = scorer.bm25.get_scores(tokenized_query)
    lexical = bm25_scores[chunk_idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0
    
    # Structural
    section_weight = scorer.section_weights.get(chunk.section_type, 1.0)
    
    # Position
    distance_from_edge = min(chunk.position, 1.0 - chunk.position)
    position = 1.0 - 0.4 * (2 * distance_from_edge) ** 2
    
    print(f"\nTop chunk: {chunk.section_name}")
    print(f"Content: {chunk.content[:200]}...\n")
    
    print("Signal Contributions:")
    print(f"  Semantic similarity:  {semantic:.3f} × 0.30 = {semantic * 0.30:.3f}")
    print(f"  Lexical match (BM25): {lexical:.3f} × 0.20 = {lexical * 0.20:.3f}")
    print(f"  Structural weight:    {section_weight:.3f} × 0.25 = {section_weight * 0.25:.3f}")
    print(f"  Position bias:        {position:.3f} × 0.15 = {position * 0.15:.3f}")
    
    final_score = (
        semantic * 0.30 +
        lexical * 0.20 +
        section_weight * 0.25 +
        position * 0.15 +
        section_weight * 0.10
    )
    print(f"\n  Final score: {final_score:.3f}")
    
    # ========================================================================
    # STEP 6: Evaluation on This Paper
    # ========================================================================
    
    if questions:
        print(f"\n{'='*70}")
        print(f"[STEP 6] Evaluating on {len(questions)} questions from this paper...")
        print(f"{'='*70}")
        
        from qasper_implementation import QASPEREvaluator
        
        evaluator = QASPEREvaluator(indexer, loader)
        
        recalls_5 = []
        recalls_10 = []
        mrrs = []
        
        for q_data in questions[:10]:  # Limit to 10 for demo
            query = q_data['question']
            evidence = q_data['evidence']
            
            retrieved = retriever.retrieve(query, top_k=10)
            
            recall_5 = evaluator.compute_recall_at_k(retrieved, evidence, k=5)
            recall_10 = evaluator.compute_recall_at_k(retrieved, evidence, k=10)
            mrr = evaluator.compute_mrr(retrieved, evidence)
            
            recalls_5.append(recall_5)
            recalls_10.append(recall_10)
            mrrs.append(mrr)
        
        print(f"\nResults on this paper:")
        print(f"  Recall@5:  {np.mean(recalls_5):.3f} ± {np.std(recalls_5):.3f}")
        print(f"  Recall@10: {np.mean(recalls_10):.3f} ± {np.std(recalls_10):.3f}")
        print(f"  MRR:       {np.mean(mrrs):.3f} ± {np.std(mrrs):.3f}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n{'='*70}")
    print("DEMO COMPLETE")
    print(f"{'='*70}")
    
    print("\nKey Takeaways:")
    print("  ✓ Multi-signal scoring considers semantic, lexical, structural cues")
    print("  ✓ Budget-aware selection optimizes token efficiency")
    print("  ✓ Structural awareness prioritizes important sections")
    print("  ✓ Position bias favors beginning/end of documents")
    
    print("\nNext steps:")
    print("  1. Run full evaluation: python qasper_implementation.py")
    print("  2. Compare methods: python baseline_comparison.py")
    print("  3. Tune hyperparameters for your use case")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(qasper_path: str = "qasper-dev-v0.3.json"):
    """
    Interactive mode: Ask questions about any indexed paper
    """
    
    from qasper_implementation import SimpleIndexer, SimpleScorer, SimpleTier1Retriever, QASPERLoader
    
    print("\n" + "="*70)
    print("INTERACTIVE RETRIEVAL MODE")
    print("="*70)
    
    # Load and index
    print("\nLoading and indexing papers... (this may take a minute)")
    
    loader = QASPERLoader(qasper_path)
    indexer = SimpleIndexer()
    
    # Index first 5 papers
    paper_ids = loader.get_paper_ids()[:5]
    
    for paper_id in paper_ids:
        sections = loader.get_paper_sections(paper_id)
        indexer.index_paper(paper_id, sections, loader)
    
    scorer = SimpleScorer(indexer)
    retriever = SimpleTier1Retriever(indexer, scorer, token_budget=3000)
    
    print(f"\n✓ Indexed {len(paper_ids)} papers ({len(indexer.chunks)} chunks total)")
    
    # List papers
    print("\nIndexed papers:")
    for i, paper_id in enumerate(paper_ids, 1):
        title = loader.data[paper_id]['title']
        print(f"  {i}. {title[:60]}...")
    
    # Interactive loop
    print("\nType your questions (or 'quit' to exit):")
    
    while True:
        query = input("\n> ")
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query.strip():
            continue
        
        # Retrieve
        retrieved = retriever.retrieve(query, top_k=5)
        
        print(f"\n✓ Found {len(retrieved)} relevant chunks:\n")
        
        for rank, chunk_id in enumerate(retrieved, 1):
            chunk = indexer.chunks[chunk_id]
            doc_title = loader.data[chunk.doc_id]['title']
            
            print(f"[{rank}] {chunk.section_name}")
            print(f"    Paper: {doc_title[:50]}...")
            print(f"    {chunk.content[:200]}...")
            print()
        
        total_tokens = sum(indexer.chunks[cid].token_count for cid in retrieved)
        print(f"Total tokens: {total_tokens}")
    
    print("\nGoodbye!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        quick_demo()