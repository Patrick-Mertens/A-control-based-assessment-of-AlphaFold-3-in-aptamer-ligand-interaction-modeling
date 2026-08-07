#!/usr/bin/env python3
"""
analyze_aptamer_comprehensive.py

Comprehensive analysis of AlphaFold3 aptamer-ligand binding predictions across
multiple experimental conditions (Mg intervals, buffer compositions).

Analyzes data at two levels:
1. SUMMARY LEVEL: Best ranking model per aptamer (from *_summary_confidences.json at top level)
2. SEED LEVEL: Individual seed-sample predictions (from seed-X_sample-Y directories)

Supports multiple study types:
- Mg interval studies (Mg0, Mg2, Mg4, Mg6, Mg8, Mg10)
- Buffer studies (optimized: 4MG, proportional: 1MG)
- With/without weights variants

Key metrics extracted:
- iPTM (interface predicted TM-score)
- pTM (predicted TM-score)  
- Ranking score
- DNA-Ligand interface iPTM (chain_pair_iptm[0][1])
- DNA-Ligand interface PAE (chain_pair_pae_min[0][1])
- DNA chain pTM (chain_ptm[0])

Usage:
    python analyze_aptamer_comprehensive.py --study buffer --output-dir ~/afoutput_buffer
    python analyze_aptamer_comprehensive.py --study mg --mg-levels 0 2 4 6 8 10
    python analyze_aptamer_comprehensive.py --compare-all --export comprehensive_results.csv
"""

import argparse
import json
import re
import sys
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics
import csv


@dataclass
class SeedMetrics:
    """
    Metrics from a single seed-sample prediction.
    
    Terminology:
    - Global iPTM/pTM: Overall scores across all chains (including buffer ions)
    - Aptamer-Target iPTM: Specific interface between DNA (chain A) and ligand (chain B)
      This is chain_pair_iptm[0][1] - the KEY metric for binding prediction
    """
    aptamer_id: str
    target: str
    condition: str  # e.g., "Mg4", "buffer_optimized", "buffer_proportional"
    seed: int
    sample: int
    
    # Global metrics (across all chains including buffer ions)
    global_iptm: float = 0.0      # Overall interface pTM (iptm field)
    global_ptm: float = 0.0       # Overall predicted TM (ptm field)
    ranking_score: float = 0.0
    fraction_disordered: float = 0.0
    has_clash: bool = False
    
    # Aptamer-Target SPECIFIC metrics (DNA chain A <-> Ligand chain B)
    # This is THE key metric for aptamer-ligand binding prediction
    aptamer_target_iptm: float = 0.0   # chain_pair_iptm[0][1]
    aptamer_target_pae: float = 0.0    # chain_pair_pae_min[0][1] (lower = better)
    
    # Individual chain metrics
    aptamer_ptm: float = 0.0      # chain_ptm[0] - DNA structure confidence


@dataclass
class AptamerConditionSummary:
    """
    Aggregated metrics for one aptamer under one condition.
    
    Two levels of data:
    1. SUMMARY LEVEL (best_*): From top-level summary_confidences.json
       - This IS the best ranking model, selected as max(ranking_score) across all seeds
       - So best_aptamer_target_iptm = MAX aptamer-target interface iPTM across all 45 seed-samples
    
    2. SEED LEVEL (mean_*, std_*): Statistics across all 45 seed-sample predictions
       - More robust estimate of binding confidence distribution
       - mean = average across 9 seeds × 5 samples = 45 predictions
       - std = standard deviation showing prediction variance
    
    Ratio Interpretation (cocaine/morphine):
       - Ratio > 1.0 = Aptamer binds cocaine better (cocaine-specific)
       - Ratio < 1.0 = Aptamer binds morphine better
       - Ratio ≈ 1.0 = Non-specific binding
    """
    aptamer_id: str
    target: str
    condition: str
    
    # =========================================================================
    # SUMMARY LEVEL (from best ranking model = top-level summary_confidences.json)
    # These represent the BEST/MAX values across all seed-samples
    # =========================================================================
    # Global metrics
    best_global_iptm: float = 0.0       # Best overall iPTM
    best_global_ptm: float = 0.0        # Best overall pTM
    best_ranking: float = 0.0           # Best ranking score
    
    # Aptamer-Target SPECIFIC (KEY METRICS for binding prediction)
    best_aptamer_target_iptm: float = 0.0   # MAX chain_pair_iptm[0][1]
    best_aptamer_target_pae: float = 0.0    # MIN chain_pair_pae_min[0][1]
    
    # Chain metrics
    best_aptamer_ptm: float = 0.0       # DNA chain pTM
    
    # =========================================================================
    # SEED LEVEL (statistics across all 45 seed-sample predictions)
    # More robust than single best model, captures prediction variance
    # =========================================================================
    n_seeds: int = 0  # Number of seed-sample predictions (typically 9×5=45)
    
    # Global metrics - seed averages
    mean_global_iptm: float = 0.0
    mean_global_ptm: float = 0.0
    std_global_iptm: float = 0.0
    mean_ranking: float = 0.0
    
    # Aptamer-Target SPECIFIC - seed averages (KEY METRICS)
    mean_aptamer_target_iptm: float = 0.0   # Average DNA-Ligand interface iPTM
    mean_aptamer_target_pae: float = 0.0    # Average DNA-Ligand PAE
    std_aptamer_target_iptm: float = 0.0    # Variance in predictions
    
    # Chain metrics - seed averages
    mean_aptamer_ptm: float = 0.0
    
    # Raw seed values for detailed analysis
    all_seed_global_iptms: List[float] = field(default_factory=list)
    all_seed_aptamer_target_iptms: List[float] = field(default_factory=list)
    all_seed_aptamer_target_paes: List[float] = field(default_factory=list)


def extract_metrics_from_json(json_path: Path) -> Optional[Dict[str, Any]]:
    """
    Extract metrics from a summary_confidences JSON file.
    
    JSON structure uses lists indexed by chain order:
    - Index 0 = Chain A = DNA aptamer
    - Index 1 = Chain B = Target ligand (cocaine/morphine)
    - Index 2+ = Buffer components (TRS, NA, K, MG, CL)
    
    Key metrics:
    - global_iptm: Overall interface pTM across ALL chains (includes buffer ions)
    - global_ptm: Overall predicted TM
    - aptamer_target_iptm: chain_pair_iptm[0][1] - SPECIFIC to DNA-Ligand interface (KEY!)
    - aptamer_target_pae: chain_pair_pae_min[0][1] - DNA-Ligand PAE (lower = better)
    - aptamer_ptm: chain_ptm[0] - DNA structure confidence
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return None
    
    metrics = {}
    
    # Global metrics (across ALL chains including buffer ions)
    metrics['global_iptm'] = data.get('iptm', 0.0) or 0.0
    metrics['global_ptm'] = data.get('ptm', 0.0) or 0.0
    metrics['ranking_score'] = data.get('ranking_score', 0.0) or 0.0
    metrics['fraction_disordered'] = data.get('fraction_disordered', 0.0) or 0.0
    metrics['has_clash'] = bool(data.get('has_clash', 0))
    
    # Aptamer-Target SPECIFIC interface metrics (KEY for binding prediction)
    # chain_pair_iptm[0][1] = interface iPTM between DNA (A) and Ligand (B)
    chain_pair_iptm = data.get('chain_pair_iptm', [])
    if chain_pair_iptm and len(chain_pair_iptm) > 0 and len(chain_pair_iptm[0]) > 1:
        val = chain_pair_iptm[0][1]
        metrics['aptamer_target_iptm'] = val if val is not None else 0.0
    else:
        metrics['aptamer_target_iptm'] = 0.0
    
    # chain_pair_pae_min[0][1] = minimum PAE between DNA and Ligand
    chain_pair_pae = data.get('chain_pair_pae_min', [])
    if chain_pair_pae and len(chain_pair_pae) > 0 and len(chain_pair_pae[0]) > 1:
        val = chain_pair_pae[0][1]
        metrics['aptamer_target_pae'] = val if val is not None else 99.0
    else:
        metrics['aptamer_target_pae'] = 99.0
    
    # DNA (aptamer) chain pTM - structure confidence for DNA
    chain_ptm = data.get('chain_ptm', [])
    if chain_ptm and len(chain_ptm) > 0:
        val = chain_ptm[0]
        metrics['aptamer_ptm'] = val if val is not None else 0.0
    else:
        metrics['aptamer_ptm'] = 0.0
    
    return metrics


def parse_seed_sample_dir(dirname: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse seed and sample from directory name like 'seed-1_sample-0'."""
    match = re.match(r'seed-(\d+)_sample-(\d+)', dirname)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_aptamer_target_from_dir(dirname: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse aptamer ID and target from directory name.
    
    Handles patterns:
    - Buffer study: NC101_cocaine_1TRS_2NA_1K_4MG_12CL (multiple buffer components)
    - Mg study with 'complex': NC101_cocaine_complex_4MG
    - Mg study simple: NC101_cocaine_4MG
    
    Pattern priority is important to avoid mismatches.
    """
    # Pattern 1: Mg study with 'complex' keyword
    # Example: NC101_cocaine_complex_4MG, NC101_morphine_complex_10MG
    match = re.match(r'^(NC[A-Z0-9]+)_(cocaine|morphine)_complex_?\d*MG?', dirname, re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2).lower()
    
    # Pattern 2: Buffer study - has multiple underscore-separated components after target
    # Example: NC101_cocaine_1TRS_2NA_1K_4MG_12CL (at least 3 components: TRS, NA/K/MG, CL)
    # Key distinguisher: buffer patterns have at least 4 underscores total
    if dirname.count('_') >= 4:
        match = re.match(r'^(NC[A-Z0-9]+)_(cocaine|morphine)_\d+[A-Z]+_', dirname, re.IGNORECASE)
        if match:
            return match.group(1).upper(), match.group(2).lower()
    
    # Pattern 3: Mg study simple - NC{id}_{target}_{N}MG (exactly 3 parts)
    # Example: NC101_cocaine_4MG
    match = re.match(r'^(NC[A-Z0-9]+)_(cocaine|morphine)_(\d+MG)$', dirname, re.IGNORECASE)
    if match:
        return match.group(1).upper(), match.group(2).lower()
    
    # Fallback: Split by underscore and extract first two parts
    parts = dirname.split('_')
    if len(parts) >= 2 and parts[0].upper().startswith('NC'):
        target = parts[1].lower()
        if target in ['cocaine', 'morphine']:
            return parts[0].upper(), target
    
    return None, None


def find_mg_study_dirs(base_path: Path, mg_levels: List[int], targets: List[str], 
                       with_weights: bool = True) -> Dict[str, Dict[str, Path]]:
    """
    Find Mg study output directories.
    Returns: {condition: {target: path}}
    """
    dirs = {}
    
    for mg in mg_levels:
        condition = f"Mg{mg}"
        if with_weights:
            condition += "_weights"
        dirs[condition] = {}
        
        for target in targets:
            # Pattern: afoutput_Mg{N}_{target}_target_and_aptamer[_with_weights]
            if target == 'cocaine':
                if with_weights:
                    dir_name = f"afoutput_Mg{mg}_cocaine_target_and_aptamer_with_weights"
                else:
                    dir_name = f"afoutput_Mg{mg}_cocaine_target_and_aptamer"
            else:  # morphine
                if with_weights:
                    dir_name = f"afoutput_Mg{mg}_Morphine_target_cocaine_aptamer_with_weights"
                else:
                    dir_name = f"afoutput_Mg{mg}_Morphine_target_cocaine_aptamer"
            
            dir_path = base_path / dir_name
            if dir_path.exists():
                dirs[condition][target] = dir_path
    
    return dirs


def analyze_single_directory(output_dir: Path, condition: str, verbose: bool = True) -> Dict[str, Dict[str, AptamerConditionSummary]]:
    """
    Analyze a single output directory at both summary and seed levels.
    Returns: {aptamer_id: {target: AptamerConditionSummary}}
    """
    results = defaultdict(dict)
    
    if not output_dir.exists():
        print(f"  Warning: Directory not found: {output_dir}", file=sys.stderr)
        return dict(results)
    
    subdirs_processed = 0
    subdirs_skipped = 0
    
    for subdir in output_dir.iterdir():
        if not subdir.is_dir():
            continue
        
        # Skip non-aptamer directories (temp files, logs, etc.)
        if subdir.name.startswith('.') or subdir.name in ['TERMS_OF_USE.md', 'logs', 'temp']:
            continue
        
        aptamer_id, target = parse_aptamer_target_from_dir(subdir.name)
        if not aptamer_id or not target:
            subdirs_skipped += 1
            if verbose and subdirs_skipped <= 3:
                print(f"    Skipped (could not parse): {subdir.name}", file=sys.stderr)
            continue
        
        summary = AptamerConditionSummary(
            aptamer_id=aptamer_id,
            target=target,
            condition=condition
        )
        
        # 1. Find and parse top-level summary (best ranking model)
        # Look for *_summary_confidences.json but NOT in seed subdirectories
        top_summary_files = [
            f for f in subdir.glob('*_summary_confidences.json')
            if 'seed-' not in f.name and f.is_file()
        ]
        
        if top_summary_files:
            metrics = extract_metrics_from_json(top_summary_files[0])
            if metrics:
                summary.best_global_iptm = metrics['global_iptm']
                summary.best_global_ptm = metrics['global_ptm']
                summary.best_ranking = metrics['ranking_score']
                summary.best_aptamer_target_iptm = metrics['aptamer_target_iptm']
                summary.best_aptamer_target_pae = metrics['aptamer_target_pae']
                summary.best_aptamer_ptm = metrics['aptamer_ptm']
            else:
                print(f"    Warning: Could not parse {top_summary_files[0].name}", file=sys.stderr)
        else:
            if verbose:
                print(f"    Warning: No top-level summary_confidences.json in {subdir.name}", file=sys.stderr)
        
        # 2. Find and parse all seed-sample directories
        seed_metrics_list = []
        
        for seed_dir in subdir.iterdir():
            if not seed_dir.is_dir():
                continue
            
            seed, sample = parse_seed_sample_dir(seed_dir.name)
            if seed is None:
                continue
            
            # Find summary_confidences in seed dir
            seed_summary_files = list(seed_dir.glob('*_summary_confidences.json'))
            if not seed_summary_files:
                continue
            
            metrics = extract_metrics_from_json(seed_summary_files[0])
            if metrics:
                seed_metrics = SeedMetrics(
                    aptamer_id=aptamer_id,
                    target=target,
                    condition=condition,
                    seed=seed,
                    sample=sample,
                    global_iptm=metrics['global_iptm'],
                    global_ptm=metrics['global_ptm'],
                    ranking_score=metrics['ranking_score'],
                    aptamer_target_iptm=metrics['aptamer_target_iptm'],
                    aptamer_target_pae=metrics['aptamer_target_pae'],
                    aptamer_ptm=metrics['aptamer_ptm'],
                    fraction_disordered=metrics['fraction_disordered'],
                    has_clash=metrics['has_clash']
                )
                seed_metrics_list.append(seed_metrics)
        
        # 3. Aggregate seed-level statistics
        if seed_metrics_list:
            summary.n_seeds = len(seed_metrics_list)
            
            global_iptms = [s.global_iptm for s in seed_metrics_list]
            global_ptms = [s.global_ptm for s in seed_metrics_list]
            aptamer_target_iptms = [s.aptamer_target_iptm for s in seed_metrics_list]
            aptamer_target_paes = [s.aptamer_target_pae for s in seed_metrics_list]
            aptamer_ptms = [s.aptamer_ptm for s in seed_metrics_list]
            rankings = [s.ranking_score for s in seed_metrics_list]
            
            summary.mean_global_iptm = statistics.mean(global_iptms)
            summary.mean_global_ptm = statistics.mean(global_ptms)
            summary.mean_aptamer_target_iptm = statistics.mean(aptamer_target_iptms)
            summary.mean_aptamer_target_pae = statistics.mean(aptamer_target_paes)
            summary.mean_aptamer_ptm = statistics.mean(aptamer_ptms)
            summary.mean_ranking = statistics.mean(rankings)
            
            if len(global_iptms) > 1:
                summary.std_global_iptm = statistics.stdev(global_iptms)
                summary.std_aptamer_target_iptm = statistics.stdev(aptamer_target_iptms)
            
            summary.all_seed_global_iptms = global_iptms
            summary.all_seed_aptamer_target_iptms = aptamer_target_iptms
            summary.all_seed_aptamer_target_paes = aptamer_target_paes
        
        results[aptamer_id][target] = summary
        subdirs_processed += 1
    
    if verbose and subdirs_skipped > 3:
        print(f"    ... and {subdirs_skipped - 3} more skipped directories", file=sys.stderr)
    
    return dict(results)


def print_condition_summary(data: Dict[str, Dict[str, AptamerConditionSummary]], 
                            condition: str, targets: List[str]):
    """Print summary statistics for one condition."""
    print(f"\n{'='*120}")
    print(f"CONDITION: {condition}")
    print(f"{'='*120}")
    
    for target in targets:
        summaries = [d[target] for d in data.values() if target in d]
        if not summaries:
            continue
        
        n = len(summaries)
        
        # Best ranking model stats
        best_global_iptms = [s.best_global_iptm for s in summaries]
        best_global_ptms = [s.best_global_ptm for s in summaries]
        best_aptamer_target_iptms = [s.best_aptamer_target_iptm for s in summaries]
        best_paes = [s.best_aptamer_target_pae for s in summaries]
        best_rankings = [s.best_ranking for s in summaries]
        
        # Seed-level stats
        seed_summaries = [s for s in summaries if s.n_seeds > 0]
        mean_global_iptms = [s.mean_global_iptm for s in seed_summaries]
        mean_global_ptms = [s.mean_global_ptm for s in seed_summaries]
        mean_aptamer_target_iptms = [s.mean_aptamer_target_iptm for s in seed_summaries]
        mean_paes = [s.mean_aptamer_target_pae for s in seed_summaries]
        
        print(f"\n  {target.upper()} (n={n}):")
        print(f"  {'-'*90}")
        print(f"  BEST RANKING MODEL (Summary Level = MAX across all seeds):")
        print(f"    Global iPTM:           mean={statistics.mean(best_global_iptms):.3f}, "
              f"min={min(best_global_iptms):.3f}, max={max(best_global_iptms):.3f}")
        print(f"    Global pTM:            mean={statistics.mean(best_global_ptms):.3f}, "
              f"min={min(best_global_ptms):.3f}, max={max(best_global_ptms):.3f}")
        print(f"    Aptamer-Target iPTM:   mean={statistics.mean(best_aptamer_target_iptms):.3f}, "
              f"min={min(best_aptamer_target_iptms):.3f}, max={max(best_aptamer_target_iptms):.3f}" +
              (f", std={statistics.stdev(best_aptamer_target_iptms):.3f}" if len(best_aptamer_target_iptms) > 1 else ""))
        print(f"    Aptamer-Target PAE:    mean={statistics.mean(best_paes):.2f}, "
              f"min={min(best_paes):.2f}, max={max(best_paes):.2f}")
        print(f"    Ranking Score:         mean={statistics.mean(best_rankings):.3f}")
        
        if mean_aptamer_target_iptms:
            print(f"\n  SEED-LEVEL AVERAGE (Mean across 9 seeds × 5 samples = 45 predictions):")
            print(f"    Global iPTM:           mean={statistics.mean(mean_global_iptms):.3f}, "
                  f"min={min(mean_global_iptms):.3f}, max={max(mean_global_iptms):.3f}")
            print(f"    Global pTM:            mean={statistics.mean(mean_global_ptms):.3f}, "
                  f"min={min(mean_global_ptms):.3f}, max={max(mean_global_ptms):.3f}")
            print(f"    Aptamer-Target iPTM:   mean={statistics.mean(mean_aptamer_target_iptms):.3f}, "
                  f"min={min(mean_aptamer_target_iptms):.3f}, max={max(mean_aptamer_target_iptms):.3f}")
            print(f"    Aptamer-Target PAE:    mean={statistics.mean(mean_paes):.2f}")
            
            # Total seeds
            total_seeds = sum(s.n_seeds for s in summaries)
            print(f"    Total seed-samples:    {total_seeds}")


def print_specificity_comparison(data: Dict[str, Dict[str, AptamerConditionSummary]], 
                                  condition: str, primary: str = 'cocaine', 
                                  secondary: str = 'morphine', top_n: int = 15):
    """Print specificity comparison between two targets."""
    print(f"\n{'='*120}")
    print(f"SPECIFICITY ANALYSIS: {primary.upper()} vs {secondary.upper()} [{condition}]")
    print(f"{'='*120}")
    
    # Calculate ratios for each aptamer
    comparisons = []
    for aptamer_id, targets in data.items():
        if primary in targets and secondary in targets:
            p = targets[primary]
            s = targets[secondary]
            
            # Best model ratio
            best_ratio = p.best_aptamer_target_iptm / s.best_aptamer_target_iptm if s.best_aptamer_target_iptm > 0 else float('inf')
            
            # Seed-level ratio
            seed_ratio = p.mean_aptamer_target_iptm / s.mean_aptamer_target_iptm if s.mean_aptamer_target_iptm > 0 else float('inf')
            
            comparisons.append({
                'aptamer_id': aptamer_id,
                'primary': p,
                'secondary': s,
                'best_ratio': best_ratio,
                'seed_ratio': seed_ratio,
                'best_diff': p.best_aptamer_target_iptm - s.best_aptamer_target_iptm,
                'seed_diff': p.mean_aptamer_target_iptm - s.mean_aptamer_target_iptm,
            })
    
    if not comparisons:
        print("No aptamers found with both targets.")
        return
    
    # Sort by seed-level ratio (more robust than single best model)
    comparisons.sort(key=lambda x: x['seed_ratio'], reverse=True)
    
    print(f"\nTop {top_n} Most Specific (by Seed-Level iPTM Ratio):")
    print(f"{'-'*120}")
    print(f"{'Rank':<5} | {'Aptamer':<10} | {primary+' Best':>11} | {primary+' Seed':>11} | "
          f"{secondary+' Best':>11} | {secondary+' Seed':>11} | {'Best Ratio':>10} | {'Seed Ratio':>10}")
    print(f"{'-'*120}")
    
    for i, c in enumerate(comparisons[:top_n], 1):
        br = f"{c['best_ratio']:.3f}" if c['best_ratio'] < 100 else ">100"
        sr = f"{c['seed_ratio']:.3f}" if c['seed_ratio'] < 100 else ">100"
        print(f"{i:<5} | {c['aptamer_id']:<10} | {c['primary'].best_aptamer_target_iptm:>11.3f} | "
              f"{c['primary'].mean_aptamer_target_iptm:>11.3f} | {c['secondary'].best_aptamer_target_iptm:>11.3f} | "
              f"{c['secondary'].mean_aptamer_target_iptm:>11.3f} | {br:>10} | {sr:>10}")
    
    # Bottom performers
    print(f"\nBottom 10 (Better for {secondary}):")
    print(f"{'-'*120}")
    for i, c in enumerate(reversed(comparisons[-10:]), 1):
        br = f"{c['best_ratio']:.3f}" if c['best_ratio'] < 100 else ">100"
        sr = f"{c['seed_ratio']:.3f}" if c['seed_ratio'] < 100 else ">100"
        print(f"{i:<5} | {c['aptamer_id']:<10} | {c['primary'].best_aptamer_target_iptm:>11.3f} | "
              f"{c['primary'].mean_aptamer_target_iptm:>11.3f} | {c['secondary'].best_aptamer_target_iptm:>11.3f} | "
              f"{c['secondary'].mean_aptamer_target_iptm:>11.3f} | {br:>10} | {sr:>10}")
    
    # Overall statistics
    best_ratios = [c['best_ratio'] for c in comparisons if c['best_ratio'] < float('inf')]
    seed_ratios = [c['seed_ratio'] for c in comparisons if c['seed_ratio'] < float('inf')]
    
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS (n={len(comparisons)} aptamers)")
    print(f"{'='*80}")
    print(f"\nBest Model Ratio ({primary}/{secondary}):")
    print(f"  Mean:   {statistics.mean(best_ratios):.3f}")
    print(f"  Median: {statistics.median(best_ratios):.3f}")
    print(f"  Std:    {statistics.stdev(best_ratios):.3f}" if len(best_ratios) > 1 else "")
    print(f"  Ratio > 1.0: {sum(1 for r in best_ratios if r > 1.0)} / {len(best_ratios)}")
    
    print(f"\nSeed-Level Ratio ({primary}/{secondary}):")
    print(f"  Mean:   {statistics.mean(seed_ratios):.3f}")
    print(f"  Median: {statistics.median(seed_ratios):.3f}")
    print(f"  Std:    {statistics.stdev(seed_ratios):.3f}" if len(seed_ratios) > 1 else "")
    print(f"  Ratio > 1.0: {sum(1 for r in seed_ratios if r > 1.0)} / {len(seed_ratios)}")
    print(f"  Ratio > 1.1: {sum(1 for r in seed_ratios if r > 1.1)} / {len(seed_ratios)}")
    print(f"  Ratio > 1.2: {sum(1 for r in seed_ratios if r > 1.2)} / {len(seed_ratios)}")


def print_cross_condition_comparison(all_data: Dict[str, Dict[str, Dict[str, AptamerConditionSummary]]],
                                      targets: List[str]):
    """Compare metrics across all conditions."""
    print(f"\n{'='*140}")
    print("CROSS-CONDITION COMPARISON")
    print(f"{'='*140}")
    
    # Sort conditions: buffer first, then Mg in numeric order
    def condition_sort_key(cond):
        if cond.startswith('buffer'):
            return (0, cond)
        elif cond.startswith('Mg'):
            # Extract number from Mg0, Mg2, etc.
            try:
                num = int(''.join(filter(str.isdigit, cond.split('_')[0])))
                return (1, num)
            except:
                return (1, 999)
        return (2, cond)
    
    conditions = sorted(all_data.keys(), key=condition_sort_key)
    
    # Table 1: Seed-Level Interface iPTM by condition
    print(f"\n{'─'*100}")
    print("SEED-LEVEL INTERFACE iPTM (Mean ± Std)")
    print(f"{'─'*100}")
    print(f"{'Condition':<25} | {'Cocaine':^30} | {'Morphine':^30} | {'Ratio':>8}")
    print(f"{'':<25} | {'Mean':>10} {'Std':>8} {'N':>6} | {'Mean':>10} {'Std':>8} {'N':>6} | {'C/M':>8}")
    print(f"{'─'*100}")
    
    for condition in conditions:
        coc_summaries = [d['cocaine'] for d in all_data[condition].values() 
                        if 'cocaine' in d and d['cocaine'].n_seeds > 0]
        mor_summaries = [d['morphine'] for d in all_data[condition].values() 
                        if 'morphine' in d and d['morphine'].n_seeds > 0]
        
        if not coc_summaries and not mor_summaries:
            continue
        
        # Cocaine stats
        if coc_summaries:
            coc_means = [s.mean_aptamer_target_iptm for s in coc_summaries]
            coc_mean = statistics.mean(coc_means)
            coc_std = statistics.stdev(coc_means) if len(coc_means) > 1 else 0
            coc_n = len(coc_summaries)
            coc_str = f"{coc_mean:>10.3f} {coc_std:>8.3f} {coc_n:>6}"
        else:
            coc_mean = 0
            coc_str = f"{'N/A':>10} {'-':>8} {0:>6}"
        
        # Morphine stats
        if mor_summaries:
            mor_means = [s.mean_aptamer_target_iptm for s in mor_summaries]
            mor_mean = statistics.mean(mor_means)
            mor_std = statistics.stdev(mor_means) if len(mor_means) > 1 else 0
            mor_n = len(mor_summaries)
            mor_str = f"{mor_mean:>10.3f} {mor_std:>8.3f} {mor_n:>6}"
        else:
            mor_mean = 0
            mor_str = f"{'N/A':>10} {'-':>8} {0:>6}"
        
        # Ratio
        if coc_mean > 0 and mor_mean > 0:
            ratio = coc_mean / mor_mean
            ratio_str = f"{ratio:>8.3f}"
        else:
            ratio_str = f"{'N/A':>8}"
        
        print(f"{condition:<25} | {coc_str} | {mor_str} | {ratio_str}")
    
    print(f"{'─'*100}")
    
    # Table 2: Best Model Interface iPTM
    print(f"\n{'─'*100}")
    print("BEST MODEL INTERFACE iPTM (Mean across aptamers)")
    print(f"{'─'*100}")
    print(f"{'Condition':<25} | {'Cocaine':>12} | {'Morphine':>12} | {'Diff':>10} | {'Ratio':>8}")
    print(f"{'─'*100}")
    
    for condition in conditions:
        coc_summaries = [d['cocaine'] for d in all_data[condition].values() if 'cocaine' in d]
        mor_summaries = [d['morphine'] for d in all_data[condition].values() if 'morphine' in d]
        
        if not coc_summaries and not mor_summaries:
            continue
        
        coc_mean = statistics.mean([s.best_aptamer_target_iptm for s in coc_summaries]) if coc_summaries else 0
        mor_mean = statistics.mean([s.best_aptamer_target_iptm for s in mor_summaries]) if mor_summaries else 0
        
        diff = coc_mean - mor_mean if coc_mean and mor_mean else 0
        ratio = coc_mean / mor_mean if coc_mean and mor_mean else 0
        
        coc_str = f"{coc_mean:>12.3f}" if coc_summaries else f"{'N/A':>12}"
        mor_str = f"{mor_mean:>12.3f}" if mor_summaries else f"{'N/A':>12}"
        diff_str = f"{diff:>+10.3f}" if coc_summaries and mor_summaries else f"{'N/A':>10}"
        ratio_str = f"{ratio:>8.3f}" if coc_summaries and mor_summaries else f"{'N/A':>8}"
        
        print(f"{condition:<25} | {coc_str} | {mor_str} | {diff_str} | {ratio_str}")
    
    print(f"{'─'*100}")
    
    # Table 3: Specificity summary (how many aptamers favor cocaine)
    print(f"\n{'─'*100}")
    print("SPECIFICITY SUMMARY (Aptamers with Cocaine > Morphine)")
    print(f"{'─'*100}")
    print(f"{'Condition':<25} | {'Best Model':>20} | {'Seed-Level':>20} | {'Total Pairs':>12}")
    print(f"{'─'*100}")
    
    for condition in conditions:
        # Find aptamers with both targets
        pairs = [(apt_id, d) for apt_id, d in all_data[condition].items() 
                 if 'cocaine' in d and 'morphine' in d]
        
        if not pairs:
            continue
        
        n_total = len(pairs)
        
        # Best model comparison
        n_best_coc = sum(1 for apt_id, d in pairs 
                        if d['cocaine'].best_aptamer_target_iptm > d['morphine'].best_aptamer_target_iptm)
        
        # Seed-level comparison
        n_seed_coc = sum(1 for apt_id, d in pairs 
                        if d['cocaine'].n_seeds > 0 and d['morphine'].n_seeds > 0
                        and d['cocaine'].mean_aptamer_target_iptm > d['morphine'].mean_aptamer_target_iptm)
        n_seed_pairs = sum(1 for apt_id, d in pairs 
                          if d['cocaine'].n_seeds > 0 and d['morphine'].n_seeds > 0)
        
        best_str = f"{n_best_coc:>3} / {n_total:>3} ({100*n_best_coc/n_total:>5.1f}%)"
        seed_str = f"{n_seed_coc:>3} / {n_seed_pairs:>3} ({100*n_seed_coc/n_seed_pairs:>5.1f}%)" if n_seed_pairs > 0 else "N/A"
        
        print(f"{condition:<25} | {best_str:>20} | {seed_str:>20} | {n_total:>12}")
    
    print(f"{'─'*100}")


def export_comprehensive_csv(all_data: Dict[str, Dict[str, Dict[str, AptamerConditionSummary]]],
                              output_path: Path, targets: List[str]):
    """Export all data to CSV."""
    conditions = sorted(all_data.keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Build header
        header = ['aptamer_id']
        for condition in conditions:
            for target in targets:
                prefix = f"{condition}_{target}"
                header.extend([
                    f"{prefix}_best_aptamer_target_iptm",
                    f"{prefix}_best_interface_pae",
                    f"{prefix}_best_ranking",
                    f"{prefix}_mean_aptamer_target_iptm",
                    f"{prefix}_mean_interface_pae",
                    f"{prefix}_std_aptamer_target_iptm",
                    f"{prefix}_n_seeds",
                ])
        
        # Add specificity columns per condition
        for condition in conditions:
            header.extend([
                f"{condition}_best_iptm_ratio",
                f"{condition}_seed_iptm_ratio",
            ])
        
        writer.writerow(header)
        
        # Collect all aptamer IDs
        all_aptamers = set()
        for cond_data in all_data.values():
            all_aptamers.update(cond_data.keys())
        
        # Write data rows
        for aptamer_id in sorted(all_aptamers):
            row = [aptamer_id]
            
            for condition in conditions:
                for target in targets:
                    if aptamer_id in all_data[condition] and target in all_data[condition][aptamer_id]:
                        s = all_data[condition][aptamer_id][target]
                        row.extend([
                            f"{s.best_aptamer_target_iptm:.4f}",
                            f"{s.best_aptamer_target_pae:.4f}",
                            f"{s.best_ranking:.4f}",
                            f"{s.mean_aptamer_target_iptm:.4f}" if s.n_seeds > 0 else "",
                            f"{s.mean_aptamer_target_pae:.4f}" if s.n_seeds > 0 else "",
                            f"{s.std_aptamer_target_iptm:.4f}" if s.n_seeds > 1 else "",
                            s.n_seeds,
                        ])
                    else:
                        row.extend([''] * 7)
            
            # Specificity ratios per condition
            for condition in conditions:
                if (aptamer_id in all_data[condition] and 
                    targets[0] in all_data[condition][aptamer_id] and 
                    targets[1] in all_data[condition][aptamer_id]):
                    
                    p = all_data[condition][aptamer_id][targets[0]]
                    s = all_data[condition][aptamer_id][targets[1]]
                    
                    best_ratio = p.best_aptamer_target_iptm / s.best_aptamer_target_iptm if s.best_aptamer_target_iptm > 0 else ""
                    seed_ratio = p.mean_aptamer_target_iptm / s.mean_aptamer_target_iptm if s.mean_aptamer_target_iptm > 0 else ""
                    
                    row.extend([
                        f"{best_ratio:.4f}" if isinstance(best_ratio, float) else "",
                        f"{seed_ratio:.4f}" if isinstance(seed_ratio, float) else "",
                    ])
                else:
                    row.extend(['', ''])
            
            writer.writerow(row)
    
    print(f"\nExported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis of AlphaFold3 aptamer binding predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ===========================================================================
  # RECOMMENDED: Analyze ALL conditions at once (buffer + proportional + Mg)
  # ===========================================================================
  %(prog)s --study all --base-dir ~ --export ~/comprehensive_results.csv
  
  # ===========================================================================
  # Analyze single buffer study
  # ===========================================================================
  %(prog)s --study buffer --output-dir ~/afoutput_buffer --condition-name optimized
  
  # ===========================================================================
  # Analyze Mg interval study only (with weights)
  # ===========================================================================
  %(prog)s --study mg --base-dir ~ --mg-levels 0 2 4 6 8 10 --with-weights
  
  # ===========================================================================
  # Compare specific directories manually
  # ===========================================================================
  %(prog)s --compare-dirs ~/afoutput_buffer ~/afoutput_proportional \\
           --condition-names optimized proportional --export comparison.csv
  
  # ===========================================================================
  # Full analysis with custom paths and verbose output
  # ===========================================================================
  %(prog)s --study all --base-dir ~ \\
           --buffer-optimized-dir afoutput_buffer \\
           --buffer-proportional-dir afoutput_proportional \\
           --export ~/full_analysis.csv --top 20 --verbose

Directory Structure Expected:
  ~/afoutput_buffer/                    -> Buffer optimized (1TRS_2NA_1K_4MG_12CL)
  ~/afoutput_proportional/              -> Buffer proportional (1TRS_8NA_1K_1MG_12CL)
  ~/afoutput_Mg0_cocaine_target_and_aptamer_with_weights/
  ~/afoutput_Mg0_Morphine_target_cocaine_aptamer_with_weights/
  ~/afoutput_Mg2_cocaine_target_and_aptamer_with_weights/
  ... (Mg2, Mg4, Mg6, Mg8, Mg10)
        """
    )
    
    parser.add_argument("--study", choices=["buffer", "mg", "all"], help="Type of study to analyze")
    parser.add_argument("--output-dir", type=Path, help="Single output directory to analyze")
    parser.add_argument("--base-dir", type=Path, default=Path.home(), help="Base directory for all studies")
    parser.add_argument("--condition-name", default="buffer", help="Name for the condition being analyzed")
    
    parser.add_argument("--compare-dirs", type=Path, nargs="+", help="Multiple directories to compare")
    parser.add_argument("--condition-names", nargs="+", help="Names for compared conditions")
    
    parser.add_argument("--mg-levels", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10], 
                        help="Mg levels to analyze")
    parser.add_argument("--with-weights", action="store_true", help="Use _with_weights directories for Mg study")
    parser.add_argument("--no-weights", action="store_true", help="Also include non-weights Mg directories")
    
    # Buffer directory names (can be customized)
    parser.add_argument("--buffer-optimized-dir", default="afoutput_buffer", 
                        help="Directory name for optimized buffer results")
    parser.add_argument("--buffer-proportional-dir", default="afoutput_proportional",
                        help="Directory name for proportional buffer results")
    
    parser.add_argument("--targets", nargs="+", default=["cocaine", "morphine"], help="Targets to analyze")
    parser.add_argument("--export", type=Path, help="Export results to CSV")
    parser.add_argument("--top", type=int, default=15, help="Number of top performers to show")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress and warnings")
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("ALPHAFOLD3 COMPREHENSIVE APTAMER ANALYSIS")
    print("=" * 100)
    
    all_data: Dict[str, Dict[str, Dict[str, AptamerConditionSummary]]] = {}
    
    # Handle different analysis modes
    if args.compare_dirs:
        # Compare multiple directories
        condition_names = args.condition_names or [f"condition_{i}" for i in range(len(args.compare_dirs))]
        
        for dir_path, cond_name in zip(args.compare_dirs, condition_names):
            print(f"\nAnalyzing: {dir_path} as '{cond_name}'")
            data = analyze_single_directory(dir_path, cond_name, verbose=args.verbose)
            all_data[cond_name] = data
            print(f"  Found {len(data)} aptamers")
    
    elif args.study == "all":
        # Comprehensive analysis: Buffer + Proportional + All Mg intervals (with weights)
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS: All Conditions")
        print(f"{'='*80}")
        print(f"Base directory: {args.base_dir}")
        
        # 1. Buffer optimized
        buffer_opt_path = args.base_dir / args.buffer_optimized_dir
        if buffer_opt_path.exists():
            print(f"\n[1/3] Analyzing Buffer Optimized: {buffer_opt_path.name}")
            data = analyze_single_directory(buffer_opt_path, "buffer_optimized", verbose=args.verbose)
            all_data["buffer_optimized"] = data
            n_cocaine = sum(1 for d in data.values() if 'cocaine' in d)
            n_morphine = sum(1 for d in data.values() if 'morphine' in d)
            print(f"      Found {len(data)} aptamers (cocaine: {n_cocaine}, morphine: {n_morphine})")
        else:
            print(f"\n[1/3] Skipped Buffer Optimized (not found): {buffer_opt_path}")
        
        # 2. Buffer proportional
        buffer_prop_path = args.base_dir / args.buffer_proportional_dir
        if buffer_prop_path.exists():
            print(f"\n[2/3] Analyzing Buffer Proportional: {buffer_prop_path.name}")
            data = analyze_single_directory(buffer_prop_path, "buffer_proportional", verbose=args.verbose)
            all_data["buffer_proportional"] = data
            n_cocaine = sum(1 for d in data.values() if 'cocaine' in d)
            n_morphine = sum(1 for d in data.values() if 'morphine' in d)
            print(f"      Found {len(data)} aptamers (cocaine: {n_cocaine}, morphine: {n_morphine})")
        else:
            print(f"\n[2/3] Skipped Buffer Proportional (not found): {buffer_prop_path}")
        
        # 3. Mg interval study (with weights by default)
        print(f"\n[3/3] Analyzing Mg Interval Study (levels: {args.mg_levels})")
        
        for mg_level in args.mg_levels:
            condition = f"Mg{mg_level}"
            
            # Cocaine directory
            cocaine_dir = f"afoutput_Mg{mg_level}_cocaine_target_and_aptamer_with_weights"
            cocaine_path = args.base_dir / cocaine_dir
            
            # Morphine directory (note: "Morphine" is capitalized in your naming)
            morphine_dir = f"afoutput_Mg{mg_level}_Morphine_target_cocaine_aptamer_with_weights"
            morphine_path = args.base_dir / morphine_dir
            
            if condition not in all_data:
                all_data[condition] = {}
            
            # Process cocaine
            if cocaine_path.exists():
                print(f"      {condition} cocaine: {cocaine_dir}")
                data = analyze_single_directory(cocaine_path, condition, verbose=args.verbose)
                for apt_id, targets_data in data.items():
                    if apt_id not in all_data[condition]:
                        all_data[condition][apt_id] = {}
                    all_data[condition][apt_id].update(targets_data)
            else:
                print(f"      {condition} cocaine: SKIPPED (not found)")
            
            # Process morphine
            if morphine_path.exists():
                print(f"      {condition} morphine: {morphine_dir}")
                data = analyze_single_directory(morphine_path, condition, verbose=args.verbose)
                for apt_id, targets_data in data.items():
                    if apt_id not in all_data[condition]:
                        all_data[condition][apt_id] = {}
                    all_data[condition][apt_id].update(targets_data)
            else:
                print(f"      {condition} morphine: SKIPPED (not found)")
            
            # Count results for this Mg level
            if all_data[condition]:
                n_apt = len(all_data[condition])
                n_coc = sum(1 for d in all_data[condition].values() if 'cocaine' in d)
                n_mor = sum(1 for d in all_data[condition].values() if 'morphine' in d)
                print(f"      {condition} total: {n_apt} aptamers (cocaine: {n_coc}, morphine: {n_mor})")
    
    elif args.study == "buffer":
        # Single buffer directory
        if not args.output_dir:
            parser.error("--output-dir required for buffer study")
        
        print(f"\nAnalyzing: {args.output_dir}")
        data = analyze_single_directory(args.output_dir, args.condition_name, verbose=args.verbose)
        all_data[args.condition_name] = data
        print(f"Found {len(data)} aptamers")
    
    elif args.study == "mg":
        # Mg interval study
        print(f"\nAnalyzing Mg interval study (levels: {args.mg_levels})")
        print(f"With weights: {args.with_weights}")
        
        for mg_level in args.mg_levels:
            condition = f"Mg{mg_level}"
            if args.with_weights:
                condition += "_weights"
            
            # Find directories for each target
            for target in args.targets:
                if target == 'cocaine':
                    if args.with_weights:
                        dir_name = f"afoutput_Mg{mg_level}_cocaine_target_and_aptamer_with_weights"
                    else:
                        dir_name = f"afoutput_Mg{mg_level}_cocaine_target_and_aptamer"
                else:
                    if args.with_weights:
                        dir_name = f"afoutput_Mg{mg_level}_Morphine_target_cocaine_aptamer_with_weights"
                    else:
                        dir_name = f"afoutput_Mg{mg_level}_Morphine_target_cocaine_aptamer"
                
                dir_path = args.base_dir / dir_name
                if dir_path.exists():
                    print(f"  Analyzing: {dir_name}")
                    data = analyze_single_directory(dir_path, condition, verbose=args.verbose)
                    
                    if condition not in all_data:
                        all_data[condition] = {}
                    
                    # Merge into all_data
                    for apt_id, targets_data in data.items():
                        if apt_id not in all_data[condition]:
                            all_data[condition][apt_id] = {}
                        all_data[condition][apt_id].update(targets_data)
                else:
                    if args.verbose:
                        print(f"  Skipped (not found): {dir_name}")
    
    else:
        parser.error("Must specify --study or --compare-dirs")
    
    # Print results for each condition
    for condition, data in all_data.items():
        print_condition_summary(data, condition, args.targets)
        
        if len(args.targets) >= 2:
            print_specificity_comparison(data, condition, args.targets[0], args.targets[1], args.top)
    
    # Cross-condition comparison if multiple conditions
    if len(all_data) > 1:
        print_cross_condition_comparison(all_data, args.targets)
    
    # Export if requested
    if args.export:
        export_comprehensive_csv(all_data, args.export, args.targets)
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print("=" * 100)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
