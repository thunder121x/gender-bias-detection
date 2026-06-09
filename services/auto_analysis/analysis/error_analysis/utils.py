"""
Utility functions for error analysis.
Handles file I/O, YAML processing, and directory management.
"""

from pathlib import Path
from typing import Dict, List, Any
import yaml


def load_incorrect_items(yaml_path: str) -> List[Dict[str, Any]]:
    """
    Load incorrect items from YAML file.
    
    Args:
        yaml_path: Path to incorrect_items.yaml
    
    Returns:
        List of incorrect item dictionaries
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if isinstance(data, dict) and 'records' in data:
        return data['records']
    return data if isinstance(data, list) else []


def save_separated_items(output_dir: str, group_name: str, items_dict: Dict[str, List[Dict]]):
    """
    Save separated items to individual YAML files grouped by label.
    
    Args:
        output_dir: Output directory path
        group_name: Name of grouping (e.g., 'group_by_correct_label')
        items_dict: Dictionary mapping labels to lists of items
    """
    output_path = Path(output_dir) / group_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    for label, items in items_dict.items():
        # Create safe filename from label
        safe_label = label.replace('/', '_').replace('\\', '_')
        file_path = output_path / f"{safe_label}.yaml"
        
        # Save items
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump({'records': items}, f, allow_unicode=True, default_flow_style=False)
        
        print(f"  ✓ Saved {len(items)} items to {file_path.name}")


def save_summary(output_dir: str, group_name: str, analysis: Dict[str, Dict]):
    """
    Save analysis summary to YAML file.
    
    Args:
        output_dir: Output directory path
        group_name: Name of grouping (e.g., 'group_by_correct_label')
        analysis: Analysis dictionary from ErrorAnalyzer
    """
    output_path = Path(output_dir) / group_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / "SUMMARY.yaml"
    
    # Prepare summary (remove sample_errors for cleaner summary)
    summary_data = {}
    for label, stats in analysis.items():
        summary_data[label] = {
            'error_count': stats['count'],
            'most_common_issue': (
                f"Mispredicted as: {stats.get('most_common_misprediction')}" 
                if 'most_common_misprediction' in stats 
                else f"Actually: {stats.get('most_common_correct_label')}"
            ),
            'key_improvements': stats['key_improvements'],
        }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump(summary_data, f, allow_unicode=True, default_flow_style=False)
    
    print(f"  ✓ Saved summary to {summary_file.name}")


def print_analysis_summary(analysis: Dict[str, Dict], group_name: str):
    """
    Print human-readable analysis summary.
    
    Args:
        analysis: Analysis dictionary from ErrorAnalyzer
        group_name: Name of grouping for display
    """
    print(f"\n{'='*70}")
    print(f"ANALYSIS BY {group_name.upper()}")
    print(f"{'='*70}\n")
    
    total_errors = sum(stats['count'] for stats in analysis.values())
    print(f"Total Errors: {total_errors}\n")
    
    for label, stats in sorted(analysis.items(), key=lambda x: x[1]['count'], reverse=True):
        print(f"\n{label.upper()}")
        print(f"  Errors: {stats['count']}")
        
        if 'most_common_misprediction' in stats:
            print(f"  Most Common Misprediction: {stats['most_common_misprediction']}")
        else:
            print(f"  Most Common Correct Label: {stats['most_common_correct_label']}")
        
        print(f"  Key Improvements:")
        for i, improvement in enumerate(stats['key_improvements'], 1):
            print(f"    {i}. {improvement}")
        
        # Show sample errors
        print(f"  Sample Error:")
        if stats['sample_errors']:
            sample = stats['sample_errors'][0]
            text = sample.get('text', '')[:80]
            reason = sample.get('reason', '')[:80]
            print(f"    Text: {text}...")
            print(f"    Reason: {reason}...")
