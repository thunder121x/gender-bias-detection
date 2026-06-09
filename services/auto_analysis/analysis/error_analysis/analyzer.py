"""
Analyzer module for error analysis.
Handles separation of incorrect items and generation of improvement summaries.
"""

from collections import defaultdict
from pathlib import Path
import yaml
from typing import Dict, List, Any, Tuple


class ErrorAnalyzer:
    """Analyzes incorrect predictions and generates summaries."""
    
    def __init__(self, incorrect_items: List[Dict[str, Any]]):
        """
        Initialize the analyzer with incorrect items.
        
        Args:
            incorrect_items: List of incorrect item dictionaries
        """
        self.incorrect_items = incorrect_items
        self.by_correct_label = defaultdict(list)
        self.by_predicted_label = defaultdict(list)
        self._organize_items()
    
    def _organize_items(self):
        """Organize items by correct_label and predicted_label."""
        for item in self.incorrect_items:
            correct_label = item.get('correct_label', 'unknown')
            predicted_label = item.get('predicted_label', 'unknown')
            
            self.by_correct_label[correct_label].append(item)
            self.by_predicted_label[predicted_label].append(item)
    
    def analyze_by_correct_label(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze errors grouped by correct_label.
        Returns summary with key improvements for each label.
        
        Returns:
            Dictionary with analysis for each correct_label
        """
        analysis = {}
        
        for label, items in self.by_correct_label.items():
            analysis[label] = {
                'count': len(items),
                'most_common_misprediction': self._get_most_common_misprediction(items),
                'key_improvements': self._generate_improvements(items, 'predicted_label'),
                'sample_errors': items[:3],  # First 3 as examples
            }
        
        return analysis
    
    def analyze_by_predicted_label(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze errors grouped by predicted_label.
        Returns summary with key improvements for each label.
        
        Returns:
            Dictionary with analysis for each predicted_label
        """
        analysis = {}
        
        for label, items in self.by_predicted_label.items():
            analysis[label] = {
                'count': len(items),
                'most_common_correct_label': self._get_most_common_correct_label(items),
                'key_improvements': self._generate_improvements(items, 'correct_label'),
                'sample_errors': items[:3],  # First 3 as examples
            }
        
        return analysis
    
    @staticmethod
    def _get_most_common_misprediction(items: List[Dict]) -> str:
        """Get the most common incorrect prediction for a correct label."""
        predictions = [item.get('predicted_label') for item in items]
        from collections import Counter
        counter = Counter(predictions)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else 'unknown'
    
    @staticmethod
    def _get_most_common_correct_label(items: List[Dict]) -> str:
        """Get the most common correct label for a predicted label."""
        correct_labels = [item.get('correct_label') for item in items]
        from collections import Counter
        counter = Counter(correct_labels)
        most_common = counter.most_common(1)
        return most_common[0][0] if most_common else 'unknown'
    
    @staticmethod
    def _generate_improvements(items: List[Dict], compare_field: str) -> List[str]:
        """
        Generate key improvements based on common patterns in errors.
        
        Args:
            items: List of incorrect items
            compare_field: Field to compare against ('predicted_label' or 'correct_label')
        
        Returns:
            List of improvement suggestions
        """
        improvements = []
        
        # Analyze common patterns in reasons
        reasons = [item.get('reason', '').lower() for item in items]
        
        # Look for common keywords in reasons
        keyword_patterns = {
            'stereotype': 'Reduce stereotyping detection - many items are social critique or humor',
            'attack': 'Improve attack vs criticism distinction - context matters',
            'insult': 'Better distinguish between insults and descriptive language',
            'normative': 'Improve gender norm detection - distinguish prescriptive vs descriptive',
            'context': 'Include more context analysis in decision making',
            'intent': 'Consider speaker intent more carefully',
            'behavior': 'Distinguish between individual behavior critique and group generalization',
        }
        
        for keyword, suggestion in keyword_patterns.items():
            if any(keyword in reason for reason in reasons):
                if suggestion not in improvements:
                    improvements.append(suggestion)
        
        # If no specific patterns found, add generic improvements
        if not improvements:
            improvements.append('Review annotation guidelines alignment')
            improvements.append('Analyze sample errors for common patterns')
        
        return improvements[:5]  # Return top 5 improvements
    
    def get_all_items_by_correct_label(self) -> Dict[str, List[Dict]]:
        """Return all items organized by correct_label."""
        return dict(self.by_correct_label)
    
    def get_all_items_by_predicted_label(self) -> Dict[str, List[Dict]]:
        """Return all items organized by predicted_label."""
        return dict(self.by_predicted_label)
