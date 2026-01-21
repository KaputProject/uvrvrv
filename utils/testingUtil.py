import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

TRANSACTION_FIELDS = ['date', 'reference', 'partner', 'description', 'change', 'balance', 'outgoing', 'known_partner']


def load_ground_truth(statement_number: str) -> Optional[Dict]:
    # Nalozi pravilno parsano JSON datoteko (ground truth)
    path = Path(f'data/data/{statement_number}.json')
    if not path.exists():
        print(f"[NAPAKA] Ground truth datoteka ne obstaja: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_string(s: str) -> str:
    # Normalizira niz za primerjavo
    if not isinstance(s, str):
        return str(s)
    return ' '.join(s.lower().strip().split())


def compare_values(ground_truth_val: Any, predicted_val: Any, field_name: str) -> bool:
    # Primerja dve vrednosti in vrne True ce se ujemata
    if predicted_val is None:
        return False
    
    if field_name in ['change', 'balance']:
        # Numericna primerjava s toleranco 1 cent
        try:
            gt_num = float(ground_truth_val) if ground_truth_val is not None else 0.0
            pred_num = float(predicted_val) if predicted_val is not None else 0.0
            return abs(gt_num - pred_num) < 0.01
        except (ValueError, TypeError):
            return False
            
    elif field_name in ['outgoing', 'known_partner']:
        # Boolean primerjava
        return bool(ground_truth_val) == bool(predicted_val)
    else:
        # Tekstovna primerjava
        gt_str = str(ground_truth_val) if ground_truth_val is not None else ''
        pred_str = str(predicted_val) if predicted_val is not None else ''
        return normalize_string(gt_str) == normalize_string(pred_str)


def find_matching_transaction(gt_transaction: Dict, predicted_transactions: List[Dict]) -> Tuple[Optional[Dict], float]:
    # Poisce najboljse ujemanje za GT transakcijo med napovedanimi
    best_match = None
    best_score = 0.0
    
    for pred_tx in predicted_transactions:
        score = 0.0
        
        # Prioriteta 1: Ujemanje reference
        if gt_transaction.get('reference') and pred_tx.get('reference'):
            if str(gt_transaction['reference']) == str(pred_tx['reference']):
                score += 5.0
        
        # Prioriteta 2: Ujemanje datuma
        if gt_transaction.get('date') == pred_tx.get('date'):
            score += 2.0
        
        # Prioriteta 3: Ujemanje zneska
        try:
            gt_change = float(gt_transaction.get('change', 0))
            pred_change = float(pred_tx.get('change', 0))
            if abs(gt_change - pred_change) < 0.01:
                score += 2.0
        except (ValueError, TypeError):
            pass
        
        # Prioriteta 4: Ujemanje stanja
        try:
            gt_balance = float(gt_transaction.get('balance', 0))
            pred_balance = float(pred_tx.get('balance', 0))
            if abs(gt_balance - pred_balance) < 0.01:
                score += 1.0
        except (ValueError, TypeError):
            pass
        
        if score > best_score:
            best_score = score
            best_match = pred_tx
    
    return best_match, best_score


def compare_transactions(ground_truth: Dict, predicted: Dict) -> Dict:
    # Primerja transakcije med ground truth in napovedanimi podatki
    gt_transactions = ground_truth.get('transactions', [])
    pred_transactions = predicted.get('transactions', [])
    
    results = {
        'total_ground_truth': len(gt_transactions),
        'total_predicted': len(pred_transactions),
        'matched_transactions': 0,
        'fully_correct_transactions': 0,
        'unmatched_ground_truth': 0,
        'field_statistics': {field: {'total': 0, 'exact_matches': 0} for field in TRANSACTION_FIELDS},
        'unmatched_gt_indices': []
    }
    
    used_pred_indices = set()
    
    for gt_idx, gt_tx in enumerate(gt_transactions):
        available_pred = [(i, tx) for i, tx in enumerate(pred_transactions) if i not in used_pred_indices]
        
        if not available_pred:
            results['unmatched_ground_truth'] += 1
            results['unmatched_gt_indices'].append(gt_idx)
            continue
        
        best_match = None
        best_idx = None
        best_score = 0.0
        
        for pred_idx, pred_tx in available_pred:
            match, score = find_matching_transaction(gt_tx, [pred_tx])
            if score > best_score:
                best_score = score
                best_match = pred_tx
                best_idx = pred_idx
        
        if best_match is not None and best_score >= 2.0:
            results['matched_transactions'] += 1
            used_pred_indices.add(best_idx)
            
            all_fields_correct = True
            for field in TRANSACTION_FIELDS:
                gt_val = gt_tx.get(field)
                pred_val = best_match.get(field)
                is_match = compare_values(gt_val, pred_val, field)
                
                results['field_statistics'][field]['total'] += 1
                if is_match:
                    results['field_statistics'][field]['exact_matches'] += 1
                else:
                    all_fields_correct = False
            
            if all_fields_correct:
                results['fully_correct_transactions'] += 1
        else:
            results['unmatched_ground_truth'] += 1
            results['unmatched_gt_indices'].append(gt_idx)
    
    results['extra_predicted'] = len(pred_transactions) - len(used_pred_indices)
    return results


def calculate_statistics(comparison_results: Dict) -> Dict:
    # Izracuna koncno statistiko iz rezultatov primerjave
    stats = {
        'general': {
            'total_ground_truth_transactions': comparison_results['total_ground_truth'],
            'total_detected_transactions': comparison_results['total_predicted'],
            'fully_correct_transactions': comparison_results['fully_correct_transactions'],
            'transaction_recall': 0.0,
            'transaction_precision': 0.0
        },
        'per_field': {},
        'overall_accuracy': 0.0
    }
    
    if comparison_results['total_ground_truth'] > 0:
        stats['general']['transaction_recall'] = comparison_results['fully_correct_transactions'] / comparison_results['total_ground_truth']
    
    if comparison_results['total_predicted'] > 0:
        stats['general']['transaction_precision'] = comparison_results['fully_correct_transactions'] / comparison_results['total_predicted']
    
    recall = stats['general']['transaction_recall']
    precision = stats['general']['transaction_precision']
    
    total_fields = 0
    total_exact_matches = 0
    
    for field, field_stats in comparison_results['field_statistics'].items():
        if field_stats['total'] > 0:
            accuracy = field_stats['exact_matches'] / field_stats['total']
            stats['per_field'][field] = {
                'total_comparisons': field_stats['total'],
                'exact_matches': field_stats['exact_matches'],
                'accuracy': round(accuracy * 100, 2)
            }
            total_fields += field_stats['total']
            total_exact_matches += field_stats['exact_matches']
    
    if total_fields > 0:
        stats['overall_accuracy'] = round((total_exact_matches / total_fields) * 100, 2)
    
    return stats


def print_statistics(stats: Dict, comparison_results: Dict = None) -> None:
    # Izpise statistiko v konzolo
    print("\n" + "=" * 60)
    print("              REZULTATI TESTIRANJA OCR/LLM")
    print("=" * 60)
    
    gen = stats['general']
    print("\n[SPLOSNA STATISTIKA]")
    print(f"  GT transakcij:    {gen['total_ground_truth_transactions']}")
    print(f"  Zaznanih:         {gen['total_detected_transactions']}")
    print(f"  Polno pravilnih:  {gen['fully_correct_transactions']}")
    print(f"  Recall:           {gen['transaction_recall']:.2%}")
    print(f"  Precision:        {gen['transaction_precision']:.2%}")
    
    print("\n[STATISTIKA PO POLJIH]")
    print(f"  {'Polje':<15} {'Primerjav':<12} {'Tocnih':<10} {'Natancnost':<12}")
    print("  " + "-" * 47)
    
    for field, fs in stats['per_field'].items():
        print(f"  {field:<15} {fs['total_comparisons']:<12} {fs['exact_matches']:<10} {fs['accuracy']:>6.1f}%")
    
    print("  " + "-" * 47)
    print(f"  {'SKUPAJ':<15} {'':<12} {'':<10} {stats['overall_accuracy']:>6.1f}%")
    print("=" * 60 + "\n")


def run_comparison_test(statement_number: str, predicted_data: Any) -> Dict:
    # Izvede popoln test primerjave med ground truth in napovedanimi podatki
    ground_truth = load_ground_truth(statement_number)
    if ground_truth is None:
        return {'success': False, 'error': f'GT datoteka ni najdena: data/data/{statement_number}.json', 'statistics': None}
    
    if isinstance(predicted_data, str):
        try:
            json_match = re.search(r'\{[\s\S]*\}', predicted_data)
            predicted_data = json.loads(json_match.group()) if json_match else json.loads(predicted_data)
        except json.JSONDecodeError as e:
            print(f"[DEBUG] LLM output (prvih 500 znakov):\n{predicted_data[:500] if predicted_data else 'None'}")
            print(f"[DEBUG] Napaka: {e}")
            return {'success': False, 'error': f'Napaka pri parsanju: {e}', 'statistics': None}
        except AttributeError:
            print(f"[DEBUG] JSON ni bil najden v outputu. LLM output:\n{predicted_data[:500] if predicted_data else 'None'}")
            return {'success': False, 'error': 'JSON ni bil najden v LLM outputu', 'statistics': None}
    
    if not isinstance(predicted_data, dict):
        return {'success': False, 'error': f'Napacen format (pricakovan dict, dobljen {type(predicted_data).__name__})', 'statistics': None}
    
    comparison_results = compare_transactions(ground_truth, predicted_data)
    statistics = calculate_statistics(comparison_results)
    print_statistics(statistics, comparison_results)
    
    return {'success': True, 'error': None, 'statistics': statistics}


def get_statistics_summary(test_result: Dict) -> Dict:
    # Vrne povzetek statistike za API response
    if not test_result['success']:
        return {'success': False, 'error': test_result['error']}
    
    stats = test_result['statistics']
    return {
        'success': True,
        'general': {
            'total_ground_truth': stats['general']['total_ground_truth_transactions'],
            'total_detected': stats['general']['total_detected_transactions'],
            'fully_correct': stats['general']['fully_correct_transactions'],
            'recall': round(stats['general']['transaction_recall'] * 100, 2),
            'precision': round(stats['general']['transaction_precision'] * 100, 2)
        },
        'per_field': stats['per_field'],
        'overall_accuracy': stats['overall_accuracy']
    }
