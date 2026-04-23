"""
Test suite for CRIMSON Score - Ranking/Intuition Tests
Focus: Verify relative ranking (C1 > C2) rather than absolute scores
"""

import argparse
import json
from datetime import datetime
from CRIMSON.generate_score import CRIMSONScore
from RadJudge.test_cases import RANKING_TESTS

try:
    from RadEval.RadEval import RadEval
    RADEVAL_AVAILABLE = True
except Exception as e:
    print(f"\n⚠ RadEval import failed: {e}")
    RADEVAL_AVAILABLE = False

# Metric direction: True = higher is better, False = lower is better (error rate)
METRIC_DIRECTIONS = {
    "crimson": True,  # Higher is better
    "radgraph_complete": True,  # Higher is better
    "bleu": True,  # Higher is better
    "bertscore": True,  # Higher is better
    "green": True,  # Higher is better
    "chexbert": True,  # Higher is better
    "ratescore": True,  # Higher is better
    "radcliq": False,  # Lower perplexity/distance is better
    "rouge": True,  # Higher is better
    "rougeL": True,  # Higher is better
}

def flatten_tests(ranking_tests):
    """Flatten grouped test categories into a flat list of test cases with metadata."""
    flat = []
    for cat_idx, category in enumerate(ranking_tests, 1):
        cat_name = category["category"]
        cat_desc = category.get("description", "")
        for test in category["tests"]:
            flat.append({
                "id": test["id"],
                "category": cat_name,
                "category_description": cat_desc,
                "title": f"[{cat_name}] {test['id']}",
                "ground_truth": test["ground_truth"],
                "candidates": test["candidates"],
                "expected_ranking": test["expected_ranking"],
            })
    return flat


def run_ranking_test(test_case, scorer, radeval_scorer=None, test_num=None, total_tests=None):
    """
    Run a single ranking test case.
    
    Args:
        test_case: Dictionary with test configuration
        scorer: CRIMSONScore instance
        radeval_scorer: Optional RadEval instance for comparison metrics
        test_num: Current test number (for progress display)
        total_tests: Total number of tests (for progress display)
        
    Returns:
        dict: Test results with pass/fail status and all metric scores
    """
    progress = f" [{test_num}/{total_tests}]" if test_num and total_tests else ""
    print(f"\nTEST {test_case['id']}{progress}: {test_case['title']}")
    
    gt = test_case['ground_truth']
    candidates = test_case['candidates']
    
    # Evaluate all candidates with CRIMSON
    crimson_results = {}
    for cand_id, cand_info in candidates.items():
        context = cand_info.get('context', None)
        result = scorer.evaluate(gt, cand_info['text'], patient_context=context)
        crimson_results[cand_id] = result
    
    radeval_results = {}
    if radeval_scorer is not None:
        for cand_id, cand_info in candidates.items():
            refs = [gt]
            hyps = [cand_info['text']]
            radeval_scores = radeval_scorer(refs=refs, hyps=hyps)
            
            flat_scores = {}
            for key, value in radeval_scores.items():
                if key.startswith('rouge') and key != 'rougeL':
                    continue
                if key.startswith('chexbert') and key != 'chexbert-5_micro avg_f1-score':
                    continue
                if key.startswith('radgraph') and key not in ['radgraph_complete']:
                    continue
                
                if isinstance(value, dict):
                    if 'bleu' in key.lower():
                        flat_scores[key] = value.get('bleu-4', list(value.values())[0])
                    elif 'rouge' in key.lower():
                        flat_scores['rougeL'] = value.get('rougeL', value.get('rouge-l', list(value.values())[0]))
                    elif 'chexbert' in key.lower():
                        flat_scores['chexbert'] = value.get('chexbert-5_micro avg_f1-score', list(value.values())[0])
                    else:
                        flat_scores[key] = list(value.values())[0]
                else:
                    flat_scores[key] = value
            
            radeval_results[cand_id] = flat_scores
    
    crimson_scores = {}
    for cand_id, result in crimson_results.items():
        score = result['crimson_score']
        crimson_scores[cand_id] = score
    
    # Check CRIMSON ranking
    expected = test_case['expected_ranking']
    
    crimson_passed = check_ranking(expected, crimson_scores, metric_name="crimson")
    
    radeval_passed = {}
    if radeval_results:
        first_result = next(iter(radeval_results.values()))
        metric_names = list(first_result.keys())
        
        for metric_name in metric_names:
            metric_scores = {cand_id: radeval_results[cand_id][metric_name] 
                           for cand_id in radeval_results.keys()}
            passed = check_ranking(expected, metric_scores, metric_name=metric_name)
            radeval_passed[metric_name] = passed
    
    return {
        "test_id": test_case['id'],
        "category": test_case.get('category', ''),
        "test_title": test_case['title'],
        "expected_ranking": expected,
        "crimson": {
            "scores": {k: float(v) for k, v in crimson_scores.items()},
            "passed": bool(crimson_passed)
        },
        "radeval": {k: bool(v) for k, v in radeval_passed.items()} if radeval_results else None
    }


def check_ranking(expected, scores, metric_name="Score"):
    """
    Check if scores match expected ranking.
    
    Args:
        expected: Expected ranking string (e.g., "C2 > C1" or "C1 = C2")
                 OR list of rankings that ALL must pass (e.g., ["C3 > C2", "C2 > C1"])
                 Note: "C2 > C1" means C2 has HIGHER SCORE (better performance)
        scores: Dictionary of candidate scores
        metric_name: Name of the metric being checked
        
    Returns:
        bool: True if ranking matches expectation, False otherwise
    """
    if isinstance(expected, list):
        all_passed = True
        results = []
        
        for exp in expected:
            passed = check_single_ranking(exp, scores, metric_name, print_result=False)
            results.append((exp, passed))
            if not passed:
                all_passed = False
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        score_str = ", ".join([f"{cid}={score:.3f}" for cid, score in sorted_scores])
        
        # Print each comparison result with progress counter
        total = len(results)
        for idx, (exp, passed) in enumerate(results, 1):
            status = "✓" if passed else "✗"
            print(f"  {status} {metric_name} [{idx}/{total}]: {exp} ({score_str})")
        
        return all_passed
    else:
        return check_single_ranking(expected, scores, metric_name, print_result=True)


def check_single_ranking(expected, scores, metric_name="Score", print_result=True):
    """
    Check a single ranking expectation.
    
    Args:
        expected: Expected ranking string (e.g., "C2 > C1" or "C1 = C2")
        scores: Dictionary of candidate scores
        metric_name: Name of the metric being checked
        print_result: Whether to print the result
        
    Returns:
        bool: True if ranking matches expectation, False otherwise
    """
    higher_is_better = METRIC_DIRECTIONS.get(metric_name, True)
    
    if ' = ' in expected:
        parts = expected.split(' = ')
        id1, id2 = parts[0].strip(), parts[1].strip()
        score1 = scores[id1]
        score2 = scores[id2]
        
        passed = abs(score1 - score2) < 0.01
        
        if print_result:
            status = "✓" if passed else "✗"
            print(f"  {status} {metric_name}: {id1}={score1:.3f} ≈ {id2}={score2:.3f}")
        
        return passed
    elif ' > ' in expected:
        parts = expected.split(' > ')
        better_id, worse_id = parts[0].strip(), parts[1].strip()
        better_score = scores[better_id]
        worse_score = scores[worse_id]
        
        if higher_is_better:
            passed = better_score > worse_score
        else:
            passed = better_score < worse_score
        
        if print_result:
            status = "✓" if passed else "✗"
            print(f"  {status} {metric_name}: {better_id}={better_score:.3f} > {worse_id}={worse_score:.3f}")
        
        return passed
    
    return False


def main(cache_dir=None, test_ids=None, use_radeval=False):
    """Run all ranking tests.
    
    Args:
        cache_dir: Cache directory to pass to RadEval for model caching
        test_ids: List of test IDs to run (e.g., ['test_01', 'test_03']). If None, runs all tests.
        use_radeval: If True, also evaluate with RadEval metrics. Default False.
    """
    print("\n" + "=" * 80)
    print("CRIMSON Score Test Suite - Ranking Tests")
    print("=" * 80)
    
    scorer = CRIMSONScore(api="openai")
    
    # Initialize RadEval if requested and available
    radeval_scorer = None
    if use_radeval and RADEVAL_AVAILABLE:
        evaluator_kwargs = {
            'do_radgraph': True,
            'do_bleu': True,
            'do_bertscore': True,
            'do_green': True,
            'do_chexbert': True,
            'do_ratescore': True,
            'do_radcliq': True,
            'do_rouge': True,
        }
        if cache_dir:
            evaluator_kwargs['cache_dir'] = cache_dir
        
        try:
            radeval_scorer = RadEval(**evaluator_kwargs)
        except Exception as e:
            print(f"\n⚠ RadEval initialization failed: {e}")
            radeval_scorer = None
    
    all_results = []
    
    # Flatten grouped test structure
    all_tests = flatten_tests(RANKING_TESTS)
    
    # Filter tests if specific test IDs are provided
    tests_to_run = all_tests
    if test_ids:
        # Support exact match ("2a") or category prefix ("2" matches "2a", "2b", etc.)
        # Prefix match only applies when the filter is purely numeric (a category number).
        # e.g., "1" matches "1a","1b","1c" but NOT "10a"; "10" matches "10a" but NOT "1a".
        import re
        normalized_ids = [tid.lower().strip() for tid in test_ids]
        def _matches(test_id, filters):
            tid = test_id.lower()
            for f in filters:
                if tid == f:
                    return True
                # Prefix match: filter must be numeric and test ID must be <filter><letter>
                if f.isdigit() and re.match(rf'^{re.escape(f)}[a-z]$', tid):
                    return True
            return False
        tests_to_run = [t for t in all_tests if _matches(t['id'], normalized_ids)]
        if not tests_to_run:
            print(f"\nNo tests found matching IDs: {test_ids}")
            return
        print(f"\nRunning {len(tests_to_run)} selected test(s): {', '.join([t['id'] for t in tests_to_run])}")
    
    try:
        # Run all test cases
        for idx, test_case in enumerate(tests_to_run, 1):
            result = run_ranking_test(test_case, scorer, radeval_scorer, 
                                     test_num=idx, total_tests=len(tests_to_run))
            all_results.append(result)
        
        # Summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        
        crimson_passed = sum(1 for r in all_results if r['crimson']['passed'])
        total = len(all_results)
        
        print(f"\nCRIMSON: {crimson_passed}/{total} tests passed")
        
        # Group results by category for display
        from collections import OrderedDict
        categories = OrderedDict()
        for result in all_results:
            cat = result.get('category', 'Uncategorized')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        for cat_name, cat_results in categories.items():
            cat_passed = sum(1 for r in cat_results if r['crimson']['passed'])
            cat_total = len(cat_results)
            cat_status = "PASS" if cat_passed == cat_total else "FAIL"
            print(f"\n  [{cat_status}] {cat_name} ({cat_passed}/{cat_total})")
            for result in cat_results:
                status = "\u2713" if result['crimson']['passed'] else "\u2717"
                print(f"    {status} {result['test_id']}")
        
        # RadEval results
        if radeval_scorer is not None and all_results[0]['radeval']:
            print("\nRadEval Metrics:")
            metric_names = list(all_results[0]['radeval'].keys())
            for metric_name in metric_names:
                metric_passed = sum(1 for r in all_results if r['radeval'] and r['radeval'].get(metric_name, False))
                print(f"  {metric_name}: {metric_passed}/{total} tests passed")
        
        print("\n" + "=" * 80)
        
        # Save results to JSON
        output_file = f"data/test_suite_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": all_results,
                "summary": {
                    "total_tests": total,
                    "crimson_passed": crimson_passed,
                    "radeval_metrics": {
                        metric_name: sum(1 for r in all_results if r['radeval'] and r['radeval'].get(metric_name, False))
                        for metric_name in (metric_names if radeval_scorer and all_results[0]['radeval'] else [])
                    } if radeval_scorer else None
                }
            }, f, indent=2)
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"\n\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CRIMSON test suite with optional RadEval metrics')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for RadEval model downloads')
    parser.add_argument('--tests', nargs='+', type=str, default=None,
                       help='Specific test IDs to run (e.g., --tests test_01 test_03). If not provided, runs all tests.')
    parser.add_argument('--radeval', action='store_true',
                       help='Also evaluate with RadEval metrics (disabled by default)')
    args = parser.parse_args()
    
    main(cache_dir=args.cache_dir, test_ids=args.tests, use_radeval=args.radeval)