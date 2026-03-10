import time
from datetime import datetime
import json
import csv
from concurrent.futures import ProcessPoolExecutor
import importlib

# Import test functions from evadb_test
try:
    from morphingdb_test.series_test.slice_test.evadb_test import evadb_slioe_test
except ImportError:
    evadb_slioe_test = None

try:
    from morphingdb_test.series_test.year_predict_test.evadb_test import evadb_year_predict_test
except ImportError:
    evadb_year_predict_test = None

try:
    from morphingdb_test.image_test.cifar10.evadb_test import evadb_cifar_test
except ImportError:
    evadb_cifar_test = None

try:
    from morphingdb_test.image_test.imagenet.evadb_test import evadb_imagenet_test
except ImportError:
    evadb_imagenet_test = None

try:
    from morphingdb_test.image_test.stanford_dogs.evadb_test import evadb_stanford_dogs_test
except ImportError:
    evadb_stanford_dogs_test = None

try:
    from morphingdb_test.text_test.financial_phrasebank.evadb_test import evadb_financial_phrasebank_test
except ImportError:
    evadb_financial_phrasebank_test = None

try:
    from morphingdb_test.text_test.imdb.evadb_test import evadb_imdb_test
except ImportError:
    evadb_imdb_test = None

try:
    from morphingdb_test.text_test.sst2.evadb_test import evadb_sst2_test
except ImportError:
    evadb_sst2_test = None

try:
    from morphingdb_test.muti_query.evadb_test import evadb_muti_query_test
except ImportError:
    evadb_muti_query_test = None

# Define test cases
test_cases = [
    {
        "name": "slice_test",
        "test_function": evadb_slioe_test,
        "module_path": "morphingdb_test.series_test.slice_test.evadb_test"
    },
    {
        "name": "year_predict_test",
        "test_function": evadb_year_predict_test,
        "module_path": "morphingdb_test.series_test.year_predict_test.evadb_test"
    },
    {
        "name": "cifar_test",
        "test_function": evadb_cifar_test,
        "module_path": "morphingdb_test.image_test.cifar10.evadb_test"
    },
    {
        "name": "imagenet_test",
        "test_function": evadb_imagenet_test,
        "module_path": "morphingdb_test.image_test.imagenet.evadb_test"
    },
    {
        "name": "stanford_dogs_test",
        "test_function": evadb_stanford_dogs_test,
        "module_path": "morphingdb_test.image_test.stanford_dogs.evadb_test"
    },
    {
        "name": "financial_phrasebank_test",
        "test_function": evadb_financial_phrasebank_test,
        "module_path": "morphingdb_test.text_test.financial_phrasebank.evadb_test"
    },
    {
        "name": "imdb_test",
        "test_function": evadb_imdb_test,
        "module_path": "morphingdb_test.text_test.imdb.evadb_test"
    },
    {
        "name": "sst2_test",
        "test_function": evadb_sst2_test,
        "module_path": "morphingdb_test.text_test.sst2.evadb_test"
    },
    {
        "name": "muti_query_test",
        "test_function": evadb_muti_query_test,
        "module_path": "morphingdb_test.muti_query.evadb_test"
    }
]

def run_single_task_worker(task_id, test_function, module_path):
    """
    单个进程执行的任务单元，运行指定的test_function
    """
    try:
        # 导入模块
        module = importlib.import_module(module_path)
        # 存储原始TEXT_COUNT_LIST
        original_text_count_list = getattr(module, "TEXT_COUNT_LIST", None)
        # 修改TEXT_COUNT_LIST为只包含1000
        if original_text_count_list:
            setattr(module, "TEXT_COUNT_LIST", [1000])
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行测试函数
        test_function()
        
        # 恢复原始TEXT_COUNT_LIST
        if original_text_count_list:
            setattr(module, "TEXT_COUNT_LIST", original_text_count_list)
        
        end_time = time.time()
        
        return {
            "task_id": task_id,
            "status": "success",
            "latency": end_time - start_time
        }
    except Exception as e:
        print(f"Error in task {task_id}: {str(e)}")
        # 尝试恢复原始TEXT_COUNT_LIST
        try:
            module = importlib.import_module(module_path)
            original_text_count_list = getattr(module, "TEXT_COUNT_LIST", None)
            if original_text_count_list:
                setattr(module, "TEXT_COUNT_LIST", original_text_count_list)
        except:
            pass
        return {"task_id": task_id, "status": "failed", "error": str(e)}

def test_concurrency(concurrency_level, test_function, module_path):
    """
    Test a specific concurrency level for the given test function
    """
    print(f"Testing concurrency level: {concurrency_level}")
    
    # Use ProcessPoolExecutor to run tasks concurrently
    with ProcessPoolExecutor(max_workers=concurrency_level) as executor:
        # Submit tasks
        futures = []
        for i in range(concurrency_level):
            future = executor.submit(run_single_task_worker, i, test_function, module_path)
            futures.append(future)
        
        # Collect results
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=600)  # 10 minute timeout
                results.append(result)
            except Exception as e:
                print(f"Timeout or error in task {i}: {str(e)}")
                results.append({"task_id": i, "status": "failed", "error": f"Timeout or error: {str(e)}"})
    
    # Count success and failure rates
    successful_tasks = [r for r in results if r["status"] == "success"]
    failed_tasks = [r for r in results if r["status"] == "failed"]
    
    success_rate = len(successful_tasks) / len(results) if len(results) > 0 else 0
    failure_rate = len(failed_tasks) / len(results) if len(results) > 0 else 0
    
    print(f"Concurrency level {concurrency_level}: {len(successful_tasks)}/{len(results)} succeeded, Success rate: {success_rate:.2%}")
    
    test_result = {
        "concurrency_level": concurrency_level,
        "total_tasks": len(results),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(failed_tasks),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "results": results
    }
    
    return test_result

def find_exact_max_concurrency(start_level, end_level, test_function, module_path):
    """
    使用二分法精确找到最大安全并发数
    """
    print(f"Starting binary search in range [{start_level}, {end_level}]")
    
    all_results = []
    
    while start_level <= end_level:
        if start_level == end_level:
            # 只有一个值需要测试
            test_result = test_concurrency(start_level, test_function, module_path)
            all_results.append(test_result)
            
            if test_result["failed_tasks"] == 0:
                # 当前并发数成功，这是最大安全值
                return all_results, start_level
            else:
                # 当前并发数失败，需要返回上一个成功的值
                return all_results, start_level - 1  # 假设前一个值是成功的
        
        mid = (start_level + end_level) // 2
        test_result = test_concurrency(mid, test_function, module_path)
        all_results.append(test_result)
        
        if test_result["failed_tasks"] == 0:
            # 当前并发数成功，尝试更高
            start_level = mid + 1
        else:
            # 当前并发数失败，尝试更低
            end_level = mid - 1
        
        # 小延迟以允许系统恢复
        time.sleep(1)
    
    # 循环结束后，start_level > end_level
    # 此时end_level应该是最大的成功值
    return all_results, end_level

def find_max_concurrency_optimized(test_function, module_path, start_level=1, max_level=1000, coarse_step=32):
    """
    优化的并发查找函数，结合粗粒度搜索和二分查找
    """
    print(f"Starting optimized concurrency test: start={start_level}, max={max_level}, coarse_step={coarse_step}")
    
    all_results = []
    
    # Step 1: Coarse-grained search to find the approximate boundary
    print(f"Step 1: Coarse search with step size {coarse_step}")
    current_level = start_level
    last_success_level = start_level - 1
    
    while current_level <= max_level:
        test_result = test_concurrency(current_level, test_function, module_path)
        all_results.append(test_result)
        
        if test_result["failed_tasks"] == 0:
            last_success_level = current_level
            current_level += coarse_step
        else:
            print(f"Coarse search: Found failure at level {current_level}, last success was {last_success_level}")
            break
    
    # Determine the exact boundary using binary search in the narrowed range
    if last_success_level >= max_level:
        # All levels up to max_level worked
        return all_results, max_level
    elif last_success_level < start_level:
        # Even the start level failed
        return all_results, 0 if start_level == 1 else -1  # Return -1 to indicate start_level itself failed
    else:
        # Perform binary search between last_success_level and current_level
        # Since last_success_level worked but current_level failed, search in [last_success_level+1, current_level-1]
        if current_level - last_success_level == 1:
            # Adjacent values: last_success_level works, current_level fails
            return all_results, last_success_level
        else:
            print(f"Step 2: Binary search between {last_success_level + 1} and {current_level - 1}")
            binary_results, precise_level = find_exact_max_concurrency(
                last_success_level + 1, 
                current_level - 1, 
                test_function, 
                module_path
            )
            all_results.extend(binary_results)
            return all_results, precise_level

def run_evadb_concurrency_tests():
    """
    Run concurrency tests for each evadb test case
    """
    all_test_results = {}
    
    print("Starting EVA DB Concurrency Stress Tests")
    print("="*80)
    
    for test_case in test_cases:
        test_name = test_case["name"]
        test_function = test_case["test_function"]
        module_path = test_case["module_path"]
        
        if test_function is None:
            print(f"\nSkipping {test_name} - Test function not available")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing EVA DB: {test_name}")
        print(f"{'='*60}")
        
        # First run the original test to ensure it works
        print(f"Running original test to verify functionality...")
        try:
            # 导入模块
            module = importlib.import_module(module_path)
            # 存储原始TEXT_COUNT_LIST
            original_text_count_list = getattr(module, "TEXT_COUNT_LIST", None)
            # 修改TEXT_COUNT_LIST为只包含1000
            if original_text_count_list:
                setattr(module, "TEXT_COUNT_LIST", [1000])
            
            # 运行测试函数
            test_function()
            print(f"Original test passed")
            
            # 恢复原始TEXT_COUNT_LIST
            if original_text_count_list:
                setattr(module, "TEXT_COUNT_LIST", original_text_count_list)
        except Exception as e:
            print(f"Original test failed: {str(e)}")
            continue
        
        # Run concurrency test
        print(f"Running concurrency test...")
        all_results, max_safe_concurrency = find_max_concurrency_optimized(
            test_function=test_function,
            module_path=module_path,
            start_level=1,
            max_level=1000,
            coarse_step=32
        )
        
        # Store results
        all_test_results[test_name] = {
            "name": test_name,
            "module_path": module_path,
            "max_safe_concurrency": max_safe_concurrency,
            "results": all_results
        }
        
        print(f"Max safe concurrency for {test_name}: {max_safe_concurrency}")
    
    # Store all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results_dict = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "evadb_concurrency_stress_test",
        "test_results": all_test_results
    }
    
    # Print summary
    print("="*80)
    print("EVA DB CONCURRENCY STRESS TEST SUMMARY")
    print("="*80)
    print(f"Timestamp: {all_results_dict['timestamp']}")
    
    # Write results to file
    filename = f"evadb_concurrency_test_results_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {filename}")
    
    # Also create CSV version
    json_to_csv(filename, filename.replace('.json', '.csv'))
    
    return all_test_results

def json_to_csv(json_file_path, csv_file_path):
    """
    Convert JSON test results to CSV format.
    
    Args:
        json_file_path (str): Path to the input JSON file
        csv_file_path (str): Path to the output CSV file
    """
    # Load JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare header for concurrency test results
    header = ['Test Name', 'Module Path', 'Concurrency Level', 'Total Tasks', 'Successful Tasks', 'Failed Tasks', 
              'Success Rate', 'Failure Rate', 'Max Safe Concurrency']
    
    # Prepare rows
    rows = []
    
    # Process test results
    if 'test_results' in data:
        for test_name, test_data in data['test_results'].items():
            if 'results' in test_data:
                for result in test_data['results']:
                    row = [
                        test_name,  # Test Name
                        test_data.get('module_path', 'N/A'),  # Module path
                        result['concurrency_level'],
                        result['total_tasks'],
                        result['successful_tasks'],
                        result['failed_tasks'],
                        f"{result['success_rate']:.4f}",
                        f"{result['failure_rate']:.4f}",
                        test_data.get('max_safe_concurrency', 'N/A')  # Max safe concurrency
                    ]
                    rows.append(row)
    
    # Write to CSV
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"CSV results saved to: {csv_file_path}")

if __name__ == "__main__":
    run_evadb_concurrency_tests()
