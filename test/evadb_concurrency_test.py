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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Starting execution")
    try:
        # 导入模块
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Importing module {module_path}")
        module = importlib.import_module(module_path)
        
        # 存储原始TEXT_COUNT_LIST
        original_text_count_list = getattr(module, "TEXT_COUNT_LIST", None)
        # 修改TEXT_COUNT_LIST为只包含1000
        if original_text_count_list:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Modifying TEXT_COUNT_LIST to [1000]")
            setattr(module, "TEXT_COUNT_LIST", [1000])
        
        # 记录开始时间
        start_time = time.time()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Executing test function")
        
        # 执行测试函数
        test_function()
        
        # 恢复原始TEXT_COUNT_LIST
        if original_text_count_list:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Restoring original TEXT_COUNT_LIST")
            setattr(module, "TEXT_COUNT_LIST", original_text_count_list)
        
        end_time = time.time()
        latency = end_time - start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Execution completed successfully in {latency:.2f} seconds")
        
        return {
            "task_id": task_id,
            "status": "success",
            "latency": latency
        }
    except Exception as e:
        error_msg = str(e)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Error occurred: {error_msg}")
        # 尝试恢复原始TEXT_COUNT_LIST
        try:
            module = importlib.import_module(module_path)
            original_text_count_list = getattr(module, "TEXT_COUNT_LIST", None)
            if original_text_count_list:
                setattr(module, "TEXT_COUNT_LIST", original_text_count_list)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Restored original TEXT_COUNT_LIST after error")
        except:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {task_id}: Failed to restore original TEXT_COUNT_LIST")
        return {"task_id": task_id, "status": "failed", "error": error_msg}

def test_concurrency(concurrency_level, test_function, module_path):
    """
    Test a specific concurrency level for the given test function
    """
    print(f"Testing concurrency level: {concurrency_level}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting concurrency test with {concurrency_level} workers")
    
    # Use ProcessPoolExecutor to run tasks concurrently
    with ProcessPoolExecutor(max_workers=concurrency_level) as executor:
        # Submit tasks
        futures = []
        for i in range(concurrency_level):
            future = executor.submit(run_single_task_worker, i, test_function, module_path)
            futures.append(future)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Submitted task {i}")
        
        # Collect results
        results = []
        error_occurred = False
        
        for i, future in enumerate(futures):
            try:
                if error_occurred:
                    # If error occurred in previous task, cancel this task
                    future.cancel()
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cancelled task {i} due to previous error")
                    results.append({"task_id": i, "status": "cancelled", "error": "Cancelled due to previous task failure"})
                    continue
                
                result = future.result(timeout=600)  # 10 minute timeout
                results.append(result)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {i} completed successfully")
            except Exception as e:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Timeout or error in task {i}: {str(e)}")
                results.append({"task_id": i, "status": "failed", "error": f"Timeout or error: {str(e)}"})
                # Set error flag to cancel remaining tasks
                error_occurred = True
                # Cancel all remaining tasks
                for j in range(i + 1, len(futures)):
                    futures[j].cancel()
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cancelled task {j} due to error in task {i}")
                    results.append({"task_id": j, "status": "cancelled", "error": f"Cancelled due to error in task {i}"})
                break
    
    # Count success and failure rates
    successful_tasks = [r for r in results if r["status"] == "success"]
    failed_tasks = [r for r in results if r["status"] == "failed"]
    cancelled_tasks = [r for r in results if r["status"] == "cancelled"]
    
    success_rate = len(successful_tasks) / len(results) if len(results) > 0 else 0
    failure_rate = len(failed_tasks) / len(results) if len(results) > 0 else 0
    cancelled_rate = len(cancelled_tasks) / len(results) if len(results) > 0 else 0
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Concurrency level {concurrency_level}: {len(successful_tasks)}/{len(results)} succeeded, {len(failed_tasks)}/{len(results)} failed, {len(cancelled_tasks)}/{len(results)} cancelled")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Success rate: {success_rate:.2%}, Failure rate: {failure_rate:.2%}, Cancelled rate: {cancelled_rate:.2%}")
    
    test_result = {
        "concurrency_level": concurrency_level,
        "total_tasks": len(results),
        "successful_tasks": len(successful_tasks),
        "failed_tasks": len(failed_tasks),
        "cancelled_tasks": len(cancelled_tasks),
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "cancelled_rate": cancelled_rate,
        "results": results
    }
    
    return test_result

def find_exact_max_concurrency(start_level, end_level, test_function, module_path):
    """
    使用二分法精确找到最大安全并发数
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting binary search in range [{start_level}, {end_level}]")
    
    all_results = []
    
    while start_level <= end_level:
        if start_level == end_level:
            # 只有一个值需要测试
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search: Testing single level {start_level}")
            test_result = test_concurrency(start_level, test_function, module_path)
            all_results.append(test_result)
            
            if test_result["failed_tasks"] == 0:
                # 当前并发数成功，这是最大安全值
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search: Level {start_level} succeeded, returning as max safe concurrency")
                return all_results, start_level
            else:
                # 当前并发数失败，需要返回上一个成功的值
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search: Level {start_level} failed, returning {start_level - 1} as max safe concurrency")
                return all_results, start_level - 1  # 假设前一个值是成功的
        
        mid = (start_level + end_level) // 2
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search: Testing mid level {mid} in range [{start_level}, {end_level}]")
        test_result = test_concurrency(mid, test_function, module_path)
        all_results.append(test_result)
        
        if test_result["failed_tasks"] == 0:
            # 当前并发数成功，尝试更高
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search: Level {mid} succeeded, moving to range [{mid + 1}, {end_level}]")
            start_level = mid + 1
        else:
            # 当前并发数失败，尝试更低
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search: Level {mid} failed, moving to range [{start_level}, {mid - 1}]")
            end_level = mid - 1
        
        # 小延迟以允许系统恢复
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search: Pausing for 1 second to allow system recovery")
        time.sleep(1)
    
    # 循环结束后，start_level > end_level
    # 此时end_level应该是最大的成功值
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search: Loop ended, returning {end_level} as max safe concurrency")
    return all_results, end_level

def find_max_concurrency_optimized(test_function, module_path, start_level=1, max_level=1000, coarse_step=32):
    """
    优化的并发查找函数，结合粗粒度搜索和二分查找
    """
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting optimized concurrency test: start={start_level}, max={max_level}, coarse_step={coarse_step}")
    
    all_results = []
    
    # Step 1: Coarse-grained search to find the approximate boundary
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 1: Coarse search with step size {coarse_step}")
    current_level = start_level
    last_success_level = start_level - 1
    
    while current_level <= max_level:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Coarse search: Testing level {current_level}")
        test_result = test_concurrency(current_level, test_function, module_path)
        all_results.append(test_result)
        
        if test_result["failed_tasks"] == 0:
            last_success_level = current_level
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Coarse search: Level {current_level} succeeded, moving to next level {current_level + coarse_step}")
            current_level += coarse_step
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Coarse search: Found failure at level {current_level}, last success was {last_success_level}")
            break
    
    # Determine the exact boundary using binary search in the narrowed range
    if last_success_level >= max_level:
        # All levels up to max_level worked
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All levels up to max_level {max_level} worked, returning {max_level}")
        return all_results, max_level
    elif last_success_level < start_level:
        # Even the start level failed
        result = 0 if start_level == 1 else -1
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Even start level {start_level} failed, returning {result}")
        return all_results, result  # Return -1 to indicate start_level itself failed
    else:
        # Perform binary search between last_success_level and current_level
        # Since last_success_level worked but current_level failed, search in [last_success_level+1, current_level-1]
        if current_level - last_success_level == 1:
            # Adjacent values: last_success_level works, current_level fails
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Adjacent values found: {last_success_level} works, {current_level} fails, returning {last_success_level}")
            return all_results, last_success_level
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 2: Binary search between {last_success_level + 1} and {current_level - 1}")
            binary_results, precise_level = find_exact_max_concurrency(
                last_success_level + 1, 
                current_level - 1, 
                test_function, 
                module_path
            )
            all_results.extend(binary_results)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Binary search completed, returning {precise_level} as max safe concurrency")
            return all_results, precise_level

def run_evadb_concurrency_tests():
    """
    Run concurrency tests for each evadb test case
    """
    all_test_results = {}
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting EVA DB Concurrency Stress Tests")
    print("="*80)
    
    for test_case in test_cases:
        test_name = test_case["name"]
        test_function = test_case["test_function"]
        module_path = test_case["module_path"]
        
        if test_function is None:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipping {test_name} - Test function not available")
            continue
        
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing EVA DB: {test_name}")
        print(f"{'='*60}")
        
        # First run the original test to ensure it works
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running original test to verify functionality...")
        try:
            # 导入模块
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Importing module {module_path}")
            module = importlib.import_module(module_path)
            # 存储原始TEXT_COUNT_LIST
            original_text_count_list = getattr(module, "TEXT_COUNT_LIST", None)
            # 修改TEXT_COUNT_LIST为只包含1000
            if original_text_count_list:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Modifying TEXT_COUNT_LIST to [1000]")
                setattr(module, "TEXT_COUNT_LIST", [1000])
            
            # 运行测试函数
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Executing test function")
            test_function()
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Original test passed")
            
            # 恢复原始TEXT_COUNT_LIST
            if original_text_count_list:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Restoring original TEXT_COUNT_LIST")
                setattr(module, "TEXT_COUNT_LIST", original_text_count_list)
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Original test failed: {str(e)}")
            continue
        
        # Run concurrency test
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running concurrency test...")
        all_results, max_safe_concurrency = find_max_concurrency_optimized(
            test_function=test_function,
            module_path=module_path,
            start_level=1,
            max_level=300,
            coarse_step=32
        )
        
        # Store results
        all_test_results[test_name] = {
            "name": test_name,
            "module_path": module_path,
            "max_safe_concurrency": max_safe_concurrency,
            "results": all_results
        }
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Max safe concurrency for {test_name}: {max_safe_concurrency}")
    
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Writing results to JSON file: {filename}")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Results saved to: {filename}")
    
    # Also create CSV version
    csv_filename = filename.replace('.json', '.csv')
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating CSV version: {csv_filename}")
    json_to_csv(filename, csv_filename)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] All tests completed")
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
