import os
import re
import sys
from bs4 import BeautifulSoup
from collections import defaultdict

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define a function to process a results directory
def process_results_dir(results_dir):
    local_bug_data = {}
    
    for run_dir in os.listdir(results_dir):
        run_path = os.path.join(results_dir, run_dir)
        if not os.path.isdir(run_path):
            continue
        
        report_path = os.path.join(run_path, "summary_report.html")
        if not os.path.exists(report_path):
            continue
        
        with open(report_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        tables = soup.find_all("table")
        if not tables:
            continue
        
        bug_table = tables[0]
        rows = bug_table.find_all("tr")
        
        for row in rows[1:]:
            cells = row.find_all("td")
            if not cells or len(cells) < 8:
                continue
            
            bug_name = cells[0].text.strip()
            solution_status = cells[1].text.strip()
            patch_status = cells[2].text.strip()
            evaluation_status = cells[3].text.strip()
            correct_count = int(cells[5].text.strip()) if cells[5].text.strip() else 0
            plausible_count = int(cells[6].text.strip()) if cells[6].text.strip() else 0
            incorrect_count = int(cells[7].text.strip()) if cells[7].text.strip() else 0
            
            local_bug_data[bug_name] = {
                "solution_status": solution_status,
                "patch_status": patch_status,
                "evaluation_status": evaluation_status,
                "correct": correct_count,
                "plausible": plausible_count,
                "incorrect": incorrect_count,
                "total_patches": correct_count + plausible_count + incorrect_count
            }
    
    return local_bug_data

# Main function to handle multiple folders
def main():
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Use directories provided as command line arguments
        results_dirs = sys.argv[1:]
    else:
        # Default: use the standard ../results directory
        results_dirs = [os.path.join(script_dir, "../results")]
    
    print(f"Processing {len(results_dirs)} directories:")
    for dir_path in results_dirs:
        print(f" - {dir_path}")
    
    # Initialize combined bug data
    bug_data = {}
    
    # Process each directory and merge results
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            print(f"Warning: Directory not found: {results_dir}")
            continue
        
        local_bug_data = process_results_dir(results_dir)
        print(f"Found data for {len(local_bug_data)} bugs in {results_dir}")
        
        # Merge the local bug data into the combined bug data
        for bug_name, data in local_bug_data.items():
            # If bug exists in multiple directories, take the one with the best results
            if bug_name in bug_data:
                existing = bug_data[bug_name]
                # Prioritize entries with successful patches
                if (data["patch_status"] == "Success" and existing["patch_status"] != "Success") or \
                   (data["evaluation_status"] == "Success" and existing["evaluation_status"] != "Success") or \
                   (data["correct"] > existing["correct"]) or \
                   (data["correct"] == existing["correct"] and data["plausible"] > existing["plausible"]):
                    bug_data[bug_name] = data
            else:
                bug_data[bug_name] = data
    
    # Calculate final statuses
    final_statuses = {}
    for bug_name, data in bug_data.items():
        if data["patch_status"] != "Success":
            final_statuses[bug_name] = "No Patch"
        elif data["evaluation_status"] != "Success":
            final_statuses[bug_name] = "No Evaluation"
        elif data["correct"] > 0:
            final_statuses[bug_name] = "CORRECT"
        elif data["plausible"] > 0:
            final_statuses[bug_name] = "PLAUSIBLE"
        elif data["incorrect"] > 0 and data["incorrect"] == data["total_patches"]:
            final_statuses[bug_name] = "INCORRECT"
        else:
            final_statuses[bug_name] = "UNKNOWN"

    status_counts = defaultdict(int)
    for status in final_statuses.values():
        status_counts[status] += 1

    total_bugs = len(bug_data)
    total_with_patch = total_bugs - status_counts["No Patch"]
    total_with_evaluation = total_with_patch - status_counts["No Evaluation"]

    # Generate report
    report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Repair-of-Thought Results Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .metric {{ background-color: white; border: 1px solid #ddd; padding: 10px; border-radius: 5px; flex: 1; min-width: 150px; }}
            .success {{ color: green; }}
            .warning {{ color: orange; }}
            .error {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Repair-of-Thought Results Analysis</h1>
        
        <div class="summary">
            <h2>Overall Summary</h2>
            <p><strong>Total Bugs Analyzed:</strong> {total_bugs}</p>
            <p><strong>Bugs with Patches:</strong> {total_with_patch} ({total_with_patch/total_bugs*100:.1f}% if total_bugs > 0 else 0)</p>
            <p><strong>Bugs with Evaluations:</strong> {total_with_evaluation} ({total_with_evaluation/total_bugs*100:.1f}% if total_bugs > 0 else 0)</p>
        </div>
        
        <h2>Patch Quality Distribution</h2>
        <div class="metrics">
            <div class="metric">
                <h3 class="success">CORRECT</h3>
                <p>Bugs with at least one correct patch: {status_counts['CORRECT']}</p>
                <p>({status_counts['CORRECT']/total_with_evaluation*100:.1f}% of evaluated if total_with_evaluation > 0 else 0)</p>
            </div>
            
            <div class="metric">
                <h3 class="warning">PLAUSIBLE</h3>
                <p>Bugs with at least one plausible patch but no correct patches: {status_counts['PLAUSIBLE']}</p>
                <p>({status_counts['PLAUSIBLE']/total_with_evaluation*100:.1f}% of evaluated if total_with_evaluation > 0 else 0)</p>
            </div>
            
            <div class="metric">
                <h3 class="error">INCORRECT</h3>
                <p>Bugs with only incorrect patches: {status_counts['INCORRECT']}</p>
                <p>({status_counts['INCORRECT']/total_with_evaluation*100:.1f}% of evaluated if total_with_evaluation > 0 else 0)</p>
            </div>
        </div>
        
        <div class="summary">
            <h2>Detailed Metrics</h2>
            <p><strong>Bugs without Patches:</strong> {status_counts['No Patch']}</p>
            <p><strong>Bugs with Patches but No Evaluation:</strong> {status_counts['No Evaluation']}</p>
            <p><strong>Bugs with Unknown Status:</strong> {status_counts.get('UNKNOWN', 0)}</p>
        </div>
    """


    # Add bug details table
    bug_table = "<h2>Bug Details</h2><table>"
    bug_table += "<tr><th>Bug</th><th>Solution Status</th><th>Patch Status</th><th>Evaluation Status</th><th>CORRECT Patches</th><th>PLAUSIBLE Patches</th><th>INCORRECT Patches</th><th>Final Status</th></tr>"

    for bug_name, data in sorted(bug_data.items()):
        bug_table += f"<tr>"
        bug_table += f"<td>{bug_name}</td>"
        bug_table += f"<td>{data['solution_status']}</td>"
        bug_table += f"<td>{data['patch_status']}</td>"
        bug_table += f"<td>{data['evaluation_status']}</td>"
        bug_table += f"<td>{data['correct']}</td>"
        bug_table += f"<td>{data['plausible']}</td>"
        bug_table += f"<td>{data['incorrect']}</td>"
        bug_table += f"<td>{final_statuses[bug_name]}</td>"
        bug_table += f"</tr>"

    bug_table += "</table>"

    report += bug_table + "</body></html>"

    # Write the report to a file
    with open("repair_of_thought_analysis.html", "w", encoding="utf-8") as f:
        f.write(report)

    print("Analysis complete. Report saved to repair_of_thought_analysis.html")

if __name__ == "__main__":
    main()