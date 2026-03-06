"""
Analysis script to generate report-ready tables and statistics from experiment results.
Run this after experiment_initial_angles.py
"""

import pandas as pd
import numpy as np
import os


def load_and_analyze():
    """Load results and generate report-ready analysis."""
    
    # Load results
    results_file = "results/initial_angle_experiment.csv"
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Please run experiment_initial_angles.py first.")
        return
    
    df = pd.read_csv(results_file)
    
    print("=" * 80)
    print("DETAILED ANALYSIS FOR REPORT")
    print("=" * 80)
    print()
    
    # Table 1: Main Results Table (for report)
    print("TABLE 1: Performance Across Different Initial Angles")
    print("-" * 80)
    
    # Create a clean table for the report
    report_table = df.pivot_table(
        values='Steps Survived',
        index='Initial Angle (deg)',
        columns='Controller',
        aggfunc='first',
        fill_value=0
    )
    
    # Add success indicators
    success_table = df.pivot_table(
        values='Success',
        index='Initial Angle (deg)',
        columns='Controller',
        aggfunc='first'
    )
    
    # Combine into formatted string
    formatted_table = report_table.copy()
    for col in formatted_table.columns:
        formatted_table[col] = formatted_table[col].astype(int)
        # Add checkmark for success (1000 steps)
        mask = formatted_table[col] >= 1000
        formatted_table.loc[mask, col] = formatted_table.loc[mask, col].astype(str) + ' ✓'
    
    print(formatted_table)
    print()
    
    # Save LaTeX table
    latex_output = "results/table_latex.txt"
    with open(latex_output, 'w') as f:
        f.write(report_table.to_latex(float_format="%.0f"))
    print(f"LaTeX table saved to: {latex_output}")
    print()
    
    # Table 2: Success/Failure Summary
    print("TABLE 2: Success/Failure Summary")
    print("-" * 80)
    
    for controller in df['Controller'].unique():
        controller_df = df[df['Controller'] == controller]
        print(f"\n{controller}:")
        print(f"  Successful trials: {(controller_df['Success'] == 'Yes').sum()}/{len(controller_df)}")
        
        # Identify failure cases
        failures = controller_df[controller_df['Success'] != 'Yes']
        if len(failures) > 0:
            print(f"  Failed at angles: {failures['Initial Angle (deg)'].tolist()} degrees")
            print(f"  Average steps before failure: {failures['Steps Survived'].mean():.1f}")
        else:
            print(f"  All trials successful!")
    
    print()
    
    # Table 3: Statistical Analysis
    print("TABLE 3: Statistical Summary by Controller")
    print("-" * 80)
    
    stats_data = []
    for controller in df['Controller'].unique():
        controller_df = df[df['Controller'] == controller]
        stats_data.append({
            'Controller': controller,
            'Success Rate (%)': (controller_df['Success'] == 'Yes').sum() / len(controller_df) * 100,
            'Avg Steps': controller_df['Steps Survived'].mean(),
            'Min Steps': controller_df['Steps Survived'].min(),
            'Max Steps': controller_df['Steps Survived'].max(),
            'Std Dev': controller_df['Steps Survived'].std(),
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    print()
    
    # Table 4: Angle Range Analysis
    print("TABLE 4: Maximum Stable Angle Range")
    print("-" * 80)
    
    for controller in df['Controller'].unique():
        controller_df = df[df['Controller'] == controller].sort_values('Initial Angle (deg)')
        successful = controller_df[controller_df['Success'] == 'Yes']
        
        if len(successful) > 0:
            min_angle = successful['Initial Angle (deg)'].min()
            max_angle = successful['Initial Angle (deg)'].max()
            angle_range = max_angle - min_angle
            print(f"{controller:12s}: [{min_angle:+6.1f}°, {max_angle:+6.1f}°]  (range: {angle_range:.1f}°)")
        else:
            print(f"{controller:12s}: No successful trials")
    
    print()
    
    # Observations for report
    print("=" * 80)
    print("KEY OBSERVATIONS FOR REPORT")
    print("=" * 80)
    print()
    
    observations = []
    
    # Observation 1: Overall performance
    overall_success = (df['Success'] == 'Yes').sum() / len(df) * 100
    observations.append(
        f"1. Overall Success Rate: {overall_success:.1f}% of all trials reached the target of 1000 steps."
    )
    
    # Observation 2: Controller comparison
    best_controller = stats_df.loc[stats_df['Success Rate (%)'].idxmax(), 'Controller']
    best_rate = stats_df['Success Rate (%)'].max()
    observations.append(
        f"2. Best Performing Controller: {best_controller} achieved the highest success rate of {best_rate:.1f}%."
    )
    
    # Observation 3: Angle sensitivity
    for controller in df['Controller'].unique():
        controller_df = df[df['Controller'] == controller]
        # Check if performance degrades with larger angles
        sorted_df = controller_df.sort_values('Initial Angle (deg)', key=abs)
        if len(sorted_df) > 2:
            small_angle_success = (sorted_df.iloc[:3]['Success'] == 'Yes').mean()
            large_angle_success = (sorted_df.iloc[-3:]['Success'] == 'Yes').mean()
            if small_angle_success > large_angle_success:
                observations.append(
                    f"3. {controller}: Performance degrades with larger initial angles "
                    f"({small_angle_success*100:.0f}% success at small angles vs {large_angle_success*100:.0f}% at large angles)."
                )
    
    # Observation 4: Symmetric behavior
    for controller in df['Controller'].unique():
        controller_df = df[df['Controller'] == controller]
        pos_angles = controller_df[controller_df['Initial Angle (deg)'] > 0]
        neg_angles = controller_df[controller_df['Initial Angle (deg)'] < 0]
        
        pos_success_rate = (pos_angles['Success'] == 'Yes').mean() * 100
        neg_success_rate = (neg_angles['Success'] == 'Yes').mean() * 100
        
        if abs(pos_success_rate - neg_success_rate) < 10:
            observations.append(
                f"4. {controller}: Shows symmetric behavior (positive angles: {pos_success_rate:.0f}%, "
                f"negative angles: {neg_success_rate:.0f}% success rate)."
            )
        else:
            observations.append(
                f"4. {controller}: Shows asymmetric behavior (positive angles: {pos_success_rate:.0f}%, "
                f"negative angles: {neg_success_rate:.0f}% success rate)."
            )
    
    for obs in observations:
        print(obs)
        print()
    
    # Save observations to file
    obs_file = "results/observations.txt"
    with open(obs_file, 'w') as f:
        f.write("KEY OBSERVATIONS\n")
        f.write("=" * 80 + "\n\n")
        for obs in observations:
            f.write(obs + "\n\n")
    
    print(f"Observations saved to: {obs_file}")
    print()
    
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    load_and_analyze()