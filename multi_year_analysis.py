"""
Multi-Year Dengue Transmission Network Analysis
=============================================
This script runs the dengue network analysis on all three years (2023, 2024, 2025)
and generates comparative results.

Author: Eduardo Trunci, João Trunci
Date: 2025
"""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import logging
from datetime import datetime
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Import the analyzer class from main.py
from main import DengueNetworkAnalyzer

def setup_logging(year):
    """Setup logging for each year's analysis."""
    log_filename = f'dengue_analysis_{year}.log'
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analyze_year(year, csv_path):
    """Run complete analysis for a specific year."""
    logger = setup_logging(year)
    
    logger.info("="*80)
    logger.info(f"DENGUE TRANSMISSION NETWORK ANALYSIS - {year}")
    logger.info("="*80)
    
    # Initialize analyzer
    analyzer = DengueNetworkAnalyzer(csv_path)
    
    try:
        # Step 1: Load and preprocess data
        logger.info(f"\n🔄 Starting analysis for {year}...")
        analyzer.load_and_preprocess_data()
        
        # Step 2: Build transmission network
        analyzer.build_transmission_network()
        
        # Step 3: Compute municipality risk scores
        analyzer.compute_municipality_scores()
        
        # Step 4: Apply discrete Morse theory
        analyzer.apply_discrete_morse_theory()
        
        # Step 5: Identify transmission hubs
        top_hubs, hub_scores = analyzer.identify_transmission_hubs(top_n=20)
        
        # Step 6: Generate summary
        analyzer.generate_summary_report()
        
        # Save results
        results_df = pd.DataFrame([
            {
                'municipality': muni,
                'hub_score': score,
                'risk_score': analyzer.municipality_scores.get(muni, 0),
                'is_saddle': muni in analyzer.critical_points['saddles'],
                'is_maxima': muni in analyzer.critical_points['maxima'],
                'degree': analyzer.graph.degree(muni)
            }
            for muni, score in top_hubs
        ])
        
        results_filename = f'transmission_hubs_{year}.csv'
        results_df.to_csv(results_filename, index=False)
        logger.info(f"✓ Saved results to '{results_filename}'")
        
        # Return summary statistics
        summary = {
            'year': year,
            'total_cases': len(analyzer.df),
            'municipalities': analyzer.graph.number_of_nodes(),
            'transmission_pathways': analyzer.graph.number_of_edges(),
            'network_density': nx.density(analyzer.graph),
            'critical_points': {
                'maxima': len(analyzer.critical_points['maxima']),
                'saddles': len(analyzer.critical_points['saddles']),
                'minima': len(analyzer.critical_points['minima'])
            },
            'top_hubs': top_hubs[:10],  # Top 10 hubs
            'date_range': {
                'start': str(analyzer.df['dt_sin_pri'].min()) if 'dt_sin_pri' in analyzer.df.columns else 'N/A',
                'end': str(analyzer.df['dt_sin_pri'].max()) if 'dt_sin_pri' in analyzer.df.columns else 'N/A'
            }
        }
        
        logger.info(f"✅ Analysis complete for {year}!")
        return summary
        
    except Exception as e:
        logger.error(f"❌ Analysis failed for {year}: {e}", exc_info=True)
        return None

def generate_comparative_analysis(summaries):
    """Generate comparative analysis across all years."""
    logger = setup_logging('comparative')
    
    logger.info("="*80)
    logger.info("COMPARATIVE ANALYSIS ACROSS YEARS")
    logger.info("="*80)
    
    # Create comparison DataFrame
    comparison_data = []
    for summary in summaries:
        if summary:
            comparison_data.append({
                'Year': summary['year'],
                'Total Cases': summary['total_cases'],
                'Municipalities': summary['municipalities'],
                'Transmission Pathways': summary['transmission_pathways'],
                'Network Density': summary['network_density'],
                'Critical Points': sum(summary['critical_points'].values()),
                'Maxima': summary['critical_points']['maxima'],
                'Saddles': summary['critical_points']['saddles'],
                'Minima': summary['critical_points']['minima']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('yearly_comparison.csv', index=False)
    
    logger.info("\n📊 YEARLY COMPARISON SUMMARY:")
    logger.info("="*50)
    for _, row in comparison_df.iterrows():
        logger.info(f"\n{row['Year']}:")
        logger.info(f"  Cases: {row['Total Cases']:,}")
        logger.info(f"  Municipalities: {row['Municipalities']:,}")
        logger.info(f"  Pathways: {row['Transmission Pathways']:,}")
        logger.info(f"  Density: {row['Network Density']:.4f}")
        logger.info(f"  Critical Points: {row['Critical Points']}")
        logger.info(f"    - Maxima: {row['Maxima']}")
        logger.info(f"    - Saddles: {row['Saddles']}")
        logger.info(f"    - Minima: {row['Minima']}")
    
    # Identify consistent hubs across years
    logger.info("\n🔍 CONSISTENT HUBS ACROSS YEARS:")
    logger.info("="*50)
    
    hub_counts = defaultdict(int)
    for summary in summaries:
        if summary:
            for muni, _ in summary['top_hubs']:
                hub_counts[muni] += 1
    
    consistent_hubs = [(muni, count) for muni, count in hub_counts.items() if count >= 2]
    consistent_hubs.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Found {len(consistent_hubs)} municipalities appearing as hubs in multiple years:")
    for muni, count in consistent_hubs[:10]:
        logger.info(f"  {muni}: appears in {count} years")
    
    # Save detailed comparison
    with open('comparative_analysis.json', 'w') as f:
        json.dump({
            'yearly_summaries': summaries,
            'consistent_hubs': consistent_hubs,
            'comparison_data': comparison_data
        }, f, indent=2, default=str)
    
    logger.info("\n✅ Comparative analysis complete!")
    logger.info("📁 Files generated:")
    logger.info("  - yearly_comparison.csv")
    logger.info("  - comparative_analysis.json")
    logger.info("  - transmission_hubs_YYYY.csv (for each year)")
    logger.info("  - dengue_analysis_YYYY.log (for each year)")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("MULTI-YEAR DENGUE TRANSMISSION NETWORK ANALYSIS")
    print("Analyzing 2023, 2024, and 2025 datasets")
    print("="*80 + "\n")
    
    # Define data files
    data_files = {
        2023: "DENGBR23.csv",
        2024: "DENGBR24.csv", 
        2025: "DENGBR25.csv"
    }
    
    # Check which files exist
    available_years = []
    for year, filename in data_files.items():
        if os.path.exists(filename):
            available_years.append(year)
            print(f"✓ Found {filename}")
        else:
            print(f"✗ Missing {filename}")
    
    if not available_years:
        print("❌ No data files found!")
        return
    
    print(f"\n🔄 Starting analysis for years: {available_years}")
    
    # Run analysis for each year
    summaries = []
    for year in available_years:
        print(f"\n{'='*20} ANALYZING {year} {'='*20}")
        summary = analyze_year(year, data_files[year])
        summaries.append(summary)
    
    # Generate comparative analysis
    print(f"\n{'='*20} COMPARATIVE ANALYSIS {'='*20}")
    generate_comparative_analysis(summaries)
    
    print("\n✨ Multi-year analysis complete!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
