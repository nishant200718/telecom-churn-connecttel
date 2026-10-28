#!/usr/bin/env python3
"""
ConnectTel Telecommunications - Production Churn Scoring CLI
Usage:
    python predict.py --input sample_customers.json
    python predict.py --input new_subscribers.csv --output scored_subscribers.csv
"""

import os
import sys
import json
import argparse
import joblib
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_PATH = os.path.join(CURRENT_DIR, "churn_pipeline.joblib")

def main():
    parser = argparse.ArgumentParser(description="ConnectTel Production Churn Inference Engine")
    parser.add_argument("--input", required=True, help="Path to input JSON or CSV file containing subscriber records")
    parser.add_argument("--output", default=None, help="Optional output file path to write predictions (JSON or CSV)")
    parser.add_argument("--threshold", type=float, default=0.28, help="Classification probability threshold (default: 0.28)")
    
    args = parser.parse_args()
    
    if not os.path.exists(PIPELINE_PATH):
        print(f"Error: Production pipeline artifact not found at {PIPELINE_PATH}")
        sys.exit(1)
        
    print(f"Loading production pipeline from {PIPELINE_PATH}...")
    pipeline = joblib.load(PIPELINE_PATH)
    pipeline.business_threshold = args.threshold
    
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        sys.exit(1)
        
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df_input = pd.DataFrame(data)
    elif input_path.endswith(".csv"):
        df_input = pd.read_csv(input_path)
    else:
        print("Error: Input file must be .json or .csv")
        sys.exit(1)
        
    print(f"Scoring {len(df_input)} customer records with threshold {args.threshold}...")
    predictions = pipeline.predict_risk(df_input)
    
    # Merge predictions with input records
    pred_df = pd.DataFrame(predictions)
    output_df = pd.concat([df_input.reset_index(drop=True), pred_df.drop(columns=['customer_index'])], axis=1)
    
    if args.output:
        if args.output.endswith(".json"):
            output_df.to_json(args.output, orient="records", indent=2)
        else:
            output_df.to_csv(args.output, index=False)
        print(f"Results successfully saved to: {args.output}")
    else:
        print("\n--- INFERENCE RESULTS ---")
        print(json.dumps(predictions, indent=2))
        
if __name__ == "__main__":
    main()
