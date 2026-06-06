# ============================================
# UBER FARE PREDICTION - MAIN PIPELINE
# ============================================

import subprocess
import sys

def run_pipeline():
    print("="*60)
    print("🚕 UBER FARE PREDICTION PIPELINE")
    print("="*60)
    
    steps = [
        ("data_preprocessing.py", "📊 Data Preprocessing"),
        ("model_training_evaluation.py", "🤖 Model Training & Evaluation"),
        ("hyperparameter_tuning.py", "⚙️ Hyperparameter Tuning")
    ]
    
    for file, description in steps:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print('='*60)
        
        result = subprocess.run([sys.executable, file], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print(result.stdout[-500:])  # آخر 500 حرف
        else:
            print(f"❌ Error in {description}")
            print(result.stderr)
            break
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()