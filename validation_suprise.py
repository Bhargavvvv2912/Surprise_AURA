import sys
import numpy as np
import pandas as pd

def test_surprise_functionality():
    print("--- Surprise (JOSS 2020) Functional Verification ---")
    
    try:
        # Import inside try to catch Binary Incompatibility early
        from surprise import SVD, Dataset, Reader, accuracy
        from surprise.model_selection import train_test_split
        
        print("--> Libraries imported successfully.")

        # 1. Create a synthetic dataset to avoid downloading external files
        ratings_dict = {
            "item": [1, 2, 1, 2, 1, 2, 1, 2],
            "user": ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
            "rating": [1, 2, 2, 4, 2.5, 4, 4.5, 5],
        }
        df = pd.DataFrame(ratings_dict)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
        
        # 2. Split and Train
        # fit() calls the underlying Cythonized C-extensions
        print("--> Training SVD model (Triggers Cython/NumPy extensions)...")
        trainset, testset = train_test_split(data, test_size=0.25)
        algo = SVD()
        algo.fit(trainset)

        # 3. Predict and Validate
        print("--> Generating predictions...")
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        
        if rmse is not None:
            print(f"    [✓] Model validated. RMSE: {rmse:.4f}")
            print("--- SMOKE TEST PASSED ---")

    except ImportError as ie:
        print(f"CRITICAL BINARY ERROR: {str(ie)}")
        print("Likely caused by NumPy 2.0+ ABI incompatibility.")
        sys.exit(1)
    except AttributeError as ae:
        print(f"CRITICAL API ERROR: {str(ae)}")
        print("Likely caused by removed NumPy aliases.")
        sys.exit(1)
    except Exception as e:
        print(f"UNEXPECTED FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_surprise_functionality()