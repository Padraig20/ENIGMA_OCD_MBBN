#!/usr/bin/env python3
import os
import numpy as np
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your VMD function from wherever it is defined
# For example, if VMD is in vmdpy:
from sktime.libs.vmdpy import VMD

# Base path for data (destination directory for the subjects)
base_path = "/pscratch/sd/p/pakmasha/UKB_304_ROIs"

# VMD parameters
K = 4             # modes
DC = 0            # no DC part imposed
init = 0          # initialize omegas uniformly
tol = 1e-7        # convergence tolerance
# For testing, reduced ranges:
alpha_values = np.arange(100, 1100, 100)  # bandwidth constraint
tau_values = np.arange(0.5, 4.0, 0.5)       # noise-tolerance

# This function processes one subject for a given (alpha, tau) pair.
def process_subject(subject_id, alpha, tau):
    try:
        subject_path = os.path.join(base_path, subject_id)
        subject_file = os.path.join(subject_path, f"schaefer_400Parcels_17Networks_{subject_id}.npy")
        time_series_data = np.load(subject_file)
        # Transpose to shape: (ROIs, seq_len)
        y = time_series_data.T
        ts_length = y.shape[1]
        # Compute average across ROIs using vectorized operation
        sample_whole = np.mean(y, axis=0)
        # Z-score normalization
        sample_whole = (sample_whole - np.mean(sample_whole)) / np.std(sample_whole)
        # If length is odd, remove the last element
        if len(sample_whole) % 2:
            sample_whole = sample_whole[:-1]
        # Run VMD
        u, _, _ = VMD(sample_whole, alpha, tau, K, DC, init, tol)
        # Reconstruct the signal by summing the IMFs
        reconstructed_signal = np.sum(u, axis=0)
        # Compute MSE and RRE metrics
        mse = np.mean((sample_whole - reconstructed_signal) ** 2)
        rre = np.linalg.norm(sample_whole - reconstructed_signal) / np.linalg.norm(sample_whole)
        return mse, rre
    except Exception as e:
        print(f"Error processing subject {subject_id} for alpha={alpha}, tau={tau}: {e}", flush=True)
        return None

def main():
    # Get list of subjects (subfolder names) in base_path
    subject_list = [s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s))]
    subject_list.sort()

    results = {}  # to store results for each (alpha, tau) pair
    best_mse = float('inf')
    best_rre = float('inf')
    best_alpha_mse, best_tau_mse = None, None
    best_alpha_rre, best_tau_rre = None, None

    # Loop over all (alpha, tau) pairs
    for alpha in alpha_values:
        for tau in tau_values:
            print(f"\nProcessing for Alpha = {alpha}, tau = {tau}", flush=True)
            total_mse = 0.0
            total_rre = 0.0
            subject_count = 0

            # Process subjects in parallel using ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                futures = {executor.submit(process_subject, subject_id, alpha, tau): subject_id for subject_id in subject_list}
                for i, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    if result is not None:
                        mse, rre = result
                        total_mse += mse
                        total_rre += rre
                        subject_count += 1
                    if i % 500 == 0:
                        print(f"Processed {i} subjects for alpha={alpha}, tau={tau}: Total MSE = {total_mse}, Total RRE = {total_rre}, Count = {subject_count}", flush=True)

            # Compute mean errors for current (alpha, tau)
            mean_mse = total_mse / subject_count if subject_count > 0 else float('inf')
            mean_rre = total_rre / subject_count if subject_count > 0 else float('inf')
            # Use a string key instead of a tuple to ensure JSON compatibility.
            key = f"{int(alpha)}_{float(tau)}"
            results[key] = {
                "mean_mse": float(mean_mse),
                "mean_rre": float(mean_rre),
                "subject_count": int(subject_count)
            }
            
            # Update best parameters based on mean_mse and mean_rre
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_alpha_mse, best_tau_mse = alpha, tau
            if mean_rre < best_rre:
                best_rre = mean_rre
                best_alpha_rre, best_tau_rre = alpha, tau

            print(f"Finished alpha = {alpha}, tau = {tau}: mean MSE = {mean_mse:.6f}, mean RRE = {mean_rre:.6f}", flush=True)

    # Prepare final results with native Python types
    final_results = {
        "results_by_param": results,
        "best_by_mse": {
            "alpha": int(best_alpha_mse) if best_alpha_mse is not None else None,
            "tau": float(best_tau_mse) if best_tau_mse is not None else None,
            "mean_mse": float(best_mse)
        },
        "best_by_rre": {
            "alpha": int(best_alpha_rre) if best_alpha_rre is not None else None,
            "tau": float(best_tau_rre) if best_tau_rre is not None else None,
            "mean_rre": float(best_rre)
        }
    }
    
    # Write final results to a JSON file
    output_file = "/pscratch/sd/p/pakmasha/ENIGMA_OCD_MBBN_git/ENIGMA_OCD_MBBN/ENIGMA-OCD_results/QC/vmd_tuning_results.json"
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4)
    
    # Print final results summary
    print(f"\nBest (alpha, tau) for MSE: ({best_alpha_mse}, {best_tau_mse}) with mean MSE = {best_mse:.6f}", flush=True)
    print(f"Best (alpha, tau) for RRE: ({best_alpha_rre}, {best_tau_rre}) with mean RRE = {best_rre:.6f}", flush=True)
    print(f"Detailed results have been saved to {output_file}", flush=True)

if __name__ == '__main__':
    main()
