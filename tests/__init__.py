from ece_calculator import get_ece

ece = get_ece(y_true_flat, mean_flat, total_std_flat)
print(f"Regression ECE: {ece:.2f}%")
