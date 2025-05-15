from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def load_labels(filename):
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f if line.strip()]

def compare_kmeans_outputs(seq_file, par_file):
    seq_labels = load_labels(seq_file)
    par_labels = load_labels(par_file)

    if len(seq_labels) != len(par_labels):
        print("ERROR: Number of labels doesn't match.")
        return

    ari = adjusted_rand_score(seq_labels, par_labels)
    nmi = normalized_mutual_info_score(seq_labels, par_labels)

    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")

# Usage
compare_kmeans_outputs("result_seq", "result_omp")

# python compare_module.py file1.txt file2.txt
