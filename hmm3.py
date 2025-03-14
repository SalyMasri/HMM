import math

def parse_matrix(line):
    """Parse a matrix from a line of input."""
    elements = list(map(float, line.strip().split()))
    rows, cols = int(elements[0]), int(elements[1])
    matrix = [elements[2 + i * cols:2 + (i + 1) * cols] for i in range(rows)]
    return matrix, rows, cols

def print_matrix(matrix, name):
    """Print a matrix row by row with a label."""
    print(f"\n{name} Matrix ({len(matrix)}x{len(matrix[0])}):")
    for row in matrix:
        print(" ".join(f"{x:.6f}" for x in row))

def forward_algorithm(A, B, pi, emissions):
    """Forward pass with scaling for numerical stability."""
    N = len(A)
    T = len(emissions)
    alpha = [[0.0] * N for _ in range(T)]
    scale = [0.0] * T  # Scaling factors for each time step

    print("\n--- Forward Algorithm ---")

    # Initialize alpha for t=0
    print("\nInitializing alpha for t=0:")
    for i in range(N):
        alpha[0][i] = pi[0][i] * B[i][emissions[0]]
        scale[0] += alpha[0][i]
    scale[0] = 1.0 / scale[0]
    for i in range(N):
        alpha[0][i] *= scale[0]
    print_matrix(alpha, "Alpha (t=0)")

    # Compute alpha for t > 0
    for t in range(1, T):
        print(f"\nComputing alpha for t={t}:")
        for i in range(N):
            for j in range(N):
                alpha[t][i] += alpha[t - 1][j] * A[j][i]
            alpha[t][i] *= B[i][emissions[t]]
            scale[t] += alpha[t][i]
        scale[t] = 1.0 / scale[t]
        for i in range(N):
            alpha[t][i] *= scale[t]
        print_matrix(alpha, f"Alpha (t={t})")

    return alpha, scale

def backward_algorithm(A, B, emissions, scale):
    """Backward pass with scaling for numerical stability."""
    N = len(A)
    T = len(emissions)
    beta = [[0.0] * N for _ in range(T)]

    print("\n--- Backward Algorithm ---")

    # Initialize beta at time T-1
    print("\nInitializing beta for t=T-1:")
    for i in range(N):
        beta[T - 1][i] = scale[T - 1]
    print_matrix(beta, "Beta (t=T-1)")

    # Compute beta for t < T-1
    for t in range(T - 2, -1, -1):
        print(f"\nComputing beta for t={t}:")
        for i in range(N):
            for j in range(N):
                beta[t][i] += A[i][j] * B[j][emissions[t + 1]] * beta[t + 1][j]
            beta[t][i] *= scale[t]
        print_matrix(beta, f"Beta (t={t})")

    return beta

def log_prob(scale):
    """Compute log(P(O|lambda)) using scaling factors."""
    return -sum(math.log(s) for s in scale)

def baum_welch(A, B, pi, emissions, max_iterations=100, threshold=1e-6):
    """Train HMM using Baum-Welch algorithm with scaling."""
    N = len(A)
    M = len(B[0])
    T = len(emissions)

    old_log_prob = -float("inf")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")

        # Forward and backward passes
        alpha, scale = forward_algorithm(A, B, pi, emissions)
        beta = backward_algorithm(A, B, emissions, scale)

        # Compute gamma and di-gamma
        gamma = [[0.0] * N for _ in range(T)]
        di_gamma = [[[0.0] * N for _ in range(N)] for _ in range(T - 1)]

        print("\nComputing gamma and di-gamma:")
        for t in range(T - 1):
            for i in range(N):
                gamma[t][i] = sum(
                    alpha[t][i] * A[i][j] * B[j][emissions[t + 1]] * beta[t + 1][j]
                    for j in range(N)
                )
                for j in range(N):
                    di_gamma[t][i][j] = (
                        alpha[t][i] * A[i][j] * B[j][emissions[t + 1]] * beta[t + 1][j]
                    )
        for i in range(N):
            gamma[T - 1][i] = alpha[T - 1][i]
        print_matrix(gamma, "Gamma")

        # Re-estimate A, B, and pi
        print("\nRe-estimating A, B, and pi:")
        for i in range(N):
            # Update pi
            pi[0][i] = gamma[0][i]

            # Update A
            denom = sum(gamma[t][i] for t in range(T - 1))
            for j in range(N):
                numer = sum(di_gamma[t][i][j] for t in range(T - 1))
                A[i][j] = numer / denom if denom != 0 else 0.0

            # Update B
            denom = sum(gamma[t][i] for t in range(T))
            for k in range(M):
                numer = sum(gamma[t][i] for t in range(T) if emissions[t] == k)
                B[i][k] = numer / denom if denom != 0 else 0.0

        print_matrix(A, "Updated A")
        print_matrix(B, "Updated B")

        # Check for convergence using log probability
        log_prob_curr = log_prob(scale)
        print(f"Log Probability: {log_prob_curr:.6f}")
        if log_prob_curr - old_log_prob < threshold:
            print("Convergence achieved.")
            break
        old_log_prob = log_prob_curr

    return A, B

def main():
    import sys
    input_lines = sys.stdin.read().strip().split("\n")

    A, _, _ = parse_matrix(input_lines[0])
    B, _, _ = parse_matrix(input_lines[1])
    pi, _, _ = parse_matrix(input_lines[2])
    emissions_input = list(map(int, input_lines[3].split()))
    emissions = emissions_input[1:]

    A, B = baum_welch(A, B, pi, emissions)

    print_matrix(A, "Final A")
    print_matrix(B, "Final B")

if __name__ == "__main__":
    main()
