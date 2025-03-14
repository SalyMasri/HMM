def parse_matrix(input_line):
    """Parses a matrix from a single input line."""
    print("Parsing matrix from input line:")
    print(input_line)
    values = list(map(float, input_line.split()))
    rows, cols = int(values[0]), int(values[1])
    matrix = [values[2 + i * cols: 2 + (i + 1) * cols] for i in range(rows)]
    print(f"Parsed matrix ({rows}x{cols}):")
    for row in matrix:
        print(row)
    print()
    return matrix, rows, cols

def multiply_matrices(A, B):
    """Performs matrix multiplication for matrices A and B."""
    print("Multiplying matrices:")
    print("Matrix A:")
    for row in A:
        print(row)
    print("Matrix B:")
    for row in B:
        print(row)
    print()
    
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions do not match for multiplication.")
    
    result = [[sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)] for i in range(rows_A)]
    
    print("Result of multiplication:")
    for row in result:
        print(row)
    print()
    
    return result

def main():
    import sys
    input_lines = sys.stdin.read().strip().split("\n")
    
    print("Starting HMM computation...\n")
    
    # Parse the matrices
    print("Parsing input matrices...\n")
    transition_matrix, t_rows, t_cols = parse_matrix(input_lines[0])
    emission_matrix, e_rows, e_cols = parse_matrix(input_lines[1])
    initial_state_distribution, i_rows, i_cols = parse_matrix(input_lines[2])
    
    # Step 1: Multiply initial state distribution with transition matrix
    print("Step 1: Multiplying initial state distribution with transition matrix...\n")
    state_distribution_after_transition = multiply_matrices(initial_state_distribution, transition_matrix)
    
    # Step 2: Multiply result with emission matrix
    print("Step 2: Multiplying state distribution with emission matrix...\n")
    emission_probability_distribution = multiply_matrices(state_distribution_after_transition, emission_matrix)
    
    # Output result
    print("Final result:")
    rows, cols = len(emission_probability_distribution), len(emission_probability_distribution[0])
    result_flattened = [f"{x:.6f}" for row in emission_probability_distribution for x in row]
    print(f"{rows} {cols} {' '.join(result_flattened)}")

if __name__ == "__main__":
    main()
