def parse_matrix(input_line):
    """Parses a matrix from a single input line."""
    values = list(map(float, input_line.split()))
    rows, cols = int(values[0]), int(values[1])
    matrix = [values[2 + i * cols: 2 + (i + 1) * cols] for i in range(rows)]
    print(f"Parsed matrix ({rows}x{cols}):")
    for row in matrix:
        print(" ".join(map(str, row)))
    return matrix, rows, cols

def viterbi_algorithm(transition_matrix, emission_matrix, initial_state_dist, emissions):
    """
    Viterbi algorithm to compute the most likely sequence of states with detailed logging.
    """
    num_states = len(initial_state_dist[0])
    num_emissions = len(emissions)

    print("\nInitializing Viterbi algorithm...")

    # Initialize delta and delta_idx
    delta = [[0.0] * num_states for _ in range(num_emissions)]
    delta_idx = [[0] * num_states for _ in range(num_emissions)]

    # Step 1: Initialize delta for time t=0
    print("\nStep 1: Initializing delta for t=0")
    for i in range(num_states):
        delta[0][i] = initial_state_dist[0][i] * emission_matrix[i][emissions[0]]
        delta_idx[0][i] = 0  # No previous state for t=0
        print(f"delta[0][{i}] = {initial_state_dist[0][i]} * {emission_matrix[i][emissions[0]]} = {delta[0][i]}")

    # Step 2: Recursion to compute delta for t > 0
    print("\nStep 2: Recursion for t > 0")
    for t in range(1, num_emissions):
        print(f"\nTime step t={t}")
        for i in range(num_states):
            max_prob = float('-inf')
            max_state = 0
            print(f"  Calculating delta[{t}][{i}]...")
            for j in range(num_states):
                prob = delta[t - 1][j] * transition_matrix[j][i]
                print(f"    delta[{t-1}][{j}] * transition[{j}][{i}] = {delta[t-1][j]} * {transition_matrix[j][i]} = {prob}")
                if prob > max_prob:
                    max_prob = prob
                    max_state = j
            delta[t][i] = max_prob * emission_matrix[i][emissions[t]]
            delta_idx[t][i] = max_state
            print(f"  max_prob * emission[{i}][{emissions[t]}] = {max_prob} * {emission_matrix[i][emissions[t]]} = {delta[t][i]}")
            print(f"  delta_idx[{t}][{i}] = {max_state}")

        print("  Delta matrix after this step:")
        for row in delta:
            print(" ".join(map(str, row)))

    # Step 3: Backtrace to find the most likely sequence
    print("\nStep 3: Backtrace to find the most likely sequence")
    most_likely_sequence = [0] * num_emissions
    most_likely_sequence[-1] = max(range(num_states), key=lambda i: delta[num_emissions - 1][i])
    print(f"Starting backtrace. Last state: {most_likely_sequence[-1]}")

    for t in range(num_emissions - 2, -1, -1):
        most_likely_sequence[t] = delta_idx[t + 1][most_likely_sequence[t + 1]]
        print(f"most_likely_sequence[{t}] = delta_idx[{t+1}][{most_likely_sequence[t+1]}] = {most_likely_sequence[t]}")

    print("\nFinal delta matrix:")
    for row in delta:
        print(" ".join(map(str, row)))

    print("\nFinal most likely sequence of states:")
    print(" ".join(map(str, most_likely_sequence)))
    return most_likely_sequence

def main():
    import sys
    input_lines = sys.stdin.read().strip().split("\n")

    # Parse the matrices
    print("\nParsing input matrices...")
    transition_matrix, t_rows, t_cols = parse_matrix(input_lines[0])
    emission_matrix, e_rows, e_cols = parse_matrix(input_lines[1])
    initial_state_distribution, i_rows, i_cols = parse_matrix(input_lines[2])

    # Parse the emissions sequence
    print("\nParsing emissions sequence...")
    emissions_input = list(map(int, input_lines[3].split()))
    num_emissions = emissions_input[0]
    emissions = emissions_input[1:]

    print(f"Number of emissions: {num_emissions}")
    print(f"Emissions sequence: {emissions}")

    # Ensure the number of emissions matches the input
    if len(emissions) != num_emissions:
        raise ValueError("Number of emissions does not match the input sequence length.")

    # Compute the most likely sequence of states using the Viterbi algorithm
    most_likely_sequence = viterbi_algorithm(
        transition_matrix, emission_matrix, initial_state_distribution, emissions
    )

    # Output the result as space-separated values
    print("\nMost likely sequence of states:")
    print(" ".join(map(str, most_likely_sequence)))

if __name__ == "__main__":
    main()
