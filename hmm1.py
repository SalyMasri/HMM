def parse_matrix(input_line):
    """Parses a matrix from a single input line."""
    print(f"\nParsing matrix from input: {input_line}")
    values = list(map(float, input_line.split()))
    rows, cols = int(values[0]), int(values[1])
    matrix = [values[2 + i * cols: 2 + (i + 1) * cols] for i in range(rows)]
    print(f"Parsed matrix ({rows}x{cols}):")
    for row in matrix:
        print(row)
    return matrix, rows, cols


def forward_algorithm(transition_matrix, emission_matrix, initial_state_dist, emissions):
    """
    Forward algorithm to compute the probability of an emission sequence.
    """
    num_states = len(initial_state_dist[0])
    num_emissions = len(emissions)
    print(f"\nNumber of states: {num_states}, Number of emissions: {num_emissions}")

    # Initialize alpha matrix
    alpha = [[0.0] * num_states for _ in range(num_emissions)]

    print("\nStep 1: Initialize alpha at time t=1")
    for i in range(num_states):
        # Using emissions[0] to select emission probability for time t=0
        alpha[0][i] = initial_state_dist[0][i] * emission_matrix[i][emissions[0]]
        print(f"alpha[0][{i}] = {initial_state_dist[0][i]:.4f} * emission[{i}][{emissions[0]}] ({emission_matrix[i][emissions[0]]:.4f}) = {alpha[0][i]:.4f}")

    print("\nAlpha matrix after initialization:")
    for row in alpha:
        print(" ".join(f"{value:.2f}" for value in row))

    print("\nStep 2: Compute alpha for t > 1")
    for t in range(1, num_emissions):
        print(f"\nTime step t={t} (emission sequence value: {emissions[t]})")
        for i in range(num_states):
            alpha_t_i = sum(alpha[t - 1][j] * transition_matrix[j][i] for j in range(num_states))
            # Using emissions[t] to select emission probability for current time step
            alpha[t][i] = alpha_t_i * emission_matrix[i][emissions[t]]
            print(f"  Calculating alpha[{t}][{i}]:")
            for j in range(num_states):
                print(f"    alpha[{t-1}][{j}] * transition[{j}][{i}] = {alpha[t-1][j]:.2f} * {transition_matrix[j][i]:.2f}")
            print(f"  Sum of products: {alpha_t_i:.2f}, emission[{i}][{emissions[t]}] ({emission_matrix[i][emissions[t]]:.2f})")
            print(f"  Result: alpha[{t}][{i}] = {alpha_t_i:.2f} * {emission_matrix[i][emissions[t]]:.2f} = {alpha[t][i]:.2f}")

        print("\nAlpha matrix after time step t=" + str(t) + ":")
        for row in alpha:
            print(" ".join(f"{value:.2f}" for value in row))

    print("\nStep 3: Sum up final alpha values to get the probability of the sequence")
    final_probability = sum(alpha[num_emissions - 1][i] for i in range(num_states))
    print("\nFinal alpha values at t=T:")
    for i in range(num_states):
        print(f"alpha[{num_emissions - 1}][{i}] = {alpha[num_emissions - 1][i]:.2f}")
    print(f"\nFinal probability of the sequence: {final_probability:.4f}")
    return final_probability


def main():
    import sys
    input_lines = sys.stdin.read().strip().split("\n")

    print("\nParsing matrices and emission sequence...")
    # Parse the matrices
    transition_matrix, t_rows, t_cols = parse_matrix(input_lines[0])
    emission_matrix, e_rows, e_cols = parse_matrix(input_lines[1])
    initial_state_distribution, i_rows, i_cols = parse_matrix(input_lines[2])

    # Parse the emissions sequence
    emissions_input = list(map(int, input_lines[3].split()))
    num_emissions = emissions_input[0]
    emissions = emissions_input[1:]
    print(f"\nEmissions sequence: {emissions}")

    # Ensure the number of emissions matches the input
    if len(emissions) != num_emissions:
        raise ValueError("Number of emissions does not match the input sequence length.")

    print("\nStarting forward algorithm...")
    # Compute the probability using the forward algorithm
    probability = forward_algorithm(transition_matrix, emission_matrix, initial_state_distribution, emissions)

    # Output the result
    print(f"\nFinal probability of the emission sequence: {probability:.6f}")


if __name__ == "__main__":
    main()
