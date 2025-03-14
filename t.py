import numpy as np
import time
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from concurrent.futures import ThreadPoolExecutor


def fish_hook_dist(fish_pos, hook_pos):
    x = abs(fish_pos[0] - hook_pos[0])
    return min(20 - x, x) + abs(hook_pos[1] - fish_pos[1])


def calc_heuristic(node):
    score_diff = node.state.player_scores[0] - node.state.player_scores[1]
    max_fish_value = 0
    opponent_advantage = 0
    blocking_val = 0
    gen_av = 0

    my_hook_pos = node.state.hook_positions[0]
    opp_hook_pos = node.state.hook_positions[1]
    for fish_id, fish_pos in node.state.fish_positions.items():
        sign = +1
        fish_score = node.state.fish_scores[fish_id]
        if fish_score <= 0:
            sign = -1
            continue  # Skip undesirable fish
        # hook distances
        my_distance = fish_hook_dist(fish_pos, my_hook_pos)
        opponent_distance = fish_hook_dist(fish_pos, opp_hook_pos)
        # if my_distance == 0:
        # return float("inf")
        # elif opponent_distance == 0:
        # return float("-inf")
        # Weight fish based on score and accessibility
        weighted_value = fish_score * sign / (my_distance + 1)
        max_fish_value = max(max_fish_value, weighted_value)

        # Opp's advantage if closer to valuable fish
        if opponent_distance < my_distance:
            opponent_advantage += fish_score / (opponent_distance + 1)

        # Blocking evaluation: does moving here block the opponent's horizontal access?
        if my_distance < opponent_distance and abs(my_hook_pos[0] - fish_pos[0]) < abs(opp_hook_pos[0] - fish_pos[0]):
            blocking_val += fish_score / (opponent_distance + 1)  # reward blocking
        # Prioritize score difference, accessible valuable fish, and blocking opportunities

    heuristic_value = score_diff + max_fish_value - 0.1 * opponent_advantage + 0.0 * blocking_val
    return heuristic_value


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.state_cache = {}
        self.Max_t = 0.062
        self.Start_t = None

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """
        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(init_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, init_node):
        """
        Iterative Deepening Search with Parallel Top-Level Move Evaluation.
        :param initial_tree_node: Initial game tree node
        :return: Best move as a string
        """
        self.Start_t = time.time()

        depth = 1
        best_move = 0
        best_value = float("-inf")

        while True:
            try:
                moves = init_node.compute_and_get_children()

                # Evaluate top-level moves in parallel
                with ThreadPoolExecutor() as executor:
                    futures = {
                        executor.submit(self.alpha_beta_pruning, child, depth, float("-inf"), float("inf"),
                                        1): child.move
                        for child in moves
                    }

                    for future in futures:
                        move = futures[future]
                        value = future.result()  # Wait for result

                        if value > best_value:
                            best_value = value
                            best_move = move

                # Increment depth for next iteration
                depth += 1

                if time.time() - self.Start_t > self.Max_t:
                    break  # Stop deepening if time runs out
            except TimeoutError:
                break

        return ACTION_TO_STR[best_move]

    def alpha_beta_pruning(self, node, depth, alpha, beta, current_player):
        """
        Minimax with Alpha-Beta Pruning and State Caching.
        :param node: Current node
        :param depth: Current depth
        :param alpha: Alpha value
        :param beta: Beta value
        :param current_player: Current player (0 or 1)
        :return: Evaluated heuristic value
        """
        if time.time() - self.Start_t >= self.Max_t:
            raise TimeoutError

        state_key = (
                str(node.state.get_hook_positions())
                + str(node.state.get_fish_positions())
                + str(node.state.player_scores)
        )

        if state_key in self.state_cache:
            return self.state_cache[state_key]

        if depth == 0 or not node.compute_and_get_children():
            heur_val = calc_heuristic(node)
            self.state_cache[state_key] = heur_val
            return heur_val

        if current_player == 0:
            max_eval = float("-inf")
            for child in node.compute_and_get_children():
                child_value = self.alpha_beta_pruning(child, depth - 1, alpha, beta, 1)
                max_eval = max(max_eval, child_value)
                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break  # Prune
            self.state_cache[state_key] = max_eval
            return max_eval
        else:
            min_eval = float("inf")
            for child in node.compute_and_get_children():
                child_value = self.alpha_beta_pruning(child, depth - 1, alpha, beta, 0)
                min_eval = min(min_eval, child_value)
                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  # Prune
            self.state_cache[state_key] = min_eval
            return min_eval
