import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Poker environment
class PokerEnvironment:
    def __init__(self, big_blind=10, starting_chips=1000):
        self.big_blind = big_blind
        self.starting_chips = starting_chips
        self.raise_occurred = False
        self.reset()

    def reset(self):
        self.deck = self._initialize_deck()
        random.shuffle(self.deck)

        self.pot = 0
        self.players = [{"chips": self.starting_chips, "current_bet": 0, "folded": False, "hand": []} for _ in range(2)]
        self.current_player = 0
        self.current_bet = 0
        self.round_stage = "pre-flop"
        self.round_over = False
        self.community_cards = []
        self.raise_occurred = False

        # Deal hands
        for player in self.players:
            player["hand"] = [self.deck.pop(), self.deck.pop()]

        self.action_log = []

    def _initialize_deck(self):
        suits = ["♠", "♥", "♦", "♣"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
        return [rank + suit for suit in suits for rank in ranks]

    def set_blinds(self, small_blind_player):
        big_blind_player = 1 - small_blind_player

        # Deduct blinds from respective players
        self.players[small_blind_player]["chips"] -= self.big_blind // 2
        self.players[small_blind_player]["current_bet"] = self.big_blind // 2

        self.players[big_blind_player]["chips"] -= self.big_blind
        self.players[big_blind_player]["current_bet"] = self.big_blind

        # Update pot and current bet
        self.pot += self.big_blind + self.big_blind // 2
        self.current_bet = self.big_blind

        # Set the current player to the one after the big blind
        self.current_player = (big_blind_player + 1) % 2

    def deal_community_cards(self):
        if self.round_stage == "flop":
            self.community_cards.extend([self.deck.pop() for _ in range(3)])
        elif self.round_stage in ["turn", "river"]:
            self.community_cards.append(self.deck.pop())

    def take_action(self, action):
        player = self.players[self.current_player]
        opponent = self.players[1 - self.current_player]

        if action == "fold":
            player["folded"] = True
            self.round_over = True
            return f"Player {2 - self.current_player} wins by fold!"

        elif action == "call":
            to_call = self.current_bet - player["current_bet"]
            if to_call > player["chips"]:  # All-in scenario
                to_call = player["chips"]  # Partial call
            player["chips"] -= to_call
            player["current_bet"] += to_call
            self.pot += to_call

        elif action == "check":
            # Check is invalid if there's an active bet
            if self.current_bet > player["current_bet"]:
                return "Invalid action: Cannot check when there's a bet."

        elif action == "raise":
            to_raise = max(self.big_blind, self.current_bet * 2)
            if to_raise > player["chips"] + player["current_bet"]:  # All-in scenario
                to_raise = player["chips"] + player["current_bet"]
            raise_amount = to_raise - player["current_bet"]
            player["chips"] -= raise_amount
            player["current_bet"] = to_raise
            self.pot += raise_amount
            self.current_bet = to_raise
            self.raise_occurred = True

        # Handle the case where a player goes all-in
        if player["chips"] == 0:
            self.all_in = True  # Mark the hand as having an all-in

        # Check if betting is complete
        if player["current_bet"] == opponent["current_bet"]:
            if self.round_stage == "river":
                self.round_over = True
            else:
                self.advance_round()
        else:
            self.current_player = 1 - self.current_player

    def advance_round(self):
        stages = ["pre-flop", "flop", "turn", "river"]
        current_index = stages.index(self.round_stage)
        if current_index < len(stages) - 1:
            self.round_stage = stages[current_index + 1]
            self.deal_community_cards()
            self.current_bet = 0
            self.raise_occurred = False
            for player in self.players:
                player["current_bet"] = 0

    def get_state(self):
        return {
            "pot": self.pot,
            "players": self.players,
            "current_player": self.current_player + 1,
            "current_bet": self.current_bet,
            "round_stage": self.round_stage,
            "community_cards": self.community_cards,
        }


# Q-Learning agent
class DeepQLearningAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.5, epsilon=1.0, epsilon_decay=0.995,
                 epsilon_min=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.model = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = []

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)

            target = reward
            if not done:
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            q_target = q_values.clone().detach()
            q_target[action] = target

            self.optimizer.zero_grad()
            loss = self.loss_fn(q_values, q_target)
            loss.backward()  # Backpropagation step
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# semi-random agent
def calculate_preflop_strength(hand):
    values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
              '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    card1, card2 = hand
    rank1, rank2 = card1[:-1], card2[:-1]
    suit1, suit2 = card1[-1], card2[-1]

    strength = values[rank1] + values[rank2]
    if suit1 == suit2:  # Suited
        strength += 1
    if rank1 == rank2:  # Pocket Pair
        strength += 2

    return strength


def calculate_postflop_game_value(hand_strength, board_cards, stage):
    board_strength = sum([int(card[:-1]) if card[:-1].isdigit() else
                          {'J': 11, 'Q': 12, 'K': 13, 'A': 14}[card[:-1]] for card in board_cards])
    weights = {'flop': (0.9, 0.1), 'turn': (0.4, 0.6), 'river': (0.1, 0.9)}
    hand_weight, board_weight = weights[stage]
    return (hand_strength * hand_weight) + (board_strength * board_weight)


def semi_random_agent(hand, community_cards, stage, current_bet, raise_occurred):
    """
    A semi-random agent that takes actions based on hand strength and current stage.
    Prevents re-raises if a raise has already occurred.
    """
    if stage == "pre-flop":
        hand_strength = calculate_preflop_strength(hand)
        if hand_strength < 20:
            return "fold"
        elif 20 <= hand_strength <= 25:
            return "call" if current_bet > 0 else "check"
        else:
            return "call" if raise_occurred else "raise"
    else:
        hand_strength = calculate_preflop_strength(hand)
        game_value = calculate_postflop_game_value(hand_strength, community_cards, stage)
        if game_value < 20:
            return "fold"
        elif 20 <= game_value <= 30:
            return "call" if current_bet > 0 else "check"
        else:
            return "call" if raise_occurred else "raise"


def evaluate_hand(cards):
    """
    Evaluate the best 5-card hand from a given set of cards (hole + community).
    Returns a tuple (hand_rank, high_card_values) where:
        - hand_rank: Integer representing the hand type (e.g., 9 = royal flush, 8 = straight flush, etc.).
        - high_card_values: List of card values for tie-breaking.
    """
    values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
              '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    ranks = sorted([values[card[:-1]] for card in cards], reverse=True)
    suits = [card[-1] for card in cards]

    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}

    # Helper functions
    def is_flush():
        flush_suit = None
        for suit, count in suit_counts.items():
            if count >= 5:
                flush_suit = suit
                break
        if flush_suit:
            flush_cards = [card for card in cards if card[-1] == flush_suit]
            return True, sorted([values[card[:-1]] for card in flush_cards], reverse=True)
        return False, None

    def is_straight():
        unique_ranks = sorted(set(ranks))
        for i in range(len(unique_ranks) - 4):
            if unique_ranks[i:i + 5] == list(range(unique_ranks[i], unique_ranks[i] + 5)):
                return True, unique_ranks[i + 4]  # Highest card in the straight
        return False, None

    # Check for poker hands
    flush, flush_cards = is_flush()
    straight, high_straight_card = is_straight()

    if flush and straight:
        # Check if it's a Royal Flush
        if set([14, 13, 12, 11, 10]).issubset(flush_cards):
            return 9, [14]  # Royal Flush (highest card is Ace)
        return 8, [high_straight_card]  # Straight Flush
    if flush:
        return 5, flush_cards[:5]  # Flush (highest 5 cards)
    if straight:
        return 4, [high_straight_card]  # Straight

    rank_count_list = sorted(rank_counts.items(), key=lambda x: (-x[1], -x[0]))
    most_common = rank_count_list[0]

    if most_common[1] == 4:
        return 7, [most_common[0]] + [rank for rank in ranks if rank != most_common[0]]  # Four of a kind
    if most_common[1] == 3:
        second_common = rank_count_list[1]
        if second_common[1] >= 2:
            return 6, [most_common[0], second_common[0]]  # Full house
        return 3, [most_common[0]] + [rank for rank in ranks if rank != most_common[0]]  # Three of a kind
    if most_common[1] == 2:
        second_pair = rank_count_list[1]
        if second_pair[1] == 2:
            return 2, [most_common[0], second_pair[0]] + [rank for rank in ranks if
                                                          rank not in (most_common[0], second_pair[0])]  # Two pair
        return 1, [most_common[0]] + [rank for rank in ranks if rank != most_common[0]]  # One pair

    return 0, ranks[:5]  # High card (highest 5 cards)

def determine_winner(hole_cards1, hole_cards2, community_cards):
    """
    Determine the winner between two players based on their hole cards and community cards.
    Returns 0 if Player 1 wins, 1 if Player 2 wins, and -1 if it's a tie.
    """
    best_hand1 = evaluate_hand(hole_cards1 + community_cards)
    best_hand2 = evaluate_hand(hole_cards2 + community_cards)

    # Compare hand ranks
    if best_hand1[0] > best_hand2[0]:
        return 0
    if best_hand1[0] < best_hand2[0]:
        return 1

    # Compare high card values for tie-breaking
    for v1, v2 in zip(best_hand1[1], best_hand2[1]):
        if v1 > v2:
            return 0
        if v1 < v2:
            return 1

    return -1

# Game simulation
def simulate_game_with_logs(env, dqn_agent, num_hands=10):
    player_chips = [env.starting_chips, env.starting_chips]
    small_blind_player = 0

    dqn_wins = 0
    sr_wins = 0
    tie_count = 0

    Game_Hand_Num = []
    P1_Chip = []
    P2_Chip = []

    for hand in range(num_hands):
        # Check stopping condition
        if player_chips[0] <= 0:
            print("\nGame Over: Player 1 is out of chips!")
            print(f"Player 2 wins with {player_chips[1]} chips remaining.")
            break
        if player_chips[1] <= 0:
            print("\nGame Over: Player 2 is out of chips!")
            print(f"Player 1 wins with {player_chips[0]} chips remaining.")
            break

        print(f"\n=== Starting Hand {hand + 1} ===")
        env.reset()
        env.players[0]["chips"] = player_chips[0]
        env.players[1]["chips"] = player_chips[1]
        Game_Hand_Num.append(hand + 1)
        P1_Chip.append(env.players[0]['chips'])
        P2_Chip.append(env.players[1]['chips'])
        env.all_in = False  # Reset the all-in flag

        env.set_blinds(small_blind_player)
        print(f"Player {small_blind_player + 1} is the Small Blind.")
        print(f"Player {2 - small_blind_player} is the Big Blind.")

        print(f"Player 1's Hand: {env.players[0]['hand']}")
        print(f"Player 2's Hand: {env.players[1]['hand']}")

        while not env.round_over:
            state = np.array(
                [env.pot, env.players[0]["chips"], env.players[1]["chips"], env.current_bet, env.current_player]
            )
            print(f"State: Pot={env.pot}, Player1 Chips={env.players[0]['chips']}, "
                  f"Player2 Chips={env.players[1]['chips']}, Current Bet={env.current_bet}, "
                  f"Current Player={env.current_player + 1}, Round Stage={env.round_stage}, "
                  f"Community Cards={env.community_cards}")

            if env.all_in:  # Allow only check or forced actions in all-in scenarios
                if env.current_bet > env.players[env.current_player]["current_bet"]:
                    # Check is invalid, force a random choice between call and fold
                    to_call = env.current_bet - env.players[env.current_player]["current_bet"]
                    if env.players[env.current_player]["chips"] >= to_call:
                        action_name = random.choice(["call", "fold"])
                    elif env.players[env.current_player]["chips"] > 0:  # Partial call allowed
                        action_name = "call"
                    else:
                        action_name = "fold"  # Cannot afford to call, must fold
                else:
                    action_name = "check"
            elif env.current_player == 0:  # DQN Agent
                action = dqn_agent.act(state)
                action_name = ["fold", "call", "check", "raise"][action]
            else:  # Semi-random Agent
                action_name = semi_random_agent(
                    env.players[1]["hand"],
                    env.community_cards,
                    env.round_stage,
                    env.current_bet,
                    env.raise_occurred
                )

            print(f"Player {env.current_player + 1} takes action: {action_name}")
            result = env.take_action(action_name)

            if result:
                print(f"Result: {result}")

            next_state = np.array(
                [env.pot, env.players[0]["chips"], env.players[1]["chips"], env.current_bet, env.current_player]
            )
            done = env.round_over

            reward = 0

            if env.round_over:
                if env.players[0]["folded"]:
                    winner = 1
                    sr_wins += 1
                    reward -= 10
                elif env.players[1]["folded"]:
                    winner = 0
                    dqn_wins += 1
                    reward += 10
                else:
                    winner = determine_winner(
                        env.players[0]["hand"],  # Player 1's hole cards
                        env.players[1]["hand"],  # Player 2's hole cards
                        env.community_cards     # Community cards
                    )

                    if winner == -1:  # It's a tie
                        print("It's a tie! The pot is split.")
                        env.players[0]["chips"] += env.pot // 2
                        env.players[1]["chips"] += env.pot // 2
                        tie_count += 1
                        reward = 0
                        break
                    elif winner == 0:
                        dqn_wins += 1
                        reward += 10
                    else:
                        sr_wins += 1
                        reward -= 5

            dqn_agent.remember(state, action, reward, next_state, done)

            if env.round_over:
                env.players[winner]["chips"] += env.pot
                print(f"Player {winner + 1} wins the pot of {env.pot} chips!")

                player_chips[0] = env.players[0]["chips"]
                player_chips[1] = env.players[1]["chips"]

                # Check stopping condition after the hand
                if player_chips[0] <= 0 or player_chips[1] <= 0:
                    print(f"\nGame Over: Player {1 if player_chips[1] > 0 else 2} is out of chips!")
                    break

                print(f"Hand {hand + 1} completed.")
                print(f"Community Cards: {env.community_cards}")
                print("Final State:")
                print(f"  Player 1 Chips: {env.players[0]['chips']}")
                print(f"  Player 2 Chips: {env.players[1]['chips']}")
                print(f"DQN Agent Wins: {dqn_wins}")
                print(f"Semi-random Agent Wins: {sr_wins}")
                print(f"Ties: {tie_count}")
                break

        if len(dqn_agent.memory) >= 5:  # Replay if enough samples exist
            dqn_agent.replay(batch_size=5)

        small_blind_player = 1 - small_blind_player

    plt.plot(Game_Hand_Num, P1_Chip, label="Player 1 Chip Count", color='blue')
    plt.plot(Game_Hand_Num, P2_Chip, label="Player 2 Chip Count", color='red')
    plt.title("Chip Counts During the Game")
    plt.xlabel("Hand Number")
    plt.ylabel("Chip Count")
    plt.legend()
    plt.show()

# Run simulation
env = PokerEnvironment()
state_size = 5
action_size = 4
dqn_agent = DeepQLearningAgent(state_size, action_size)
simulate_game_with_logs(env, dqn_agent, num_hands=100000)

