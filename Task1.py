#A

grid = eval(input("Enter the forest grid: "))
r,c = eval(input("Enter the center position (r, c): "))
m = int(input("Enter the window size m: "))


# Forest map
grid = [
    [1, 0, 1, 0, 0],
    [0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [0, 1, 0, 0, 0]
]

# Input
r, c, m = 2, 2, 3  # center position (r,c) and window size m



if not (0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 1):
    print("Center position is not a tree or out of grid bounds. No valid window.")
    exit()
    

n = len(grid)
p = len(grid[0])

    
# Find valid window positions
t_min = max(0, r - m + 1)
t_max = min(r, n - m)
l_min = max(0, c - m + 1)
l_max = min(c, p - m)

# Find best window
best_t, best_l = 0, 0
max_trees = 0

for t in range(t_min, t_max + 1):
    for l in range(l_min, l_max + 1):
        # Count trees
        trees = 0
        for i in range(t, t + m):
            for j in range(l, l + m):
                trees += grid[i][j]
        
        # Update best
        if trees > max_trees:
            max_trees = trees
            best_t, best_l = t, l

# Print result
print(f"Best window at: ({best_t}, {best_l})")
print(f"Total trees: {max_trees}")

# Print the window
for i in range(best_t, best_t + m):
    print(grid[i][best_l:best_l + m])
    


#B
class Player:
    def __init__(self, name: str, strength: list, weakness: list):
        self.name = name
        self.strength = strength
        self.weakness = weakness
    
def calculate_team_strength(team: list[Player]) -> int:
    combined_strength = set()
    combined_weakness = set()
    
    for player in team:
        combined_strength.update(player.strength)
        combined_weakness.update(player.weakness)
    
    # Effective strength is strengths not countered by weaknesses
    effective_strength = combined_strength - combined_weakness
    return len(effective_strength)

# Create Player objects based on the table
Virat_Kohli = Player("Virat_Kohli", ["Chase_master", "fast_bowling_destroyer", "fielding"], ["left_arm_spin"])
Rahul = Player("Rahul", ["opener", "power_play", "wicketkeeping"], ["pressure", "death_bowling"])
Bumrah = Player("Bumrah", ["death_bowling", "yorkers", "economy"], ["batting"])
Jadeja = Player("Jadeja", ["power_hitting", "off_spin", "fielding"], [])
Maxwell = Player("Maxwell", ["spin_bowling", "fielding", "finisher"], ["pace_bounce", "consistency"])
siraj = Player("siraj", ["swing_bowling", "new_ball"], ["batting"])
Shreyas = Player("Shreyas", ["middle_order", "spin_hitter"], ["express_pace", "short_ball"])
Chahal = Player("Chahal", ["leg_spin", "wicket_taker"], ["fielding", "batting", "expensive"])
DK = Player("DK", ["finisher", "wicketkeeping", "experience"], ["poor_wicketkeeping"])
Faf = Player("Faf", ["opener", "experience", "fielding"], ["slow_starter"])


all_players = [Virat_Kohli, Rahul, Bumrah, Jadeja, Maxwell, siraj, Shreyas, Chahal, DK, Faf]

k = int(input("Enter the team size k: "))

# Find all possible combinations of k players
from itertools import combinations

best_teams = []
max_score = -1

for team_combo in combinations(all_players, k):
    # Calculate team strength
    strengths = set()
    weaknesses = set()
    
    for player in team_combo:
        strengths.update(player.strength)
        weaknesses.update(player.weakness)
    
    # Net score = unique strengths - unique weaknesses
    net_score = len(strengths) - len(weaknesses)
    
    # Update best teams
    if net_score > max_score:
        max_score = net_score
        best_teams = [team_combo]
    elif net_score == max_score:
        best_teams.append(team_combo)
        
print(f"Team: {[p.name for p in team_combo]} => Net Score: {net_score} (Total Unique Strengths({len(strengths)}) - Total Unique Weaknesses({len(weaknesses)}))")

# Print results
print(f"\nBest Team(s) Found for k = {k}")
for team in best_teams:
    player_names = [p.name for p in team]
    
    # Calculate final stats for display
    strengths = set()
    weaknesses = set()
    for player in team:
        strengths.update(player.strength)
        weaknesses.update(player.weakness)
    
    print(f"{', '.join(player_names)} with net score of {max_score} (Total Unique Strengths({len(strengths)}) - Total Unique Weaknesses({len(weaknesses)}))")




