with open('build_full_solution_v5.py', 'r') as f:
    text = f.read()

# Fix intersection call
old_line = "comm_bat_200 = bat[bat['p_balls'] >= 200]['striker_id'].intersection(comm_bat)"
new_line = "comm_bat_200 = set(bat[bat['p_balls'] >= 200]['striker_id']).intersection(set(comm_bat))"
text = text.replace(old_line, new_line)

old_line2 = "comm_bowl_200 = bowl[bowl['p_balls'] >= 200]['bowler_id'].intersection(comm_bowl)"
new_line2 = "comm_bowl_200 = set(bowl[bowl['p_balls'] >= 200]['bowler_id']).intersection(set(comm_bowl))"
text = text.replace(old_line2, new_line2)

with open('build_full_solution_v5.py', 'w') as f:
    f.write(text)

print("Fixed intersection lines in build_full_solution_v5.py")
