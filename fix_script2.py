with open('build_full_solution_v5.py', 'r') as f:
    text = f.read()

text = text.replace("bat_odd.loc[comm_bat_200]", "bat_odd.loc[list(comm_bat_200)]")
text = text.replace("bat_even.loc[comm_bat_200]", "bat_even.loc[list(comm_bat_200)]")
text = text.replace("bowl_odd.loc[comm_bowl_200]", "bowl_odd.loc[list(comm_bowl_200)]")
text = text.replace("bowl_even.loc[comm_bowl_200]", "bowl_even.loc[list(comm_bowl_200)]")

with open('build_full_solution_v5.py', 'w') as f:
    f.write(text)

print("Fixed indexing with list()")
