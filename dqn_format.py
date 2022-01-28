wf = open('dqn_joined.txt', 'a')

with open("dqn.csv", "r") as f:
    i = 1
    joined_lines = ""
    for line in f:
        if line == "":
            continue
        joined_lines += line.rstrip("\n")
        if i == 2:
            joined_lines += '\n'
            wf.write(joined_lines)
            joined_lines = ""
            i = 0
            continue
        i += 1
        
wf.close()