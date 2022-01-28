#!/bin/python3
'''
Script used to modify console output for dqn tests.
It joins each 2 following lines and outputs to new file.
'''

wf = open('dqn_joined.csv', 'a')

with open("dqn.csv", "r") as f:
    i = 1
    joined_lines = ""
    for line in f:
        if line == "":
            continue
        if joined_lines == "":
            joined_lines += line.rstrip("\n")
        else:
            joined_lines += "," + line.rstrip("\n")
        if i == 2:
            joined_lines += '\n'
            wf.write(joined_lines)
            joined_lines = ""
            i = 0
            continue
        i += 1
        
wf.close()