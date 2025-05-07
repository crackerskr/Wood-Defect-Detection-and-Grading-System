from crack import hasCrack
from deadknot import hasDeadKnot
from hole import getNumHoles
from smallknot import hasSmallKnot
from undersized import isUndersized

def checkQuality(path):
    if isUndersized(path) == True:
        print("The wood need to be resized")
    elif hasDeadKnot(path) == True:
        print("The wood is graded has dead knot")
    elif hasCrack(path) == True:
        print("The wood is graded has crack")
    elif getNumHoles(path) <3:
        if hasSmallKnot(path) == True:
            print("The wood is graded has small knot")
        else:
            print("The wood is graded as Grade A")
    elif getNumHoles(path) < 10:
        print("The wood is graded holes")
    elif getNumHoles(path) >= 10:
        print("The wood is graded has many holes")

print("Welcome to PT Wood Industry Wood Surface Quality Check")
print("Please input the picture path")
path = input()
checkQuality(path)