import numpy as np
from tabulate import tabulate

def trapezoidal_rule(f, h, step=1):
    num = len(f)
    
    result = 0
    for i in range(0, num - step, step):
        result += (f[i] + f[i + step])
    
    return result * step * h / 2

def runge_rule(res_trapez_1, res_trapez_2):
    return (4 * res_trapez_1 - res_trapez_2)/(3)

def simpsons_rule(f, h):
    result = f[0]
    for i in range (1, len(f) - 2, 2):
        result += 4 * f[i] + 2 * f[i + 1]
    return (result + 4 * f[-2] + f[-1]) * h / 3

def main():

    print ("Input data:")
    table = [['x', 'f(x)'], 
        ['0.00', '1.000000'], 
        ['0.15', '1.007568'], 
        ['0.30', '1.031121'],
        ['0.45', '1.073456'],
        ['0.60', '1.140228'],
        ['0.75', '1.242129'],
        ['0.90', '1.400176'],
        ['1.05', '1.660300'],
        ['1.20', '2.143460'],
        ]

    print("----------------------------------")

    len_table = len(table)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    x = []
    f = []

    for i in range (1, len_table):
        x.append(float(table[i][0]))
        f.append(float(table[i][1]))

    h = x[1] - x[0]

    print ("Let's calculate the integral with the trapezoidal rule...")
    res_trapez_1 = trapezoidal_rule(f, h)
    print("Result: " + str(res_trapez_1))
    print("----------------------------------")


    print("Let's calculate the integral with the trapezoidal rule with the double step... ")
    res_trapez_2 = trapezoidal_rule(f, h, 2)
    print("Result: " + str(res_trapez_2))
    print("----------------------------------")

    print("Let's clarify the previous integral with the runge rule... ")
    res_runge = runge_rule(res_trapez_1, res_trapez_2)
    print("Result: " + str(res_runge))
    print("----------------------------------")

    print("Let's calculate the integral with the simpson's rule... ")
    res_simpson = simpsons_rule(f, h)
    print("Result: " + str(res_simpson))
    print("----------------------------------")

    print("Conclusion:")
    print("Trapezoidal rule:", res_trapez_1)
    print("Trapezoidal rule with the double step:", res_trapez_2)
    print("Applying the runge rule:", res_runge)
    print("Simpson's rule:", res_simpson)

main()