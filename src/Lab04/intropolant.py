import numpy as np
from matplotlib import pyplot as plt
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from tabulate import tabulate

# 1910 – 92228496 человек,
# 1920 – 106021537,
# 1930 – 123202624,
# 1940 – 132164569,
# 1950 – 151325798,
# 1960 – 179323175,
# 1970 – 203211926,
# 1980 – 226545805,
# 1990 – 248709873,
# 2000 – 281421906.

class ICallable (ABC):

    @abstractmethod
    def __call__(self):
        pass

class GraphCreator(ICallable):

    graph = 0
    def __init__(self, x_size, y_size, name = ""):
        self.graph = plt.figure(figsize=[x_size, y_size])
        plt.title(name)

    def set_xyscale (self, x, y):
        ax = self.graph.add_subplot(111)
        ax.set_xscale(x)
        ax.set_yscale(y)

    def add_graph(self, x_arr, y_arr, x_name = "x", y_name = "y", color = "red", legend_name = ''):
        plt.plot (x_arr, y_arr, color, label = legend_name)

        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.grid()
    
    def __call__(self):
        plt.show()

def q (arg, arg_0, h):
    return (arg - arg_0) / h 

class Spline:
    a = 0
    b = 0
    c = 0
    d = 0
    x = 0
    def __init__(self, year, pop):
        self.pop = int(pop)
        self.b = 0
        self.c = 0
        self.d = 0
        self.year = int(year)

def main ():

    true_value = 308745538
    print ("Input data:")
    table = [['Year', 'Population'], 
        ['1910', '92228496'], 
        ['1920', '106021537'], 
        ['1930', '123202624'],
        ['1940', '132164569'],
        ['1950', '151325798'],
        ['1960', '179323175'],
        ['1970', '203211926'],
        ['1980', '226545805'],
        ['1990', '248709873'],
        ['2000', '281421906'],
        ]
    len_table = len(table)
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    year_gap = 10
    year = 2010
    print ("Let's extrapolate the function to " + str(year) + " with Newton interpolation:")
    print('-------------------------\n')
    print ("Year gap: " + str(year_gap))
    
    print('-------------------------\n')
    
    years = []
    values = []
    
    print ('Calculate divided difference:')

    for i in range (1, len_table):
        years.append(float(table[i][0]))
        values.append(int(table[i][1]))
    print(years)
    year_0 = years[0]

    div_diffs_arr = np.empty ([len(years), len(years)], dtype = int)
    div_diffs_arr[0] = values

    for i in range (1, len(div_diffs_arr[0])):
        for j in range (0, len(div_diffs_arr[0])):
            div_diffs_arr[i][j] = 0

    for i in range (1, len(div_diffs_arr[0])):
        for j in range (0, len(div_diffs_arr[0]) - i):
            div_diffs_arr[i][j] = (div_diffs_arr[i - 1][j + 1] - div_diffs_arr[i - 1][j]) / i

    print (div_diffs_arr)
    print('-------------------------\n')


    res = div_diffs_arr[0][0]

    print('Calculate Newton\'s interpolant: \n')

    for i in range (1, len(years)):
        
        j = 0
        next = div_diffs_arr[i][0]
        while (j < i):
            next *= (q(year, year_0, year_gap) - j)
            j += 1
        res += next

    print('-------------------------\n')
    
     
    print ('Calculated value: ' + str(res))
    print ('True value: ' + str(true_value))

    print('-------------------------\n')        

    print ("Let's extrapolate the function to " + str(year) + " with Spline interpolation:")
    
    splines = []
    for i in range (1, len_table):
        splines.append(Spline(table[i][0], table[i][1]))
    
    x_gap_arr = []
    y_gap_arr = []
    a_coefs = [0]
    b_coefs = [0]

    print ("Building Spline method...")
    print ("Calculting splines coefficients...")
    for i in range (1, len_table - 1):
        x_gap_arr.append(int(table[i + 1][0]) - int(table[i][0]))
        y_gap_arr.append(int(table[i + 1][1]) - int(table[i][1]))

    for i in range (1, len_table - 2):
        x_gap = x_gap_arr[i]
        prev_x_gap = x_gap_arr[i - 1]
        tmp = prev_x_gap * (a_coefs[i - 1] + 2) + 2 * x_gap
        a_coefs.append(-x_gap / tmp)
        b_coefs.append((6 * (y_gap_arr[i]/x_gap - y_gap_arr[i - 1]/prev_x_gap) - prev_x_gap * b_coefs[i - 1]) / tmp)

    for i in range (len_table - 3, 0, -1):
        splines[i].c = a_coefs[i] * splines[i + 1].c + b_coefs[i]
    
    for i in range (len_table - 2, 0, -1):
        cur_c_coeff = splines[i].c
        prev_c_coeff = splines[i - 1].c
        prev_x_gap = x_gap_arr[i - 1]
        splines[i].d = (cur_c_coeff - prev_c_coeff) / prev_x_gap
        splines[i].b = prev_x_gap * (2 * cur_c_coeff + prev_c_coeff) / 6 + y_gap_arr[i - 1] / prev_x_gap

    i = 1
    for elem in splines:
        print ('Spline #' + str(i))
        print ('\tpopulation: ' + str(elem.pop))
        print ('\tb: ' + str(elem.b))
        print ('\tc: ' + str(elem.c))
        print ('\td: ' + str(elem.d))
        print ('\tyear: ' + str(elem.year))
        i += 1

    print('-------------------------\n')

    print('Calculating value with gotten spline-functions...')
    num_splines = len(splines)
    pivot = splines[0]

    if year <= splines[0].year:
        pivot = splines[0]
    elif year >= splines[num_splines - 1].year:
        pivot = splines[num_splines - 1]
    else:
        i = 0
        j = num_splines - 1
        while i + 1 < j:
            k = i + (j - i) // 2
            if year <= splines[k].year:
                j = k
            else:
                i = k
        pivot = splines[j]

    diff_x = year - pivot.year
    res = pivot.pop + pivot.b * diff_x + pivot.c / 2 * diff_x ** 2 + 1 / 6 * pivot.d * diff_x ** 3
    print ('Calculated value: ' + str(res))
    print ('True value: ' + str(true_value))

    print('-------------------------\n')      

main()