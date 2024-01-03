#!/usr/bin/env python3
import bisect

import numpy as np


def calculate_tax(base, income):
    def tax(income):
        tax_points = np.array(
            [0, 36000, 144000, 300000, 420000, 660000, 960000, 10000000])
        tax_rate = np.array([0.03, 0.1, 0.2, 0.25, 0.30, 0.35, 0.45])
        income_part = tax_points[1:] - tax_points[:-1]
        idx = max(1, bisect.bisect_left(tax_points, income))
        income_part[idx:] = 0
        income_part[idx - 1] = income - tax_points[idx - 1]
        tax = tax_rate * income_part
        return tax.sum()
    return tax(base + income) - tax(base)


def main():
    incomes = []
    tax = []
    income_ntax = []

    part_sum = 0
    for income in incomes:
        tax.append(calculate_tax(part_sum, income))
        income_ntax.append(income - tax[-1])
        part_sum = part_sum + income

    print(incomes)
    print(income_ntax)
    print(tax)


if __name__ == "__main__":
    main()
