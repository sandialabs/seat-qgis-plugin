Probabilities (optional)
------------------------

This section explains the optional probability file, which provides information about the run order and the probability of each model condition occurring. The probability file is not required but can enhance the analysis if provided. It follows a specific naming convention (see the device/baseline model section) and corresponds to the return interval in years.

If you choose to include a probability file, it must contain the following columns:

- `run order`: The run order of each condition.
- `% of year``: The probability of the condition occurring within a year.
- Other columns, such as `wave height` and `period`, can be included but will be ignored.

You also have the option to include an 'Exclude' column, which allows you to specify runs that should be excluded from the analysis.

.. note::

   Currently `% of year` is not rescaled to 100 if Exclude is included.


.. figure:: ../../media/probabilities_input.webp
   :scale: 100 %
   :alt: Probablities Input

**Probabilities Input Example**

Here's an example of the probability input file:

+------+--------+--------+-------------+---------+-----------+---------+
| Hs[m]| Tp[s]  | Dp[deg]| % of dir bin| % of yr | run order | Exclude |
+------+--------+--------+-------------+---------+-----------+---------+
| 1.76 |   6.6  | 221.8  |    15.41    |   0.39  |    6      |         |
+------+--------+--------+-------------+---------+-----------+---------+
| 2.67 |   8.62 | 220.8  |    40.68    |   1.029 |   16      |         |
+------+--------+--------+-------------+---------+-----------+---------+
| 4.06 |  10.16 | 221.3  |    23.47    |   0.593 |   20      |         |
+------+--------+--------+-------------+---------+-----------+---------+
| 1.37 |  15.33 | 224    |    8.06     |   0.204 |    2      |         |
+------+--------+--------+-------------+---------+-----------+---------+
| 7.05 |  12.6  | 223.6  |    3.42     |   0.086 |   24      |    x    |
+------+--------+--------+-------------+---------+-----------+---------+

In this table, 'Hs' represents wave height, 'Tp' represents wave period, 'Dp' represents direction, '% of dir bin' is the percentage of direction bin, '% of yr' is the percentage of the year, 'run order' is the order of the run, and 'Exclude' is used to mark runs for exclusion.
