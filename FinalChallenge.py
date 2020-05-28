#!/usr/bin/env python
# coding: utf-8

import time
import sys
import numpy as np
import csv
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql import SQLContext, Row


# Numeric codes for NYC 5 boroughs.
# 1 = Manhattan
# 2 = Bronx
# 3 = Brooklyn
# 4 = Queens
# 5 = Staten Island


# process cscl data
def process_cscl(r, record):
    if r == 0:
        next(record)
    reader = csv.reader(record)
    for row in reader:
        physical_id = int(row[0])
        full_street = format_street(row[28])
        street_label = format_street(row[10])

        borocode = int(row[13])

        # convert range fields from string to float
        for i in list(range(2, 6)):
            if row[i]:
                if row[i].isdigit():
                    row[i] = float(row[i])
                else:
                    before, row[i] = row[i].split('-')
                    row[i] = str(int(row[i]))
                    row[i] = float(before + '.' + row[i])
            else:
                row[i] = 0.0

        yield {"physical_id": physical_id, "street": street_label, "borocode": borocode, "low": row[2], "high": row[3],
               "is_left": 1}
        yield {"physical_id": physical_id, "street": street_label, "borocode": borocode, "low": row[4], "high": row[5],
               "is_left": 0}
        # col 2 and 3 are range of left, col 4 and 5 are range of right
        # if ST_LABEL is not equal to FULL_STREE, we need output twice for both of them
        # we need use the key to join vio data
        if street_label != full_street:
            yield {"physical_id": physical_id, "street": full_street, "borocode": borocode, "low": row[2],
                   "high": row[3], "is_left": 1}
            yield {"physical_id": physical_id, "street": full_street, "borocode": borocode, "low": row[4],
                   "high": row[5], "is_left": 0}


# process violation data
def process_violation(r, record):
    county2boro = {'R': 5, 'BX': 2, 'Q': 4, 'K': 3, 'NY': 1, 'BK': 3, 'ST': 5, 'QN': 4, 'MN': 1}
    if r == 0:
        next(record)
    reader = csv.reader(record)
    for row in reader:
        try:
            year = int(str(row[4][-2:]))
        except Exception as e:
            print(str(e))
            continue

        # we only need the county in NYC
        if row[21] in county2boro.keys():
            borocode = county2boro[row[21]]
        else:
            continue

        if row[23]:
            if row[23].isdigit():
                # it's an even num, then it's left
                is_left = int(row[23]) % 2
                house_n = float(row[23])
            else:
                try:
                    before, house_n = row[23].split('-')
                    house_n = str(int(house_n))
                    is_left = int(house_n) % 2
                    house_n = float(before + '.' + house_n)
                except:
                    # continue
                    house_n = -1
        else:
            continue

        street = format_street(row[24])
        yield {"year": year, "street": street, "borocode": borocode, "house_n": house_n, "is_left": is_left}


# remove use less space in word
def format_street(street):
    if street is not None:
        street = street.lower().strip()
        return street.replace(" ", "")

    return None


# column 1 in each row is represented year
def process_year(record):
    min_year = 15
    max_year = 19
    for row in record:
        record_list = [0, 0, 0, 0, 0]
        year = row[0][1]
        if min_year <= year <= max_year:
            record_list[year - min_year] = row[1]

        yield row[0][0], tuple(record_list)


# calculate ols
def compute_ols(y, x=list(range(15, 20))):
    x, y = np.array(x), np.array(y)
    n = np.size(x)
    x_m, y_m = np.mean(x), np.mean(y)
    Y = np.sum(x * y) - n * x_m * y_m
    X = np.sum(x * x) - n * x_m * x_m
    ols_coef = Y / X
    return float(str(ols_coef))


def to_csv(data):
    return ','.join(str(d) for d in data)


if __name__ == "__main__":
    start = time.time()
    output = sys.argv[1]

    sc = SparkContext()
    spark = SparkSession(sc)
    sq = SQLContext(sc)

    # for complete test in hdfs
    # rdd_cscl = sc.textFile('hdfs:///tmp/bdm/nyc_cscl.csv').mapPartitionsWithIndex(process_cscl)
    # rdd_violations = sc.textFile('hdfs:///tmp/bdm/nyc_parking_violation/').mapPartitionsWithIndex(process_violation)

    # for a local simple test
    df_cscl = sq.createDataFrame(
        sc.textFile('data/nyc_cscl.csv').mapPartitionsWithIndex(process_cscl))
    df_vio = sq.createDataFrame(
        sc.textFile('data/2016.csv').mapPartitionsWithIndex(process_violation))

    # register a temp table in spark sql
    df_cscl.registerTempTable("cscl")
    df_vio.registerTempTable("vio")

    join = " on cscl.borocode == vio.borocode and vio.is_left == cscl.is_left and (instr(vio.street, cscl.street) != 0 or instr(cscl.street, vio.street) != 0)"
    cond = " where vio.house_n >= cscl.low and vio.house_n <= cscl.high"
    group = " group by cscl.physical_id, vio.year, cscl.borocode"
    df_join = sq.sql("select cscl.physical_id, year, count(1) as count from cscl join vio " + join + cond + group)
    df_join.rdd.map(lambda x: ((x[0], x[1]), x[2])) \
        .mapPartitions(process_year) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4])) \
        .sortByKey() \
        .mapValues(lambda y: y + (compute_ols(y=list(y)),)) \
        .map(lambda x: ((x[0],) + x[1])) \
        .map(to_csv) \
        .saveAsTextFile(output)

    print('Done! Running time : {} secs'.format(time.time() - start))

