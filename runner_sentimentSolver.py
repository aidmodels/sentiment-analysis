#coding:utf-8
import os
import argparse
import sys
os.chdir(sys.path[0])

parser = argparse.ArgumentParser(description='CVPM Runner')
parser.add_argument('port', type=int, help='Listening on Port')

args = parser.parse_args()

from sentiment_prediction.solver import SentimentSolver

solver = SentimentSolver()

solver.start(args.port)