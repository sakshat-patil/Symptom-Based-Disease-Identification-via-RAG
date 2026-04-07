# Mining params used to generate `association_rules.csv`

We dropped `min_support` from 0.01 to 0.005 between Check-in 3 and 4
because the 0.01 cut left 21 of 41 diseases without any rule. The lower
threshold gives 100% disease coverage at the cost of a noisier rule set;
the overlap weighting in `MiningScorer.score` neutralises most of the
noise (see `src/mining_scorer.py`).

```
python src/mining.py --min_support 0.005 --min_confidence 0.5
```

Output (after the change):

- 23,839 rules across all 41 disease classes
- Median confidence: 1.000
- Mean lift: 40.4
