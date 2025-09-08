# T,T,F Inconsistent Triangles Analysis Results

## Summary

This table shows the count of T,T,F inconsistent triangles found in each dataset using KNN graph-based analysis.

**T,T,F Pattern**: Triangles where exactly 2 out of 3 edges are predicted as True and 1 edge is predicted as False, indicating logical inconsistency in the model's predictions.

## Results

| Dataset | Total Pairs | Total Triangles | T,T,F Triangles | T,T,F Rate (%) vs Triangles | T,T,F Rate (%) vs Pairs |
| --- | --- | --- | --- | --- | --- |
| WDC-Product | 13,034 | 30,681 | 2,089 | 6.8088 | 16.0273 |
| Persons-Leipzig | 21,179 | 65,820 | 418 | 0.6351 | 1.9737 |
| Bib-Kyoto | 23,406 | 60,356 | 295 | 0.4888 | 1.2604 |
| Walmart-Amazon | 14,024 | 24,032 | 20 | 0.0832 | 0.1426 |
| Music-Leipzig | 14,870 | 15,505 | 181 | 1.1674 | 1.2172 |
| Music-Leipzig | 22,450 | 15,505 | 304 | 1.9607 | 1.3541 |


## Notes

- **Total Pairs**: Number of record pairs evaluated by the LLM
- **Total Triangles**: Number of triangles found in the KNN graph
- **T,T,F Triangles**: Number of triangles with T,T,F inconsistent pattern
- **T,T,F Rate (%) vs Triangles**: Percentage of inconsistent triangles among all triangles
- **T,T,F Rate (%) vs Pairs**: Percentage of inconsistent triangles relative to total pairs

**Analysis Method**: KNN graph-based triangle detection for computational efficiency
