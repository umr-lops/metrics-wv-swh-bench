# metrics-wv-swh-bench

Several scientific teams are providing signficant wave height (Hs or SWH) predicted from SAR WV Sentinel-1 satellite mission.
In 2019 ESA project Climate Change Initiative (CCI) sea-state, developed a work-bench to give a score to the SWH predictions.
This python library aims at providing the methods to compute the score defined in the CCI sea-state document: https://climate.esa.int/media/documents/Sea_State_cci_PVASR_v1.1-signed.pdf 


## usage

```python
import mwsb
metrics_ndbc,metrics_cmems,total_score,inc_scores = mwsb.metrics.compute_metrics(df_wv_ndbc=matchups,ds_wv_cmems=colocated_s1_cmems)
```
