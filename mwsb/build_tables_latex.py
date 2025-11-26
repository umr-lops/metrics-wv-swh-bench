from pylatex import Document, Tabular, NoEscape
import pandas as pd
def create_table_per_incidence(metrics):
    

    doc = Document()

    # Add necessary packages and settings for better table rendering
    doc.packages.append(NoEscape(r"\usepackage{array}"))

    # Increase row spacing and make vertical lines more visible
    doc.append(NoEscape(r"\renewcommand{\arraystretch}{1.8}"))
    doc.append(NoEscape(r"\setlength{\arrayrulewidth}{0.5pt}"))

    # Table with vertical lines everywhere
    table = Tabular("|l|c|c|c|c|c|c|c|c|c|")

    # Title above the table
    doc.append(NoEscape(r"\textbf{SAR Round Robin CMEMS S1 wv2}"))
    doc.append(NoEscape(r"\\[6pt]"))

    # Simple header row 1 - main categories
    table.add_hline()
    table.add_row((
        "Domain (SWH, m)",
        "SWH %",
        NoEscape(r"\textbf{RMSE DLR}"),
        NoEscape(r"\textbf{RMSE IFREMER}"),
        NoEscape(r"\textbf{BIAS DLR}"),
        NoEscape(r"\textbf{BIAS IFREMER}"),
        NoEscape(r"\textbf{No-sea DLR}"),
        NoEscape(r"\textbf{No-sea IFREMER}"),
        NoEscape(r"\textbf{Outliers DLR}"),
        NoEscape(r"\textbf{Outliers IFREMER}")
    ))
    table.add_hline()

    # # Data rows
    # data = [
    #     ("0.0 - 1.5", 11.893, 0.34212, 0.35084, -0.15318, 0.19698, 5.24198, 0.13409, 0.43233, 0.50656),
    #     ("1.5 - 3.0", 61.740, 0.22708, 0.22315, -0.02274, 0.04539, 1.35065, 0.11037, 0.89351, 0.42080),
    #     ("3.0 - 6.0", 24.580, 0.33291, 0.34659,  0.02356, 0.01367, 0.18875, 0.14450, 0.89729, 0.40756),
    #     ("6.0 <",     1.786, 0.55897, 0.60751,  0.02022, 0.12333, 0.24029, 0.16234, 1.04125, 0.20325),
    #     ("TOTAL",    100.00, 0.28019, 0.28548, -0.02535, 0.01016, 1.50771, 0.12251, 0.84942, 0.47539),
    # ]

    for row in metrcs.iterrows():
        if row[0] == "TOTAL":
            # Make the TOTAL row bold
            formatted_row = [NoEscape(r"\textbf{%s}" % x) for x in row]
            table.add_row(formatted_row)
        else:
            table.add_row(row)
        table.add_hline()

    doc.append(table)
    print(doc.dumps())
    return doc.dumps()

def create_pivot_table(df, metric_filter, round_digits=3):
    """
    Create a pivot table for specific metrics
    """

    # enforce custom order
    order = ['low', 'medium', 'high', 'very high']
    df['swh_category'] = pd.Categorical(df['swh_category'], categories=order, ordered=True)


    filtered_df = df[df['metric'].isin(metric_filter)]
    
    # Create multi-level columns: dataset_wavetype as columns, swh_category as rows
    pivot = filtered_df.pivot_table(
        index=['swh_category', 'metric'], 
        columns=['reference', 'wv'], 
        values='value', 
        aggfunc='first'
    ).round(round_digits)
    
    return pivot

def create_dataframe_rows_from_dicts(metrics):
    """
    
    purpose: create a dataframe for nice display of the score
    
    """
    df = pd.DataFrame()
    cpt = 0
    for src in metrics:
        for kk in metrics[src]:
            print(src,kk,'===========================\n',)
            for jj in metrics[src][kk]:
                # print(type(metrics[kk][jj]))
                # mlksmdkf
                for pp in metrics[src][kk][jj]:
                    cpt+=1
                    df = pd.concat([df,pd.DataFrame({'reference':src,'wv':kk,'swh_category':jj,
                                                     'metric':pp,'value':metrics[src][kk][jj][pp]},index=[cpt])])
                    # try:
                    #     print('\t',kk,jj,pp,'%1.3f'%metrics[src][kk][jj][pp])
                    # except:
                    #     print('\t',kk,jj,pp,metrics[src][kk][jj][pp])
    return df

def table_compare_2_datasets(meta_metrics,metric_filter, round_digits=3):
    """
    
    Args:
        pivots (dict): with key->name of datasets and values->pivot table of score
    
    """
    # dfcat = pd.concat([meta_metrics[uu] for uu in meta_metrics],index=[uu for uu in meta_metrics])
    # dfcat['dataset'] = dfcat.index
    # Corrected code
    dfcat = pd.concat([meta_metrics[uu] for uu in meta_metrics], keys=[uu for uu in meta_metrics])
    dfcat['dataset'] = dfcat.index.get_level_values(0)
    # df = pd.DataFrame()
    # cpt = 0
    # for challenger in meta_metrics:
    #     for src in meta_metrics:
    #         for kk in meta_metrics[src]:
    #             # print(src,kk,'===========================\n',)
    #             for jj in meta_metrics[src][kk]:
    #                 # print(type(metrics[kk][jj]))
    #                 # mlksmdkf
    #                 for pp in meta_metrics[src][kk][jj]:
    #                     cpt+=1
    #                     df = pd.concat([df,pd.DataFrame({'dataset':challenger,'reference':src,'wv':kk,'swh_category':jj,
    #                                                     'metric':pp,'value':meta_metrics[src][kk][jj][pp]},index=[cpt])])
    filtered_df = dfcat[dfcat['metric'].isin(metric_filter)]
        
    # Create multi-level columns: dataset_wavetype as columns, swh_category as rows
    pivot = filtered_df.pivot_table(
        index=['swh_category', 'metric'], 
        columns=['reference', 'wv','dataset'], 
        values='value', 
        aggfunc='first'
    ).round(round_digits)
    return pivot