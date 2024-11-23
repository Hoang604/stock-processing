from visualization import plot_all_bank, plot_all_normal_company
from data_acquisition_and_processing import get_ticker_icb_code
tickers = ['TCB']

for ticker in tickers:
    try:
        if get_ticker_icb_code(ticker) == get_ticker_icb_code('TCB'):
            plot_all_bank(ticker)
        else:
            plot_all_normal_company(ticker)
    except Exception:
        continue

# ['ACB', 'DHG', 'DNW', 'DC1', 'BDW', 'TS3', 'BWE', 'BGW', 'BID', 'GCF', 'FCS', 'FMC', 'ANT', 'FPT', 'FOX', 'BVB', 'DSE', 'GMA', 'DCF', 'CLC', 'HHV', 'BMJ', 'BMP', 'GHC', 'AGP', 'DNC', 'DHD', 'ABB', 'HGT', 'ASM', 'CMF', 'DTT', 'CTG', 'DWS', 'DWC', 'CCC', 'ABW', 'DNE', 'CTI', 'DNP', 'DBD', 'CDN', 'GEX', 'CCL', 'CTR', 'BVH', 'PTB', 'PHN', 'DTP', 'ACE', 'TSJ', 'HDG', 'HTI', 'CPH', 'IDP', 'GEG', 'GMD', 'IN4', 'INN', 'DHC', 'KLB', 'IMP', 'CRC', 'LIX', 'LPB', 'MBB', 'BAB', 'NAS', 'VCE', 'HDB', 'LDW', 'NSC', 'OCB', 'MCH', 'MSB', 'PDN', 'PBC', 'PC1', 'HC1', 'PNJ', 'PPP', 'PTX', 'PTT', 'PVT', 'SAF', 'SAC', 'SBT', 'LHC', 'SCS', 'PHS', 'SSB', 'NAB', 'SZC', 'THN', 'TIG', 'NBW', 'OPC', 'SHB', 'SJ1', 'TVS', 'SGS', 'SIP', 'STG', 'SNZ', 'STB', 'SSI', 'SHI', 'SWC', 'TCB', 'TDM', 'TDP', 'THG', 'THW', 'MBS', 'TMW', 'TNG', 'TOT', 'TPB', 'TPP', 'TRA', 'VAB', 'VNC', 'VCS', 'VCB', 'VIB', 'VNM', 'VGR', 'VMA', 'VND', 'VBB', 'VPB', 'VLW', 'VSC', 'VTQ', 'DFC']