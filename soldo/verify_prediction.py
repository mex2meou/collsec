import pandas as pd

# verify the prediction against last day logs
def verify_prediction(l_blacklist, ca_blacklist, knn_ca_blacklist, last_day_logs, window):
    
    w_stats = []
    
    # for every victim in the blacklist
    for victim in l_blacklist:
        
        stats = {}
                
        # get the blacklist and whitelist set
        l_bl_set = set([x for x in l_blacklist[victim] if l_blacklist[victim][x] == 1])
        l_wl_set = set([y for y in l_blacklist[victim] if l_blacklist[victim][y] == 0])
        ca_bl_set = set([x for x in ca_blacklist[victim] if ca_blacklist[victim][x] == 1])
        ca_wl_set = set([y for y in ca_blacklist[victim] if ca_blacklist[victim][y] == 0])
        knn_ca_bl_set = set([x for x in knn_ca_blacklist[victim] if knn_ca_blacklist[victim][x] == 1])
        knn_ca_wl_set = set([y for y in knn_ca_blacklist[victim] if knn_ca_blacklist[victim][y] == 0])
                
        # keep only the victim's logs
        c = last_day_logs[last_day_logs['target_ip'] == victim]
        
        # find the number of attackers that attacked that victim
        attackers = c['src_ip'].unique()
        
        # get last day's set
        ground_truth = set(attackers)
        
        stats["victim"] = victim
        stats["D"] = window + 22

        stats["tp_local"] = len(l_bl_set & ground_truth)
        stats["fp_local"] = len(l_bl_set - ground_truth)
        stats["fn_local"] = len(l_wl_set & ground_truth)
        stats["tn_local"] = len(l_wl_set - ground_truth)
        
        # calculate stats of ts + knn
        stats["tp_ca"] = len(ca_bl_set & ground_truth)
        stats["fp_ca"] = len(ca_bl_set - ground_truth)
        stats["fn_ca"] = len(ca_wl_set & ground_truth)
        stats["tn_ca"] = len(ca_wl_set - ground_truth)
        
        # calculate the stats of ts + knn + ca
        stats["tp_knn_ca"] = len(knn_ca_bl_set & ground_truth)
        stats["fp_knn_ca"] = len(knn_ca_bl_set - ground_truth)
        stats["fn_knn_ca"] = len(knn_ca_wl_set & ground_truth)
        stats["tn_knn_ca"] = len(knn_ca_wl_set - ground_truth)
        
        # remove the data 
        del c; del l_bl_set; del l_wl_set; del ca_bl_set; del ca_wl_set; del knn_ca_bl_set; del knn_ca_wl_set;
        del ground_truth; del attackers;        
        
        w_stats.append(stats)
            
    return pd.DataFrame(w_stats)