"""Independent small data fixtures. NOT experimental HemoBoundary results.
The historical writer follows the original saved schema using an explicit loop,
separate from hb_data.make_trials, so parity checks exercise the serialization.
"""
from pathlib import Path
import numpy as np
import h5py
import original_data as od
from hb_data import (PIDS,cache_path,pred_path,external_h5,D102,write_rows,make_trials,save_npz)


def legacy_bank(profile,path,seed,split,npos,nneg):
    ns=('RAW_D_SCREEN',path.stem,split)
    if seed!=1:ns=('MULTISEED_REPEAT',seed,*ns)
    rng=np.random.default_rng(od.stable_seed(*ns))
    trials=[od.synthetic_trial(profile,rng,True) for _ in range(npos)]
    trials += [od.synthetic_trial(profile,rng,False) for _ in range(nneg)]
    rng.shuffle(trials)
    items={k:[] for k in ('present','onset','peak','offset','censored','states','clean','obs','state_feat','ch_anti')}
    for tr in trials:
        for k in ('present','onset','peak','offset','censored','states'):items[k].append(tr[k])
        ev=od.evidence(tr['hbo'],tr['hbr'])
        items['clean'].append(tr['clean_shape']);items['obs'].append(od.whole_vector(ev));items['state_feat'].append(od.state_features(ev));items['ch_anti'].append(od.channel_anti(ev,profile.n_channels))
    return {k:np.asarray(v,dtype=np.float32 if k in ('clean','obs','state_feat','ch_anti') else int if k in ('present','censored','states') else float) for k,v in items.items()}


def prediction_rows(vals,pid,seed,dataset='SHIN'):
    rows=[]
    for i in range(36):
        r={'participant':pid,'repeat_seed':seed,'synthetic_index':i,
           **{'truth_'+k:vals[k][i] for k in ('present','onset','peak','offset','censored')},
           'pred_response':int(vals['present'][i])}
        for k in ('onset','peak','offset'):
            r['pred_'+k]=float(vals[k][i]+.1) if vals['present'][i] else np.nan
        if dataset=='SHIN':r['row']='BASE_PLUS_A_PLUS_B_PLUS_C_PLUS_LBR'
        else:r.update(dataset=dataset,model='OPTIONB_HEMOBOUNDARY')
        rows.append(r)
    return rows


def create_project(root,pids=('VP001','VP002','VP003','VP004'),seeds=(1,2),external=True):
    root=Path(root);rng=np.random.default_rng(405);times=np.arange(-5,20.001,.1)
    for pi,pid in enumerate(pids):
        c=3+(pi%2);H=rng.normal(size=(6,c,len(times)));R=.6*rng.normal(size=H.shape)
        pp=root/'02_preprocessed'/f'{pid}_WG_ALIGNED_PREPROCESSED.npz'
        save_npz(pp,nirs_times=times,hbo_causal_lp02_epochs=H,hbr_causal_lp02_epochs=R)
    for pid in pids:
        pp=root/'02_preprocessed'/f'{pid}_WG_ALIGNED_PREPROCESSED.npz';pr=od.estimate_noise_profile(pp)
        pr=od.NoiseProfile(3,pr.hbo_scale[:3],pr.hbr_scale[:3],pr.phi_hbo,pr.phi_hbr)
        for seed in seeds:
            train=legacy_bank(pr,pp,seed,'train',30,15);test=legacy_bank(pr,pp,seed,'test',24,12)
            save_npz(cache_path(root,seed,pid),**{'tr_'+k:v for k,v in train.items()},**{'te_'+k:v for k,v in test.items()})
            write_rows(pred_path(root,seed,pid),prediction_rows(test,pid,seed))
    if external:
        qc=[];saved={s:[] for s in seeds}
        for ds,ids in (('BALINT',('C01','C02')),('KHAN',('S01','S02'))):
            hp=external_h5(root,ds);hp.parent.mkdir(parents=True,exist_ok=True)
            labels=np.repeat(np.array(ids,dtype='S3'),6);H=rng.normal(size=(12,3,len(times)));R=rng.normal(size=H.shape)*.6
            with h5py.File(hp,'w') as f:
                f['times']=times;f['participant']=labels;f['hbo_epochs']=H;f['hbr_epochs']=R
            store=od.load_external_hdf5(hp)
            for pid in ids:
                profile,valid,total=od.noise_profile_from_external(store,pid)
                qc.append({'dataset':ds,'participant':pid,'usable_channels':valid,'channels_before_validity_filter':total,
                           'phi_hbo':profile.phi_hbo,'phi_hbr':profile.phi_hbr,'median_hbo_noise_scale':np.median(profile.hbo_scale),'median_hbr_noise_scale':np.median(profile.hbr_scale)})
                _,_,test=make_trials(profile,('RAW_D_SCREEN_EXTERNAL',ds,pid,'test'),24,12)
                for seed in seeds:saved[seed].extend(prediction_rows(test,pid,seed,ds))
        write_rows(root/D102/'audit/EXTERNAL_FIXED_BANK_QC.csv',qc)
        for seed in seeds:write_rows(root/D102/'checkpoints'/f'seed_{seed:02d}_PER_TRIAL.csv.gz',saved[seed])
    return root
