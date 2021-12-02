import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matlab.engine
from scipy import signal
import matlab
import matplotlib.cm as cm



def heatmap_snapshots(sf,df_plot,norm,cmap,cbar_label,cbar_ticks_fontsize=24,alpha_under_color=0.5):
    fig = plt.figure(figsize=(26, 26))
    grid = plt.GridSpec(24, 50, hspace=0.1, wspace=0.4)

    ax1 = fig.add_subplot(grid[:24, 0:12])
    ax2 = fig.add_subplot(grid[:24, 12:24])
    ax3 = fig.add_subplot(grid[:24, 24:36])
    ax4 = fig.add_subplot(grid[:24, 36:48])
    ax5 = fig.add_subplot(grid[10:13, 48:49])
    
    axes=[ax1,ax2,ax3,ax4,ax5]
    
    cmap.set_under('gray',alpha=alpha_under_color)


    sf_plot=sf.copy().join(df_plot)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    for col,snapshot in enumerate(df_plot.columns):
            ax=axes[col]
            chorop=sf_plot.plot(color=m.to_rgba(sf_plot[snapshot].astype('float64')),
            ax=ax,
            edgecolor='grey')
            ax.axis('off')
            #ax.text(0.45,-0.1,str(col),transform=ax.transAxes)

    cb1 = mpl.colorbar.ColorbarBase(ax5, cmap=cmap,
                                    norm=norm,
                                   # orientation='vartical'
                                   )
    cb1.set_label(cbar_label,fontsize=30)
    
    for label in (ax5.get_yticklabels()):
        label.set_fontsize(cbar_ticks_fontsize)

    
    return fig,cb1

def ZeroPad(df_signal,freq='7D'):
    data_size=len(df_signal)
    log2size=np.log2(data_size)
    next_power_of_2=np.ceil(log2size)
    pad_size=2**(next_power_of_2+1)-data_size
    #pad_size=1024-data_size
    back_pad=int(pad_size//2)
    front_pad=int(pad_size-back_pad)
    #print(front_pad)
    
    idx_signal=df_signal.index
    idx_back=pd.date_range(end=idx_signal[0],periods=back_pad+1,freq=freq,closed='left')
    df_back=pd.Series(data=np.zeros(back_pad),index=idx_back)
    idx_front=pd.date_range(start=idx_signal[-1],periods=front_pad+1,freq=freq,closed='right')
    df_front=pd.Series(data=np.zeros(front_pad),index=idx_front)
    df_signal_padded=df_back.append(df_signal)
    df_signal_padded=df_signal_padded.append(df_front)

    return df_signal_padded,back_pad,front_pad


def Signal_prep_original_method(signal_df):
    idx=signal_df.index
    signal_squared=np.sqrt(signal_df)
    signal_detrended=signal.detrend(signal_squared)                        
    signal_normalized=(signal_detrended-signal_detrended.mean())/signal_detrended.std()    
    df_signal_normalized=pd.Series(data=signal_normalized,index=signal_df.index)
    signal_detrended_zeropadded,back_pad,front_pad=ZeroPad(df_signal_normalized)
    return signal_detrended_zeropadded,back_pad,front_pad


def Matlab_CWT(signal_df):
    idx=signal_df.index
    signal_detrended,bpad,fpad=Signal_prep_original_method(signal_df)
    eng1 = matlab.engine.start_matlab()
    signal_list=matlab.double(signal_detrended.tolist())
    [wt_matlab,freqs_matlab,coi_matlab]=eng1.cwt(signal_list,'amor',nargout=3);

    wt=np.asarray(wt_matlab)
    freqs=np.asarray(freqs_matlab,dtype=float)
    coi_nonarray=np.asarray(coi_matlab,dtype=float)
    
    eng1.quit()
    
    
    
    freqs_weeks=np.zeros(len(freqs))
    for i in range(len(freqs)):
        freqs_weeks[i]=freqs[i][0]

    coi=np.zeros(len(coi_nonarray))
    for i in range(len(coi)):
        coi[i]=coi_nonarray[i][0]


    wave=wt[:,bpad:-fpad]                          
    coi=coi[bpad:-fpad] 
    coi_period=1/coi


    period_days=(1/freqs_weeks)*7
    period_years=period_days/365
    
    

    df_CWT=pd.DataFrame(data=wave,index=period_years,columns=idx)
    df_CWT.index.name='Period'
    
    var_signal=signal_detrended.iloc[bpad:-fpad].var()
    
    return df_CWT,coi,var_signal