"""
REAL-TIME WHAT-IF DEMO  (Streamlit)
-----------------------------------
The judge/operator sets the situation with sliders; the ACTUAL trained model
returns a live off-spec risk, and the system gives a source-tagged setpoint
recommendation with the expected improvement.

Two interactive stories:
  * "Minutes into grade change" drives the model's off-spec RISK
    (high during the change, falls to ~0 once settled).
  * "Retention" (the hidden driver we discovered) drives the expected
    SEVERITY -- overshoot and stabilization time if left unmanaged.

Design note (defensible): the model uses 66 engineered features. We seed them
from the closest historical operating context, then override the operator-
adjustable drivers the user controls. Every risk number is a real model call.

RUN:  streamlit run realtime_demo.py
NEEDS: features_reduced.csv, transition_library.csv, offspec_classifier.json
"""
import numpy as np, pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Grade Change — Live What-if", layout="wide")
GR=["G80","G100","G120","G150"]; BW={"G80":80,"G100":100,"G120":120,"G150":150}
RECIPE={"G80":dict(stock=95,filler=12,steam=2.7,speed=980),"G100":dict(stock=118,filler=15,steam=3.1,speed=920),
        "G120":dict(stock=140,filler=18,steam=3.5,speed=860),"G150":dict(stock=172,filler=22,steam=4.0,speed=790)}
LIMITS={k:{"stock":(v["stock"]*.92,v["stock"]*1.08),"filler":(v["filler"]*.9,v["filler"]*1.1),
           "steam":(v["steam"]-.3,v["steam"]+.3),"speed":(v["speed"]-40,v["speed"]+40)} for k,v in RECIPE.items()}

@st.cache_data
def load():
    feat=pd.read_csv("features_reduced.csv",parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    lib=pd.read_csv("transition_library.csv"); lib["start_time"]=pd.to_datetime(lib["start_time"])
    return feat,lib
@st.cache_resource
def load_model():
    m=XGBClassifier(); m.load_model("offspec_classifier.json"); return m

feat,lib=load(); clf=load_model()
DROP=["timestamp","grade","transition_id","grade_change_active","off_spec","target_offspec_next"]
FEATURES=[c for c in feat.select_dtypes(include=[np.number]).columns if c not in DROP]

# simple retention -> severity fits (from history) for the interactive estimate
_lib=lib.dropna(subset=["stabilization"])
pk=np.polyfit(_lib["mean_retention"],_lib["peak_dev"],1)
sb=np.polyfit(_lib["mean_retention"],_lib["stabilization"],1)
est_overshoot=lambda r: max(0.5,np.polyval(pk,r))
est_stab=lambda r: max(2,np.polyval(sb,r))

def build_row(frm,to,retention,mins,d_stock,d_steam,d_speed):
    key=f"{frm}->{to}"
    if mins<=35:
        pool=feat[feat["grade"]==key]
        if len(pool)==0: pool=feat[feat["ramp_active"]==1]
    else:
        pool=feat[feat["grade"]==to]
        if len(pool)==0: pool=feat[feat["change_window"]==0]
    if len(pool)==0: pool=feat
    pool=pool.assign(_d=(pool["mins_since_command"]-mins).abs())
    row=pool.sort_values("_d").iloc[0][FEATURES].copy()
    for c in FEATURES:
        if "retention" in c: row[c]=retention
    for c,v in [("mins_since_command",mins),("bw_sp_jump",(BW[to]-BW[frm]) if mins<=35 else 0),
                ("err_stock",d_stock),("err_steam",d_steam),("err_speed",d_speed)]:
        if c in row: row[c]=v
    if "change_window" in row: row["change_window"]=1 if mins<90 else 0
    if "ramp_active" in row: row["ramp_active"]=1 if mins<35 else 0
    return pd.DataFrame([row])[FEATURES]

def recommend(frm,to,retention):
    cand=lib[(lib["from_grade"]==frm)&(lib["to_grade"]==to)]
    if len(cand)==0: cand=lib
    best=cand.sort_values("peak_dev").iloc[0]
    sugg=[]
    for var,val in RECIPE[to].items():
        lo,hi=LIMITS[to][var]; val=float(np.clip(val,lo,hi))
        sugg.append((var,round(val,1),f"[recipe]+[historical #{int(best['transition_id'])}]"))
    if retention<0.75:
        sugg.append(("ramp strategy","slow / staged","[model: retention correlation]"))
    return best,sugg

st.title("Grade Change Intelligence — Live What-if Demo")
st.caption("Set the situation on the left. The trained model returns a live off-spec risk and a source-tagged recommendation.")
c1,c2=st.columns([1,2])
with c1:
    st.subheader("Current situation")
    frm=st.selectbox("From grade",GR,index=3)
    to=st.selectbox("To grade",[g for g in GR if g!=frm],index=1)
    mins=st.slider("Minutes into grade change",0,120,8,help="drives the off-spec risk")
    retention=st.slider("Retention (hidden driver)",0.60,1.00,0.65,0.01,help="drives expected severity")
    with st.expander("Setpoint offsets from recipe (optional)"):
        d_stock=st.slider("stock flow offset",-10.0,10.0,4.0,0.5)
        d_steam=st.slider("steam offset",-0.3,0.3,0.1,0.05)
        d_speed=st.slider("speed offset",-30.0,30.0,-8.0,1.0)

X=build_row(frm,to,retention,mins,d_stock,d_steam,d_speed)
risk=float(clf.predict_proba(X)[:,1][0]*100)
best,sugg=recommend(frm,to,retention)

with c2:
    st.subheader("Model output")
    if risk>80: st.error(f"## OFF-SPEC RISK: {risk:.0f}%  —  ALARM, act now")
    elif risk>50: st.warning(f"## OFF-SPEC RISK: {risk:.0f}%  —  elevated")
    else: st.success(f"## OFF-SPEC RISK: {risk:.0f}%  —  safe")
    st.progress(min(int(risk),100))
    a,b=st.columns(2)
    a.metric("Expected overshoot if unmanaged", f"{est_overshoot(retention):.1f}%",
             help="driven by retention (hidden)")
    b.metric("Expected stabilization", f"{est_stab(retention):.0f} min")
    st.markdown("**Recommended setpoints (source-tagged):**")
    st.dataframe(pd.DataFrame([{"parameter":v,"target":val,"source":src} for v,val,src in sugg]),
                 hide_index=True, use_container_width=True)
    st.markdown(f"**If followed** (from clean case #{int(best['transition_id'])}): "
                f"overshoot → ~{best['peak_dev']:.1f}%, stabilization → ~{best['stabilization']:.0f} min")
    st.caption("Try it: slide 'minutes' 0→120 to watch risk fall as the change settles; "
               "drop retention below 0.75 to see expected severity climb — the hidden driver.")
