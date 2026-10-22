// QUEST AI TRADER v2.2 LAB · 5x — Cloudflare Worker — SIMULATION ONLY
// Isolated from v2.1 and v2.2 1x: this Worker ONLY reads/writes the KV key quest_lab_v22_5x_state_01.
// Synthetic 5x exposure; NO real exchange orders, margin borrowing, credentials or guarantees.
const C = Object.freeze({capital:1000, stake:50, fee:0.004, slip:0.001,
  leverage:5, longThreshold:.58, shortThreshold:.42, stop:.03, target:.06,
  maintenance:.005,
  minHoldBars:4, maxHoldBars:48, maxDayLoss:.025, maxDrawdown:.08,
  hour:3600, minValidation:24});
const STATE_KEY='quest_lab_v22_5x_state_01';
const cap=(n,lo,hi)=>Math.max(lo,Math.min(hi,n));
const avg=a=>a.length?a.reduce((s,n)=>s+n,0)/a.length:0;
const sign=side=>side==='LONG'?1:-1;
const iso=ms=>new Date(ms).toISOString();
const round=(n,nDigits=4)=>Number(n.toFixed(nDigits));
function fresh(){return {v:22, running:false, cash:C.capital, position:null, initial:C.capital,
  lastBar:0,lastRun:null,lastPrice:null,signal:null,decision:null,decisions:[],trades:[],
  series:[],validation:null,comparison:null,source:null,day:'',dayEquity:C.capital,
  peak:C.capital,lastError:null,warning:'Separates 5x-Paper-Labor. Margin kann vollständig verloren gehen. Keine Echtgeld-Orders.'};}
function parseBars(raw,now){
  if(!Array.isArray(raw))throw Error('Kursantwort ist kein Array');
  const bars=raw.filter(r=>Array.isArray(r)&&r.length>=6).map(r=>({t:+r[0],l:+r[1],h:+r[2],o:+r[3],c:+r[4],v:+r[5]}))
    .filter(x=>[x.t,x.l,x.h,x.o,x.c,x.v].every(Number.isFinite)&&x.t>0&&x.l>0&&x.h>=x.l&&x.o>0&&x.c>0&&x.v>=0
      &&x.t*1000+C.hour*1000<=now-30000).sort((a,b)=>a.t-b.t);
  const unique=[];for(const b of bars)if(!unique.length||b.t>unique.at(-1).t)unique.push(b);
  if(unique.length<150)throw Error('Zu wenige abgeschlossene Stundenkerzen: '+unique.length);
  if(now/1000-(unique.at(-1).t+C.hour)>3*C.hour)throw Error('Letzte Stundenkerze zu alt; keine Order');
  return unique.slice(-300);
}
function feat(bs,i){if(i<25)return null;
  const p=bs[i].c,r=n=>cap(100*Math.log(p/bs[i-n].c),-20,20);
  const ma=avg(bs.slice(i-23,i+1).map(b=>b.c));
  const ret=[];for(let j=i-11;j<=i;j++)ret.push(100*Math.log(bs[j].c/bs[j-1].c));
  const m=avg(ret),sd=Math.sqrt(avg(ret.map(z=>(z-m)**2)));
  const v=avg(bs.slice(i-23,i).map(b=>b.v));
  return [1,cap(r(1)/2,-3,3),cap(r(3)/4,-3,3),cap(r(8)/6,-3,3),
    cap((p/ma-1)*20,-3,3),cap(sd/3,0,3),cap(Math.log((bs[i].v+1)/(v+1)),-3,3),
    cap(100*(bs[i].c-bs[i].o)/bs[i].o/3,-3,3)];
}
const sigmoid=z=>1/(1+Math.exp(-cap(z,-30,30)));
const prob=(w,x)=>sigmoid(x.reduce((s,n,i)=>s+n*w[i],0));
function train(rows,passes=35){
  const w=Array(rows[0].x.length).fill(0),b=cap(avg(rows.map(r=>r.y)),.05,.95);
  w[0]=Math.log(b/(1-b));
  for(let e=0;e<passes;e++){
    const g=w.map(()=>0);
    for(const row of rows){const d=prob(w,row.x)-row.y;row.x.forEach((x,k)=>g[k]+=d*x);}
    w.forEach((weight,k)=>w[k]-=.095*(g[k]/rows.length+(k===0?0:.007*weight)));
  }return w;
}
function score(rows,w){let loss=0,correct=0;for(const r of rows){const p=cap(prob(w,r.x),.00001,.99999);
  loss-=r.y*Math.log(p)+(1-r.y)*Math.log(1-p);correct+=((p>=.5?1:0)===r.y?1:0);}
  return {n:rows.length,loss:rows.length?loss/rows.length:null,accuracy:rows.length?correct/rows.length:null};}
function rowsOf(bars){const rows=[];for(let i=25;i<bars.length-3;i++){
  if(bars[i+3].t-bars[i].t!==3*C.hour)continue;
  rows.push({i,x:feat(bars,i),y:Math.log(bars[i+3].c/bars[i].c)>0?1:0});
}return rows;}
function signal(w,bs,i,enabled=true){const x=feat(bs,i),p=prob(w,x);
  const trend=x[4]>0?1:x[4]<0?-1:0;
  const label=!enabled?'GESPERRT':p>=C.longThreshold&&trend>0?'LONG':p<=C.shortThreshold&&trend<0?'SHORT':'WARTEN';
  return {label,p,trend,t:bs[i].t,c:bs[i].c};}
// Synthetic isolated-margin accounting; C.stake is maximum collateral per trade.
// Closing slippage and fees are reflected in mark-to-market equity.
function markClosePrice(p,market){return market*(p.side==='LONG'?1-C.slip:1+C.slip);}
function markNet(p,market){const px=markClosePrice(p,market);
  return sign(p.side)*p.qty*(px-p.entry)-p.qty*px*C.fee;}
function equity(s,price){if(!s.position)return s.cash;
  const p=s.position;return s.cash+Math.max(0,p.margin+markNet(p,price));}
// Approximate isolated-margin liquidation when remaining equity reaches 0.5%
// of notional. Full collateral is conservatively forfeited on liquidation.
// Actual exchange liquidation differs by instrument, fees and funding.
function liquidationPrice(p){const q=p.qty,part=p.side==='LONG'
  ? (1-C.slip)*(1-C.fee)-C.maintenance
  : (1+C.slip)*(1+C.fee)+C.maintenance;
  return p.side==='LONG'?(q*p.entry-p.margin)/(q*part)
    :(q*p.entry+p.margin)/(q*part);}
function open(s,side,price,time){if(s.position||s.cash<C.stake*(1+C.fee*C.leverage))return false;
  const entry=price*(side==='LONG'?1+C.slip:1-C.slip);
  const qty=C.stake*C.leverage/entry,entryFee=C.stake*C.leverage*C.fee;
  s.cash-=C.stake+entryFee;
  s.position={side,margin:C.stake,qty,entry,at:time,openedBar:Math.floor(time/C.hour),entryFee,
    notional:C.stake*C.leverage,leverage:C.leverage};
  s.trades.unshift({action:'OPEN_'+side,time:iso(time*1000),price:round(entry,2),eur:C.stake,
    notional:round(C.stake*C.leverage,2),leverage:C.leverage,fee:round(entryFee,4),pnl:null});
  s.trades=s.trades.slice(0,100);return true;}
function close(s,price,time,reason,{liquidated=false}={}){if(!s.position)return false;
  const p=s.position,exit=markClosePrice(p,price),gross=sign(p.side)*p.qty*(exit-p.entry),exitFee=p.qty*exit*C.fee;
  const realized=liquidated?-p.margin:Math.max(-p.margin,gross-exitFee);
  const released=liquidated?0:Math.max(0,p.margin+realized);
  s.cash+=released;
  s.trades.unshift({action:(liquidated?'LIQUIDATED_':'CLOSE_')+p.side,time:iso(time*1000),price:round(exit,2),eur:p.margin,
    notional:round(p.notional||p.margin*C.leverage,2),leverage:C.leverage,
    fee:round(exitFee,4),pnl:round(realized-p.entryFee,4),reason});
  s.trades=s.trades.slice(0,100);s.position=null;return true;}
function riskExit(p,bar){
  const st=p.entry*(1-C.stop*sign(p.side)),tp=p.entry*(1+C.target*sign(p.side));
  const liq=liquidationPrice(p);
  // Opening gaps have priority: if price crosses liquidation before the
  // hourly execution check, treat it as a full-margin loss.
  if(p.side==='LONG'){
    if(bar.o<=liq)return {price:bar.o,reason:'Liquidation durch Kurslücke (simuliert)',liquidated:true};
    if(bar.o<=st)return {price:bar.o,reason:'Stop-Loss durch Kurslücke'};
    if(bar.l<=st)return {price:st,reason:'Stop-Loss'};
    if(bar.l<=liq)return {price:liq,reason:'Liquidation (simuliert)',liquidated:true};
    if(bar.h>=tp)return {price:tp,reason:'Take-Profit'};
  }else{
    if(bar.o>=liq)return {price:bar.o,reason:'Liquidation durch Kurslücke (simuliert)',liquidated:true};
    if(bar.o>=st)return {price:bar.o,reason:'Stop-Loss durch Kurslücke'};
    if(bar.h>=st)return {price:st,reason:'Stop-Loss'};
    if(bar.h>=liq)return {price:liq,reason:'Liquidation (simuliert)',liquidated:true};
    if(bar.l<=tp)return {price:tp,reason:'Take-Profit'};
  }return null;
}
// Signals come only from previous CLOSED candles; fill at NEXT candle open in backtest.
function backtest(bars,testRows,w,allowShort){
  const s={cash:C.capital,position:null,trades:[]};let countOpen=0,countClose=0,wins=0,peak=C.capital,dd=0;
  for(const row of testRows){const next=bars[row.i+1];if(!next||next.t-bars[row.i].t!==C.hour)continue;
    const sig=signal(w,bars,row.i),held=s.position?Math.max(0,Math.round((next.t/ C.hour)-s.position.openedBar)):0;
    let exited=false;
    if(s.position&&(held>=C.maxHoldBars||(held>=C.minHoldBars&&sig.label!==s.position.side&&sig.label!=='WARTEN'))){
      const old=s.trades[0];close(s,next.o,next.t,'Signal oder Maximaldauer');countClose++;wins+=s.trades[0].pnl>0?1:0;exited=true;
    }
    if(!s.position&&!exited&&(sig.label==='LONG'||(allowShort&&sig.label==='SHORT'))){
      if(open(s,sig.label,next.o,next.t))countOpen++;
    }
    if(s.position){const r=riskExit(s.position,next);if(r){close(s,r.price,next.t,r.reason,{liquidated:!!r.liquidated});countClose++;wins+=s.trades[0].pnl>0?1:0;}}
    const eq=equity(s,next.c);peak=Math.max(peak,eq);dd=Math.max(dd,(peak-eq)/Math.max(1,peak));
  }
  const end=equity(s,bars.at(-1).c);
  return {returnPct:round((end/C.capital-1)*100,2),maxDrawdownPct:round(dd*100,2),opened:countOpen,
    closed:countClose,wins,unclosed:s.position?1:0};
}
function research(bars){const rows=rowsOf(bars);if(rows.length<120)throw Error('Zu wenige Trainingsbeispiele');
  const n=rows.length,trEnd=Math.floor(n*.60),vaStart=trEnd+3,vaEnd=Math.floor(n*.80),teStart=vaEnd+3;
  const trainRows=rows.slice(0,trEnd),validation=rows.slice(vaStart,vaEnd),testRows=rows.slice(teStart);
  if(validation.length<C.minValidation||testRows.length<20)throw Error('Unzureichende Validierungs-/Testdaten');
  const wTrain=train(trainRows);const val=score(validation,wTrain),test=score(testRows,wTrain);
  const base=cap(avg(trainRows.map(r=>r.y)),.01,.99);
  const baseWeights=[Math.log(base/(1-base)),...Array(wTrain.length-1).fill(0)];
  const baseLoss=score(validation,baseWeights).loss;
  const approved=val.loss<baseLoss-.002;
  const longOnly=backtest(bars,testRows,wTrain,false),both=backtest(bars,testRows,wTrain,true);
  // Refit on all LABELLED samples ONLY after holdout scoring for the live forecast.
  const liveWeights=train(rows,24);const current=signal(liveWeights,bars,bars.length-1,approved);
  return {signal:current,validation:{train:trainRows.length,val:validation.length,test:testRows.length,
     validationLoss:round(val.loss,4),baselineLoss:round(baseLoss,4),testAccuracyPct:round(test.accuracy*100,1),
     approved,disclaimer:'Kurzer, zeitlich getrennter Test. Keine Gewinnprognose.'},
     comparison:{longOnly,both,feesPctEachSide:C.fee*100,slippagePctEachSide:C.slip*100,
       nTestCandles:testRows.length},};
}
function record(s,now,code,title,detail,action='SKIP'){
  s.decision={at:iso(now),bar:s.lastBar,code,title,detail,action};
  s.decisions=[s.decision,...s.decisions].slice(0,20);
}
function apply(s,bars,info,now){
  const bar=bars.at(-1),t=bar.t;s.lastRun=iso(now);s.lastPrice=bar.c;s.signal=info.signal;
  s.validation=info.validation;s.comparison=info.comparison;s.lastError=null;
  if(t<=s.lastBar){record(s,now,'NO_NEW_BAR','Keine neue Stundenkerze','Signal aktualisiert; keine doppelte Order.');return s;}
  s.lastBar=t;const day=iso(now).slice(0,10);
  if(s.day!==day){s.day=day;s.dayEquity=equity(s,bar.c);}
  const eq=equity(s,bar.c);s.peak=Math.max(s.peak,eq);
  const dayLoss=(s.dayEquity-eq)/Math.max(1,s.dayEquity),drawdown=(s.peak-eq)/Math.max(1,s.peak);
  let code='WAIT',title='Kein Einstieg',detail='Kein hinreichendes Long- oder Short-Signal.',action='SKIP';
  const positionRisk=s.position&&t>s.position.at?riskExit(s.position,bar):null;
  // Live orders are simulated at the LAST COMPLETED candle close (not a tradable quote).
  // The historical comparison fills NEXT candle open; costs are applied to both.
  if(positionRisk){
    close(s,positionRisk.price,t,positionRisk.reason,{liquidated:!!positionRisk.liquidated});
    code=positionRisk.liquidated?'LIQUIDATION':'STOP_OR_TAKE';title='Position geschlossen';
    detail=positionRisk.reason;action='CLOSE';
    if(positionRisk.liquidated)s.running=false;
  }else if(!s.running){code='PAUSED';title='Simulation pausiert';detail='Markt analysiert, aber keine neuen Orders erlaubt.';}
  else if(dayLoss>=C.maxDayLoss||drawdown>=C.maxDrawdown){
    const was=!!s.position;if(was)close(s,bar.c,t,'Not-Aus');s.running=false;
    code='RISK_PAUSE';title='Verlustgrenze erreicht – pausiert';detail='Tagesverlust oder Rückgang hat die Begrenzung erreicht.';
    action=was?'CLOSE':'SKIP';
  }else if(s.position){
    const p=s.position,held=Math.max(0,Math.floor((t-p.at)/C.hour));
    // The position's first candle has already been processed at entry.
    if(!info.validation.approved){close(s,bar.c,t,'Modell-Validierung fehlgeschlagen');
      code='MODEL_EXIT';title='Position sicherheitshalber geschlossen';detail='Modell hat die neue Validierung nicht bestanden.';action='CLOSE';}
    else if(held>=C.maxHoldBars||(held>=C.minHoldBars&&s.signal.label!==p.side&&s.signal.label!=='WARTEN')){
      close(s,bar.c,t,'Gegensignal oder Zeitlimit');code='SIGNAL_EXIT';title='Position geschlossen';
      detail='Gegensignal oder maximale Haltedauer erreicht.';action='CLOSE';
    }else{code='HOLD';title=p.side+' wird gehalten';detail='Position offen, kein gültiges Ausstiegssignal.';}
  }else if(!info.validation.approved){code='MODEL_BLOCKED';title='Modell nicht freigegeben';detail='Validierung schlägt Basisprognose nicht ausreichend. Kein Echtgeld und keine neuen Paper-Orders.';}
  else if(s.signal.label==='LONG'||s.signal.label==='SHORT'){
    if(open(s,s.signal.label,bar.c,t)){code='OPEN_'+s.signal.label;title=s.signal.label+' simuliert eröffnet';
      detail='5x: 50 € Margin, 250 € Marktposition. Gebühren und Slippage berücksichtigt.';action='OPEN';}
    else{code='NO_CASH';title='Guthaben reicht nicht';detail='Zu wenig freies virtuelles Guthaben.';}
  }else{code='NO_SIGNAL';title='Auf Long/Short warten';detail='Modellwert '+round(s.signal.p*100,1)+' %, Trendfilter: '+(s.signal.trend>0?'positiv':s.signal.trend<0?'negativ':'neutral')+'.';}
  const post=equity(s,bar.c);
  if(s.running&&((s.dayEquity-post)/Math.max(1,s.dayEquity)>=C.maxDayLoss
    ||(s.peak-post)/Math.max(1,s.peak)>=C.maxDrawdown)){
    s.running=false;
    detail+=' Verlustgrenze erreicht: Simulation pausiert.';
    title=title+' · PAUSE';
  }
  record(s,now,code,title,detail,action);
  const end=equity(s,bar.c);s.series=[...s.series,{t,e:round(end,3)}].slice(-200);
  return s;
}
async function fetchBars(){
  let first='';
  try{const r=await fetch('https://api.exchange.coinbase.com/products/BTC-EUR/candles?granularity=3600',
    {headers:{Accept:'application/json','User-Agent':'QuestPaperLab/2.2'},cf:{cacheTtl:0}});
    if(!r.ok)throw Error('HTTP '+r.status);
    return {bars:parseBars(await r.json(),Date.now()),provider:'Coinbase Exchange'};
  }catch(e){first=String(e.message||e);}
  try{const r=await fetch('https://api.kraken.com/0/public/OHLC?pair=XBTEUR&interval=60',
    {headers:{Accept:'application/json'},cf:{cacheTtl:0}});
    if(!r.ok)throw Error('HTTP '+r.status);
    const j=await r.json();if(j.error?.length)throw Error(j.error.join(','));
    const key=Object.keys(j.result||{}).find(x=>x!=='last');if(!key)throw Error('Keine OHLC-Daten');
    return {bars:parseBars(j.result[key].map(x=>[x[0],x[3],x[2],x[1],x[4],x[6]]),Date.now()),provider:'Kraken'};
  }catch(e){throw Error('Daten nicht verfügbar. Coinbase: '+first+'; Kraken: '+String(e.message||e));}
}
async function read(env){const s=await env.TRADER_KV.get(STATE_KEY,'json');return s&&s.v===22?s:fresh();}
async function save(env,s){await env.TRADER_KV.put(STATE_KEY,JSON.stringify(s));}
async function tick(env){const s=await read(env),now=Date.now();
  try{const {bars,provider}=await fetchBars();s.source=provider;
    if(bars.at(-1).t<=s.lastBar){s.lastRun=iso(now);s.lastError=null;return await save(env,s),s;}
    const researchResult=research(bars);apply(s,bars,researchResult,now);await save(env,s);return s;
  }catch(err){s.lastRun=iso(now);s.lastError=String(err?.message||err).slice(0,340);
    // Never alter positions or last processed candle on incomplete/failed data.
    await save(env,s);return s;
  }
}
function publicState(s){const total=equity(s,s.lastPrice||s.position?.entry||0);
  return {...s,equity:round(total,2),returnPct:round((total/C.capital-1)*100,2),
    settings:{stake:C.stake,leverage:C.leverage,stopPct:C.stop*100,targetPct:C.target*100,
      longThresholdPct:C.longThreshold*100,shortThresholdPct:C.shortThreshold*100,
      feePct:C.fee*100,slipPct:C.slip*100,
      maintenanceMarginPct:C.maintenance*100,notional:C.stake*C.leverage}};}
const HTML=`<!doctype html><html lang="de"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"/>
<meta name="theme-color" content="#0b1225"/><title>Quest AI Trader v2.2 · LAB</title>
<style>
:root{color-scheme:dark;--bg:#0b1225;--panel:#151f33;--panel2:#1a2a3d;--line:#2b3d53;--text:#eef6ff;--mute:#9aadbf;--good:#55e8be;--bad:#ff8295;--amber:#ffcf87}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#1b3346,#0b1225 65%);color:var(--text);font:15px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}button,input{font:inherit}
main{max-width:740px;margin:auto;padding:24px 15px 50px}header,.flex,.section{display:flex;align-items:center;justify-content:space-between;gap:12px}header{margin-bottom:23px}.mark{background:#213c42;border-radius:13px;padding:12px;color:var(--good);font-size:20px}.brand{display:flex;align-items:center;gap:12px}h1{font-size:18px;margin:0;letter-spacing:-.3px}h2{font-size:18px;margin:0}p{line-height:1.55}.sub,.small{color:var(--mute);font-size:12px}.tag{border:1px solid #42776e;border-radius:99px;padding:7px 10px;color:var(--good);font-size:11px;font-weight:800;white-space:nowrap}
.card{background:linear-gradient(145deg,#1a2a3e,#111a2b);padding:19px;border-radius:21px;border:1px solid var(--line);margin:12px 0}.hero{background:linear-gradient(115deg,#1e3b42,#162037 78%)}.eyebrow{font-size:11px;letter-spacing:1.3px;text-transform:uppercase;color:#aac1cb;font-weight:750}.amount{font-size:clamp(30px,9vw,45px);font-weight:800;letter-spacing:-1.4px;margin:6px 0}.value{font-size:23px;font-weight:760;margin-top:9px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.grid .card{margin:0}.section{margin:27px 0 12px}.section small{color:var(--mute);font-size:11px}.divider{height:1px;background:var(--line);margin:15px 0}.accent{color:var(--good)}.negative{color:var(--bad)}.amber{color:var(--amber)}
button{border-radius:13px;border:1px solid var(--line);padding:13px 8px;color:var(--text);background:#283a51;font-weight:750;min-height:47px;cursor:pointer}.primary{background:var(--good);color:#06251c;border:0}.danger{background:#422b39;border-color:#784356;color:#ffd6df}.buttons{display:grid;grid-template-columns:1fr 1fr;gap:10px}.note{font-size:12px;color:var(--mute);line-height:1.5}.alert{padding:12px;border-radius:12px;background:#3b2b2c;border:1px solid #795054;color:#ffd8df;font-size:13px;line-height:1.5;margin-top:10px}.info{padding:12px;border-radius:12px;background:#2d3033;border:1px solid #62574e;color:#f6d2aa;font-size:12px;line-height:1.5}.pill{border-radius:99px;border:1px solid var(--line);padding:5px 9px;font-size:11px;color:var(--amber)}
.row{display:flex;justify-content:space-between;align-items:start;gap:12px;padding:12px 0;border-bottom:1px solid var(--line)}.row:last-child{border:0}.itemtitle{font-weight:740;font-size:13px}.itemsub{font-size:12px;color:var(--mute);line-height:1.5;margin-top:5px}.two{display:grid;grid-template-columns:1fr 1fr;gap:10px}.mini{padding:13px;border-radius:12px;background:#111c2d}.mini b{display:block;font-size:19px;margin-top:6px}.hidden{display:none!important}.input{width:100%;padding:15px;border:1px solid #446077;border-radius:12px;margin:14px 0;background:#0c1627;color:#fff}.footer{text-align:center;color:var(--mute);line-height:1.6;font-size:11px;margin-top:25px}button:disabled{opacity:.55}.rowleft{display:flex;gap:8px;align-items:center;flex-wrap:wrap}canvas{width:100%;height:160px;display:block}
@media(min-width:570px){main{padding:25px 25px 50px}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style></head><body><main>
<header><div class="brand"><div class="mark">⌁</div><div><h1>QUEST AI TRADER</h1><div class="sub">Mobile Lab · BTC/EUR · v2.2 · 5x</div></div></div><span id="badge" class="tag">PAPER · 5x</span></header>
<section id="login" class="card"><div class="eyebrow">Separater Test-Bot</div><h2 style="margin:10px 0">Willkommen im v2.2-Labor (5x)</h2><p class="note">Bitte das für den neuen Cloudflare-Worker gesetzte Secret <b>APP_PASSWORD</b> eingeben. Niemals auf GitHub speichern.</p><input class="input" id="password" type="password" minlength="12" autocomplete="off" placeholder="Cloudflare-Passwort"/><button id="loginBtn" class="primary" style="width:100%">Anmelden</button><div id="loginErr" class="alert hidden"></div></section>
<div id="app" class="hidden">
<section class="card hero"><div class="eyebrow">Virtuelles Gesamtvermögen · Separates Labor</div><div id="total" class="amount">—</div><div class="flex"><div class="small">Seit Start <b id="returns">—</b></div><div class="small">Start 1.000 €</div></div><div class="divider"></div><div class="flex"><div class="small">Letzte abgeschlossene BTC/EUR Stunde</div><b id="price">—</b></div></section>
<div class="grid"><div class="card"><div class="eyebrow">Freies Guthaben</div><div id="cash" class="value">—</div></div><div class="card"><div class="eyebrow">Position</div><div id="position" class="value">—</div></div></div>
<div class="section"><h2>KI-Handelssignal</h2><small>3-Stunden-Richtung</small></div>
<section class="card"><div class="flex"><div><div id="signal" style="font-size:24px;font-weight:850">—</div><div id="sigdetail" class="note">—</div></div><b id="probability" style="font-size:26px">—</b></div><div class="divider"></div><div class="flex"><span class="note">Modell-Validierung</span><b id="approval">—</b></div><div class="flex" style="margin-top:9px"><span class="note">Datenstand</span><span id="dataTime" class="small">—</span></div></section>
<div class="section"><h2>Automatischer Handel</h2><small id="running">—</small></div>
<section class="card"><div class="buttons"><button class="primary" id="startBtn">▶ Starten</button><button class="danger" id="pauseBtn">Ⅱ Pausieren</button><button id="checkBtn">↻ Daten prüfen</button><button id="logoutBtn">Abmelden</button></div><p class="info" style="margin:13px 0">Ausschließlich virtuelle 5x-Orders. Stündlicher Cloudflare-Zeitplan erforderlich. Manuelle Prüfung kann eine neue Stundenkerze handeln, wenn die Simulation aktiv ist.</p><p class="note">Letzter Durchlauf: <span id="lastRun">—</span> · Daten: <span id="source">—</span></p><div id="err" class="alert hidden"></div></section>
<div class="section"><h2>Warum (k)ein Trade?</h2><small>Diagnose v2.2</small></div>
<section class="card"><div class="flex"><b id="decisionTitle" style="font-size:19px">—</b><span id="action" class="pill">—</span></div><p id="decisionDetail" class="note">Neue Protokolle ab dem nächsten abgeschlossenen Stundenlauf.</p><div class="divider"></div><div class="flex"><span class="note">Letzte Entscheidung</span><span id="decisionAt" class="small">—</span></div></section>
<div class="section"><h2>Strategievergleich</h2><small>Historischer Holdout</small></div>
<section class="card"><div class="two"><div class="mini"><div class="small">Nur Long · 5x</div><b id="longReturn">—</b><div id="longTrades" class="small">—</div></div><div class="mini"><div class="small">Long + Short · 5x</div><b id="bothReturn">—</b><div id="bothTrades" class="small">—</div></div></div><div class="divider"></div><p class="note" id="comparisonNote">Backtest ist keine Gewinnzusage. Er enthält simulierte Gebühren und Slippage; Kurztests sind wenig aussagekräftig.</p></section>
<div class="section"><h2>Letzte Entscheidungen</h2><small>Auch übersprungene Trades</small></div><section class="card" id="decisions">Noch keine Einträge.</section>
<div class="section"><h2>Trade-Verlauf</h2><button id="exportBtn" style="min-height:35px;padding:8px">CSV ↓</button></div><section class="card" id="trades">Noch keine Orders.</section>
<div class="section"><h2>Strategie & Sicherheit</h2></div><section class="card"><div class="two"><div class="mini"><span class="small">Margin / Exposure</span><b>50 € / 250 €</b></div><div class="mini"><span class="small">Hebel</span><b>5x</b></div><div class="mini"><span class="small">Stop-Loss</span><b>3 %</b></div><div class="mini"><span class="small">Take-Profit</span><b>6 %</b></div></div><p class="note" style="margin-top:15px">Synthetische 5x Longs/Shorts, KEINE echten Börsenorders. 50 € Margin entspricht etwa 250 € Position. 3 % Kurs-Stop sind etwa 15 % Margin-Verlust VOR Kosten; Liquidation kann 100 % der Margin kosten. Annahme: 0,5 % Wartungsmargin, vereinfachte Liquidation ohne Funding. Kurslücken und stündliche Prüfung können Stopps überspringen. 0,4 % Handelsgebühr plus 0,1 % Slippage je Seite. Backtest: nächste Stunden-Eröffnung; Live-Paper: letzter Schlusskurs als Näherung. Nicht für Echtgeld geeignet.</p></section>
<div class="footer">Unabhängiges v2.2-Testsystem · Keine Finanzberatung · KI-Prognosen können falsch sein · Nicht mit dem v2.1-Produktionsbot verwechseln.</div>
</div></main>
<script>
(()=>{'use strict';const el=id=>document.getElementById(id),fmt=new Intl.NumberFormat('de-DE',{style:'currency',currency:'EUR'});
const dt=x=>x?new Date(x).toLocaleString('de-DE',{day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}):'—';
let pw=sessionStorage.getItem('quest_lab_pw')||'',state=null,busy=false;
function txt(id,v){el(id).textContent=v==null?'—':String(v)};
function showError(message){el('err').textContent=message;el('err').classList.remove('hidden')}
async function api(path,options={}){const res=await fetch(path,{...options,headers:{'Authorization':'Bearer '+pw,...(options.headers||{})}});
const data=await res.json().catch(()=>({error:'Antwort konnte nicht gelesen werden'}));if(!res.ok)throw Error(data.error||'HTTP '+res.status);return data;}
const safe=s=>String(s||'');
function row(title,detail,date){const d=document.createElement('div');d.className='row';
const a=document.createElement('div');a.style.flex='1';const b=document.createElement('div');b.className='itemtitle';b.textContent=title;
const c=document.createElement('div');c.className='itemsub';c.textContent=detail||'';a.append(b,c);
const f=document.createElement('div');f.className='small';f.textContent=date||'';d.append(a,f);return d;}
function render(){if(!state)return;const s=state;txt('total',fmt.format(s.equity));txt('cash',fmt.format(s.cash));
const p=s.position;txt('position',p?p.side+' · '+fmt.format(p.margin)+' / 5x':'Keine');
txt('returns',(s.returnPct>=0?'+':'')+s.returnPct.toFixed(2)+' %');el('returns').className=s.returnPct>=0?'accent':'negative';
txt('price',s.lastPrice?fmt.format(s.lastPrice):'—');txt('signal',s.signal?s.signal.label:'NOCH KEINE DATEN');
el('signal').className=s.signal?.label==='LONG'?'accent':s.signal?.label==='SHORT'?'negative':'amber';
txt('sigdetail',s.signal?'Trend: '+(s.signal.trend>0?'positiv':s.signal.trend<0?'negativ':'neutral'):'Erster Datencheck steht aus');
txt('probability',s.signal?(s.signal.p*100).toFixed(1).replace('.',',')+' %':'—');
txt('approval',s.validation?(s.validation.approved?'Freigegeben':'Gesperrt'):'—');
txt('dataTime',s.signal?dt(s.signal.t*1000):'—');txt('running',s.running?'PAPER AKTIV':'PAUSIERT');
txt('badge',s.running?'● PAPER AKTIV':'PAPER · 5x');txt('lastRun',dt(s.lastRun));txt('source',s.source||'—');
if(s.lastError)showError(s.lastError);else el('err').classList.add('hidden');
txt('decisionTitle',s.decision?.title||'Noch keine Handelsentscheidung');txt('action',s.decision?.action||'WARTET');
txt('decisionDetail',s.decision?.detail||'Ab der ersten neu verarbeiteten Stundenkerze wird hier der Grund protokolliert.');
txt('decisionAt',dt(s.decision?.at));
const k=s.comparison;if(k){txt('longReturn',(k.longOnly.returnPct>=0?'+':'')+k.longOnly.returnPct.toFixed(2)+' %');
txt('longTrades',k.longOnly.closed+' geschlossen · '+k.longOnly.opened+' eröffnet');
txt('bothReturn',(k.both.returnPct>=0?'+':'')+k.both.returnPct.toFixed(2)+' %');
txt('bothTrades',k.both.closed+' geschlossen · '+k.both.opened+' eröffnet');
txt('comparisonNote',k.nTestCandles+' Holdout-Beobachtungen; 5x Exposition, gleiche Kosten (0,4 % Gebühr + 0,1 % Slippage je Seite). Offene Positionen bewertet, nicht gezielt abgeschlossen. 3-Stunden-KI-Prüfung: '+(s.validation.approved?'freigegeben':'nicht freigegeben')+'.');}
const dec=el('decisions');dec.replaceChildren();if(!s.decisions.length)dec.textContent='Noch keine protokollierten Entscheidungen.';
for(const d of s.decisions.slice(0,10))dec.append(row(d.title,d.detail,dt(d.at)));
const trades=el('trades');trades.replaceChildren();if(!s.trades.length)trades.textContent='Noch keine simulierten Orders.';
for(const t of s.trades.slice(0,12))trades.append(row(t.action+' · '+fmt.format(t.price),t.reason||('Kollateral '+fmt.format(t.eur)),dt(t.time)));
el('startBtn').disabled=busy;el('pauseBtn').disabled=busy;el('checkBtn').disabled=busy;
}
async function login(){pw=el('password').value;try{state=await api('/api/status');sessionStorage.setItem('quest_lab_pw',pw);
el('login').classList.add('hidden');el('app').classList.remove('hidden');render();}catch(e){txt('loginErr',e.message);el('loginErr').classList.remove('hidden');}}
async function act(path,action){if(busy)return;busy=true;render();try{state=await api(path,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({action})});render();}catch(e){showError(e.message);}finally{busy=false;render();}}
el('loginBtn').onclick=login;el('password').onkeydown=e=>{if(e.key==='Enter')login()};
el('startBtn').onclick=()=>act('/api/action','start');el('pauseBtn').onclick=()=>act('/api/action','pause');el('checkBtn').onclick=()=>act('/api/tick','');
el('logoutBtn').onclick=()=>{sessionStorage.removeItem('quest_lab_pw');pw='';state=null;el('app').classList.add('hidden');el('login').classList.remove('hidden');};
el('exportBtn').onclick=async()=>{try{const r=await fetch('/api/export',{headers:{Authorization:'Bearer '+pw}});if(!r.ok)throw Error('CSV nicht verfügbar');
const u=URL.createObjectURL(await r.blob()),a=document.createElement('a');a.href=u;a.download='quest_v22_lab_trades.csv';a.click();setTimeout(()=>URL.revokeObjectURL(u),2000)}catch(e){showError(e.message)}};
if(pw)api('/api/status').then(s=>{state=s;el('login').classList.add('hidden');el('app').classList.remove('hidden');render();}).catch(()=>{sessionStorage.removeItem('quest_lab_pw');pw='';});
setInterval(async()=>{if(!state||document.hidden||busy)return;try{state=await api('/api/status');render()}catch(e){showError(e.message)}},60000);
})();
</script></body></html>`;
const JSON_HEADERS={'content-type':'application/json; charset=utf-8','cache-control':'no-store','x-content-type-options':'nosniff'};
function jres(data,status=200){return new Response(JSON.stringify(data),{status,headers:JSON_HEADERS});}
function eq(a,b){const x=String(a),y=String(b);let d=x.length^y.length;
  for(let i=0;i<Math.max(x.length,y.length);i++)d|=(x.charCodeAt(i)||0)^(y.charCodeAt(i)||0);return d===0;}
const configured=env=>!!(env.TRADER_KV&&env.APP_PASSWORD&&String(env.APP_PASSWORD).length>=12);
function auth(req,env){if(!configured(env))return false;const h=req.headers.get('authorization')||'';
  return h.startsWith('Bearer ')&&eq(h.slice(7),env.APP_PASSWORD);}
async function router(req,env){const u=new URL(req.url),m=req.method.toUpperCase();
  if(u.pathname==='/'&&m==='GET')return new Response(HTML,{headers:{'content-type':'text/html; charset=utf-8',
    'cache-control':'no-store','x-content-type-options':'nosniff','referrer-policy':'no-referrer','x-frame-options':'DENY',
    'content-security-policy':"default-src 'none';script-src 'unsafe-inline';style-src 'unsafe-inline';connect-src 'self';img-src 'self' data:;base-uri 'none';form-action 'none';frame-ancestors 'none'"}});
  if(u.pathname==='/api/health'&&m==='GET')return jres({name:'Quest AI Trader v2.2 LAB · 5x',mode:'paper-only',leverage:5,configured:configured(env)});
  if(!u.pathname.startsWith('/api/'))return new Response('Not Found',{status:404});
  if(!configured(env))return jres({error:'Fehlende TRADER_KV-Bindung oder APP_PASSWORD-Secret'},503);
  if(!auth(req,env))return jres({error:'Anmeldung fehlgeschlagen'},401);
  if(u.pathname==='/api/status'&&m==='GET')return jres(publicState(await read(env)));
  if(u.pathname==='/api/tick'&&m==='POST')return jres(publicState(await tick(env)));
  if(u.pathname==='/api/action'&&m==='POST'){
    if(Number(req.headers.get('content-length')||0)>1024)return jres({error:'Zu große Eingabe'},413);
    let action;try{({action}=await req.json())}catch{return jres({error:'Ungültiges JSON'},400)}
    if(!['start','pause'].includes(action))return jres({error:'Ungültige Aktion'},400);
    const s=await read(env);s.running=action==='start';
    await save(env,s);return jres(publicState(action==='start'?await tick(env):s));
  }
  if(u.pathname==='/api/export'&&m==='GET'){
    const s=await read(env);const csv='Zeit,Aktion,Preis EUR,Kollateral EUR,Gebühr EUR,Gewinn/Verlust EUR,Grund\n'+
      s.trades.map(t=>[t.time,t.action,t.price,t.eur,t.fee,t.pnl,t.reason].map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\n');
    return new Response('\ufeff'+csv,{headers:{'content-type':'text/csv; charset=utf-8','content-disposition':'attachment; filename="quest_lab_trades.csv"','cache-control':'no-store'}});
  }
  return jres({error:'Endpunkt unbekannt'},404);
}
export default {async fetch(req,env){try{return await router(req,env)}catch(e){return jres({error:'Interner Fehler: '+String(e.message||e)},500)}},
  async scheduled(_event,env,ctx){if(configured(env))ctx.waitUntil(tick(env));}};
