// The whole web UI: one page over the JSON API in web.cpp. Times are UTC.
#pragma once

static const char PAGE[] = R"HTML(<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QRSS beacon</title>
<style>
body{font:15px system-ui,sans-serif;margin:0 auto;max-width:900px;padding:0 16px 40px;color:#222;background:#fafafa}
h2{margin:24px 0 8px;font-size:17px}
table{border-collapse:collapse;width:100%}td,th{padding:4px 6px;border-bottom:1px solid #ddd;text-align:left;font-size:14px}
input,select,button{font:inherit;padding:3px 6px}label{display:inline-block;margin:3px 12px 3px 0}
.warn{color:#a40}.err{color:#b00}.ok{color:#070}pre{background:#eee;padding:8px;overflow-x:auto;font-size:12px}
#set label{display:grid;grid-template-columns:200px 1fr;max-width:520px}
</style></head><body>
<h1>QRSS beacon</h1>
<div id="st">loading...</div>

<h2>Pictures</h2>
<p>Upload beacon files made by <code>qrss_beacon.py</code>, one per segment.
<input type="file" id="file" accept=".bin" multiple> <button onclick="upload()">Upload</button> <span id="upmsg"></span></p>
<table id="pics"></table>

<h2>Schedule</h2>
<table id="sched"></table>
<p><label>Picture <select id="e_pid"></select></label>
<label>First send (UTC) <input type="datetime-local" id="e_first" step="900"></label>
<label>Sends <input type="number" id="e_count" value="1" min="0" style="width:5em"> (0 = until removed)</label>
<label>Every <select id="e_every"></select></label>
<label><input type="checkbox" id="e_tiny"> TINY test frame</label>
<button onclick="addEntry()">Add</button> <span id="emsg"></span></p>

<h2>Test carrier</h2>
<p>Plain carrier at the base frequency, for calibrating the crystal. It carries no identification.
<input type="number" id="cs" value="20" min="1" max="120" style="width:5em"> s
<button onclick="post('/api/carrier?seconds='+cs.value).then(load)">Key</button>
<button onclick="post('/api/abort').then(load)">Stop</button></p>

<h2>Settings</h2>
<div id="set"></div>
<p><button onclick="saveSettings()">Save</button> <button onclick="if(confirm('Reboot?'))post('/api/reboot')">Reboot</button> <span id="smsg"></span></p>

<h2>Log</h2><pre id="log"></pre>
<script>
const $=id=>document.getElementById(id);
const EVERY=[[0,'back to back'],[60,'1 h'],[90,'1.5 h'],[120,'2 h'],[180,'3 h'],[240,'4 h'],[360,'6 h'],[480,'8 h'],[720,'12 h'],[1440,'1 day'],[2880,'2 days']];
const FIELDS=[['base_hz','Carrier (Hz, RF)','n'],['spread_hz','Random spread (+- Hz)','n'],['lead_in_s','Lead-in carrier (s)','n'],
 ['xtal_hz','Si5351 crystal (Hz)','n'],['xtal_ppm','Crystal correction (ppm)','n'],['clk','Si5351 output (0-2)','n'],['drive_ma','Drive (mA)','n'],
 ['sda','I2C SDA (GPn, not pin number)','n'],['scl','I2C SCL (GPn, not pin number)','n'],['si_addr','Si5351 I2C address','n'],
 ['tr_gpio','T/R (GPn; -1 none)','n'],['tr_active_high','T/R active high','b'],['tr_lead_ms','T/R lead (ms)','n'],['tr_tail_ms','T/R tail (ms)','n'],
 ['update_hz','Updates per second','n'],['ntp','NTP server','s'],['sync_max_age_s','Refuse to send if sync older than (s)','n'],
 ['hostname','Hostname','s'],['web_user','Web user','s'],['web_pass','Web password (blank = keep)','p']];
let st={},pics=[],keyed=false;
const utc=s=>s?new Date(s*1000).toISOString().replace('T',' ').slice(0,16)+'Z':'-';
const grid=v=>{if(v==32767)return'';const d2=v%10,d1=Math.floor(v/10)%10,r=Math.floor(v/100);
 return String.fromCharCode(65+Math.floor(r/18),65+r%18)+d1+d2};
const esc=s=>String(s).replace(/[&<>"]/g,c=>'&#'+c.charCodeAt(0)+';');
async function api(m,u,b){const r=await fetch(u,{method:m,body:b});return r.json()}
const post=(u,b)=>api('POST',u,b);
function msg(el,r){el.className=r.ok?(r.warning?'warn':'ok'):'err';
 el.textContent=r.ok?(r.warning||'done')+(r.reboot?'; reboot to apply':''):r.error}
function writeOk(){return !keyed||confirm('A pass is on the air. Writing flash pauses the transmitter briefly and costs a short burst of phase error. Write anyway?')}

async function load(){
 st=await api('GET','/api/status');keyed=st.tx.state=='keyed';const t=st.tx;
 let h=`<b>${utc(st.now)}</b> &middot; sync ${st.sync_age_s<0?'never':st.sync_age_s+' s ago'} (step ${st.sync_step_us} us, delay ${st.sync_delay_us} us, crystal ${st.rate_ppm.toFixed(2)} ppm) &middot; ${esc(st.wifi)}<br>`;
 h+=st.not_ready?`<span class="err">Not transmitting: ${esc(st.not_ready)}</span>`:'<span class="ok">Ready</span>';
 h+=` &middot; transmitter <b>${t.state}</b>`;
 if(t.state=='keyed'||t.state=='waiting')h+=` QH ${utc(t.qh)}, ${(t.carrier_hz/1e6).toFixed(6)} MHz (step ${t.step_hz.toFixed(3)} Hz), ${Math.round(100*t.u/Math.max(1,t.n))}%, ${t.late} late`;
 if(t.state=='carrier')h+=` ${(t.carrier_hz/1e6).toFixed(6)} MHz`;
 if(t.state=='error')h+=` <span class="err">error ${t.err}</span>`;
 if(st.next)h+=`<br>Next pass: ${utc(st.next.qh)}, entry ${st.next.id}, segment ${st.next.segment}`;
 h+=`<br>Flash ${Math.round(st.fs_used/1024)} of ${Math.round(st.fs_total/1024)} KB used`;
 $('st').innerHTML=h;$('log').textContent=st.log.join('\n');
}
async function loadPics(){
 pics=await api('GET','/api/pictures');
 $('pics').innerHTML='<tr><th>Picture ID</th><th>Mode</th><th>Segment</th><th>Callsign</th><th>Grid</th><th>Keying</th><th></th></tr>'+
  pics.map(p=>`<tr><td>${p.pid}</td><td>${'ABC'[p.mode]}</td><td>${p.segment+1} of ${p.mode+1}</td><td>${esc(p.callsign)}</td><td>${grid(p.grid)}</td><td>${p.ook?'OOK':'FSK'}</td>
  <td><button onclick="delPic('${p.file}')">Delete</button></td></tr>`).join('');
 const ids=[...new Set(pics.map(p=>p.pid))];
 $('e_pid').innerHTML=ids.map(i=>{const p=pics.find(x=>x.pid==i);return`<option value="${i}">${i} (${esc(p.callsign)}, mode ${'ABC'[p.mode]})</option>`}).join('');
}
async function loadSched(){
 const es=await api('GET','/api/schedule');
 $('sched').innerHTML='<tr><th>Picture</th><th>Mode</th><th>First</th><th>Sends</th><th>Every</th><th>Frame</th><th></th></tr>'+
  es.map(e=>`<tr><td>${e.pid}</td><td>${'ABC'[e.mode]}</td><td>${utc(e.first_slot)}</td><td>${e.count||'until removed'}</td>
  <td>${(EVERY.find(x=>x[0]==e.every_min)||[0,e.every_min+' min'])[1]}</td><td>${e.tiny?'TINY':'FULL'}</td>
  <td><label><input type="checkbox" ${e.enabled?'checked':''} onchange='toggle(${JSON.stringify(e)},this.checked)'>on</label>
  <button onclick="delEntry('${e.id}')">Remove</button></td></tr>`).join('');
}
async function loadSettings(){
 const s=await api('GET','/api/settings');let h='';
 for(let i=0;i<4;i++){const w=s.wifi[i]||{};h+=`<label>WiFi ${i+1}<span><input id="ws${i}" placeholder="SSID" value="${esc(w.ssid||'')}">
  <input id="wp${i}" type="password" placeholder="${w.has_pass?'(unchanged)':'password'}"></span></label>`}
 for(const[k,l,t]of FIELDS)h+=`<label>${l}<input id="s_${k}" ${t=='b'?'type="checkbox"'+(s[k]?' checked':''):
  t=='p'?'type="password"':`value="${esc(s[k])}"`}></label>`;
 $('set').innerHTML=h;
}
async function saveSettings(){
 if(!writeOk())return;const o={wifi:[]};
 for(let i=0;i<4;i++){const ss=$('ws'+i).value.trim();if(!ss)continue;const w={ssid:ss};if($('wp'+i).value)w.pass=$('wp'+i).value;o.wifi.push(w)}
 for(const[k,,t]of FIELDS){const e=$('s_'+k);if(t=='b')o[k]=e.checked;else if(t=='p'){if(e.value)o[k]=e.value}
  else if(t=='s')o[k]=e.value;else o[k]=Number(e.value)}
 msg($('smsg'),await post('/api/settings',JSON.stringify(o)));loadSettings();
}
async function upload(){
 if(!writeOk())return;
 for(const f of $('file').files){const fd=new FormData();fd.append('file',f,f.name);
  const r=await post('/api/pictures',fd);msg($('upmsg'),r);if(!r.ok){$('upmsg').textContent=f.name+': '+r.error;break}}
 loadPics();
}
async function delPic(f){if(writeOk()&&confirm('Delete '+f+'?')){msg($('upmsg'),await api('DELETE','/api/pictures?file='+f));loadPics()}}
async function delEntry(id){if(writeOk()&&confirm('Remove this entry?')){msg($('emsg'),await api('DELETE','/api/schedule?id='+id));loadSched()}}
async function toggle(e,on){if(!writeOk())return loadSched();e.enabled=on;msg($('emsg'),await post('/api/schedule',JSON.stringify(e)));loadSched()}
async function addEntry(){
 if(!writeOk())return;const v=$('e_first').value;if(!v)return;
 const e={pid:$('e_pid').value,first_slot:Math.floor(Date.parse(v+'Z')/1000),count:Number($('e_count').value),
  every_min:Number($('e_every').value),tiny:$('e_tiny').checked,enabled:true};
 const r=await post('/api/schedule',JSON.stringify(e));msg($('emsg'),r);loadSched();
}
$('e_every').innerHTML=EVERY.map(([v,l])=>`<option value="${v}">${l}</option>`).join('');
{const q=Math.ceil(Date.now()/900e3+1)*900e3;$('e_first').value=new Date(q).toISOString().slice(0,16)}
load();loadPics();loadSched();loadSettings();setInterval(load,5000);
</script></body></html>)HTML";
