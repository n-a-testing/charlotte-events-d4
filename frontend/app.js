'use strict';
const topics = ['technology','arts','sports','wellness','career','community','academic','culture'];
const form = document.querySelector('#preferences'), button = document.querySelector('#submit');
const statusNode = document.querySelector('#status'), results = document.querySelector('#results');
let apiBase;
for (const topic of topics) {
  const label = document.createElement('label'); label.className = 'topic';
  const input = document.createElement('input'); input.type = 'checkbox'; input.name = 'interests'; input.value = topic;
  label.append(input, document.createTextNode(topic[0].toUpperCase()+topic.slice(1)));
  document.querySelector('#topics').append(label);
}
function setStatus(text, error=false) { statusNode.textContent=text; statusNode.classList.toggle('error',error); }
async function loadJSON(url, options={}) {
  const r=await fetch(url,{...options,signal:AbortSignal.timeout(120000)});
  if(!r.ok) {
    let data; try{data=await r.json();}catch{}
    const detail=Array.isArray(data?.detail)?data.detail.map(x=>x.msg).join('; '):data?.detail;
    throw new Error(detail || `Request failed (${r.status}). Please try again.`);
  }
  return r.json();
}
async function connect() {
  try {
    const config=await loadJSON('./config.json');
    if(!config.apiBase) throw new Error('The Modal API has not been configured yet.');
    const url=new URL(config.apiBase);
    if(url.protocol!=='https:' || !url.hostname.endsWith('.modal.run')) throw new Error('A deployed Modal HTTPS URL is required.');
    apiBase=config.apiBase.replace(/\/$/,'');
    await loadJSON(apiBase+'/health');
    button.disabled=false; button.textContent='Find events';
    const docs=document.querySelector('#docs'); docs.href=apiBase+'/docs'; docs.hidden=false;
    setStatus('Connected. Choose interests to find events.');
  } catch(e) {setStatus(e.message,true);button.textContent='Retry connection';button.disabled=false;}
}
form.addEventListener('submit',async e=>{
  e.preventDefault();
  if(!apiBase) {button.disabled=true;button.textContent='Connecting…';await connect();return;}
  button.disabled=true;button.textContent='Finding events…';setStatus('Ranking events for your interests…');
  results.replaceChildren();
  try {
    const chosen=document.querySelector('#window').value;
    const payload={interests:[...form.querySelectorAll('input:checked')].map(i=>i.value),free_text:document.querySelector('#query').value,
      saved_ids:[],as_of:chosen==='today'?new Date().toISOString():chosen,limit:Number(document.querySelector('#limit').value)};
    const data=await loadJSON(apiBase+'/recommendations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    setStatus(`${data.results.length} recommendations from ${data.candidate_count} candidate events.`);
    if(!data.results.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='No events in this snapshot match the selected date window. Try the November 15 demo window.';results.append(empty);}
    for(const item of data.results) {
      const card=document.createElement('article');card.className='card';
      const score=document.createElement('span');score.className='score';score.textContent=`Score ${item.score.toFixed(3)}`;
      const h=document.createElement('h3'), link=document.createElement('a');link.textContent=item.title;
      try{const url=new URL(item.url);if(['https:','http:'].includes(url.protocol)){link.href=url.href;link.target='_blank';link.rel='noopener noreferrer';}}catch{}
      h.append(link);const meta=document.createElement('div');meta.className='meta';
      meta.textContent=new Intl.DateTimeFormat('en-US',{dateStyle:'medium',timeStyle:'short',timeZone:'America/New_York'}).format(new Date(item.start_at))+' ET'+(item.location?' · '+item.location:'');
      card.append(score,h,meta);
      for(const topic of item.topics){const tag=document.createElement('span');tag.className='tag';tag.textContent=topic;card.append(tag);}
      results.append(card);
    }
  }catch(error){setStatus(error.name==='TimeoutError'?'The API took too long to respond. Please retry.':error.message,true);}
  finally{button.disabled=false;button.textContent='Find events';}
});
connect();
