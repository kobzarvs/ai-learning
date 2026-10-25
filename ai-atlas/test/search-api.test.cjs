const {test}=require('node:test'),assert=require('node:assert/strict');
const req=(q,method='GET')=>new Request('https://atlas.test/api/search?q='+encodeURIComponent(q),{method});
function db(row){return{query:async()=>[row]};}
test('semantic search is public, bounded and validates requests before enqueue',async()=>{
 const {searchResponse}=await import('../search-api.mjs');
 for(const q of ['', 'a','x'.repeat(301)])assert.equal((await searchResponse(req(q),()=>{throw Error('must not connect');})).status,400);
 assert.equal((await searchResponse(req('Bing','POST'),()=>{throw Error('must not connect');})).status,405);
 const result=await searchResponse(req('Почему  результат меняется'),async()=>db({status:'queued'}));
 assert.equal(result.status,202);assert.equal(result.headers.get('Cache-Control'),'no-store');
});
test('search response exposes only safe result fields and handles unavailable service',async()=>{
 const {searchResponse}=await import('../search-api.mjs');
 const response=await searchResponse(req('эффект'),async()=>db({status:'completed',revision:'149',results:[{topicId:'causal',title:'Причины',tab:'methods',section:'Метод',snippet:'Пример',anchor:{methodId:'random'},score:.8,private:'secret'}]}));
 assert.equal(response.status,200);const data=await response.json();assert.equal(data.semanticResults[0].topicId,'causal');assert.ok(!JSON.stringify(data).includes('secret'));
 for(const [message,status] of [['ATLAS_SEARCH_BUSY',429],['ATLAS_SEARCH_OFFLINE',503],['postgres://private-secret',503]]){
  const failed=await searchResponse(req('эффект'),async()=>({query:async()=>{throw Error(message);}}));assert.equal(failed.status,status);assert.ok(!(await failed.text()).includes('private-secret'));
 }
});
