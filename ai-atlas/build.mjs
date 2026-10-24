import { mkdir, copyFile, readFile, writeFile } from 'node:fs/promises';
const base=new URL('./',import.meta.url);
// The parent JSON is the content source; an exported standalone project can retain map-data.js.
try { const source=JSON.parse(await readFile(new URL('../map-data.json',base),'utf8'));await writeFile(new URL('map-data.js',base),'window.ATLAS_DATA = '+JSON.stringify(source)+';\n'); }
catch(error){if(error.code!=='ENOENT')throw error;}
const files=['index.html','styles.css','core.js','app.js','map-data.js'];
await mkdir(new URL('dist/',base),{recursive:true});
for(const name of files){const source=new URL(name,base);const content=await readFile(source,'utf8');if(!content.trim())throw Error('Empty source: '+name);await copyFile(source,new URL('dist/'+name,base));}
let html=await readFile(new URL('index.html',base),'utf8');
// Inline asset contents only in the offline artifact; deployment retains strict external scripts.
const css=await readFile(new URL('styles.css',base),'utf8');html=html.replace('<link rel="stylesheet" href="styles.css">',()=>'<style>'+css+'</style>');
for(const name of ['map-data.js','core.js','app.js']){const js=await readFile(new URL(name,base),'utf8');html=html.replace('<script defer src="'+name+'"></script>',()=>'<script>'+js.replace(/<\/script/gi,'<\\/script')+'</script>');}
// Defer execution until the DOM exists, preserving script order for the offline version.
const scripts=[];html=html.replace(/<script>[\s\S]*?<\/script>/g,s=>{scripts.push(s);return '';});html=html.replace('</body>',()=>scripts.join('\n')+'\n</body>');
await writeFile(new URL('../ai-atlas.html',base),html);
console.log('Built 5 deployable assets and standalone ai-atlas.html.');
