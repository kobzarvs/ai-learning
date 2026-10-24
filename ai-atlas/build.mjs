import { mkdir, copyFile, readFile, writeFile } from 'node:fs/promises';
const base=new URL('./',import.meta.url);
// The parent JSON is the content source; an exported standalone project can retain map-data.js.
try { const source=JSON.parse(await readFile(new URL('../map-data.json',base),'utf8'));await writeFile(new URL('map-data.js',base),'window.ATLAS_DATA = '+JSON.stringify(source)+';\n'); }
catch(error){if(error.code!=='ENOENT')throw error;}
const files=['index.html','styles.css','core.js','app.js','map-data.js','lesson-data.js','workspace-core.js','workspace.js','vendor/prism.js','vendor/PRISM-LICENSE.txt','ui.js','art.js','launch.js'];
await mkdir(new URL('dist/',base),{recursive:true});
for(const name of files){const source=new URL(name,base);const content=await readFile(source,'utf8');if(!content.trim())throw Error('Empty source: '+name);await mkdir(new URL('dist/'+name.substring(0,name.lastIndexOf('/')+1),base),{recursive:true});await copyFile(source,new URL('dist/'+name,base));}
await mkdir(new URL('dist/art/',base),{recursive:true});await copyFile(new URL('art/topic-icons.png',base),new URL('dist/art/topic-icons.png',base));
const artwork='data:image/png;base64,'+(await readFile(new URL('art/topic-icons.png',base))).toString('base64');
let html=await readFile(new URL('index.html',base),'utf8');
// Inline asset contents only in the offline artifact; deployment retains strict external scripts.
const css=await readFile(new URL('styles.css',base),'utf8');html=html.replace('<link rel="stylesheet" href="styles.css">',()=>'<style>'+css+'</style>');
for(const name of ['map-data.js','core.js','lesson-data.js','workspace-core.js','vendor/prism.js','ui.js','art.js','launch.js','workspace.js','app.js']){let js=await readFile(new URL(name,base),'utf8');if(name==='art.js')js=js.replace('art/topic-icons.png',artwork);html=html.replace('<script defer src="'+name+'"></script>',()=>'<script>'+js.replace(/<\/script/gi,'<\\/script')+'</script>');}
// Defer execution until the DOM exists, preserving script order for the offline version.
const scripts=[];html=html.replace(/<script>[\s\S]*?<\/script>/g,s=>{scripts.push(s);return '';});html=html.replace('</body>',()=>scripts.join('\n')+'\n</body>');
await writeFile(new URL('../ai-atlas.html',base),html);
console.log('Built Atlas assets, bundled syntax highlighting, topic art and standalone ai-atlas.html.');
