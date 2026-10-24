(function(scope){
'use strict';
const families=[
 ['text','Текст и язык'],['vision','Изображения и зрение'],['video','Видео и последовательности кадров'],
 ['audio','Звук и речь'],['timeseries','Сигналы и изменения во времени'],['graphs','Связи между объектами'],
 ['trees','Деревья решений'],['table','Табличные данные'],['neural','Слои нейросети'],
 ['attention','Внимание к частям входа'],['memory','Память и последовательности'],['generation','Создание новых данных'],
 ['probability','Вероятности и неопределённость'],['causal','Причины и последствия'],['clusters','Группы похожих объектов'],
 ['learning','Обучение по примерам'],['games','Действия, игры и награды'],['compression','Сжатие и уменьшение модели'],
 ['devices','Модель на устройстве'],['search','Поиск нужной информации'],['agents','Работа с инструментами'],
 ['science','Научные модели и физика'],['math','Математика и вычисления'],['experiments','Практические эксперименты']
];
const members={
 text:'text lab-text autoregression',
 vision:'vision cnn specialblocks lab-vision',
 video:'multimodal jepa',
 audio:'audio',
 timeseries:'timeseries classicaltime tcn lab-signal',
 graphs:'graphsdata gnn kg',
 trees:'trees isolation',
 table:'classification linear naivebayes lab-table data split quality',
 neural:'models neural mlp siamese kan liquid modern moe',
 attention:'transformer linearattention icl',
 memory:'rnn lstm ssm reservoir states cache continual',
 generation:'generative ae vae gan diffusion flows flowmatching ebm lab-generative',
 probability:'probabilistic bayes gp probability uncertainty mixtures',
 causal:'causal explain',
 clusters:'cluster decomposition unsupervised knn anomaly recommend',
 learning:'root training signal learningprocess supervised selfsupervised semisupervised contrastive masked scratch transfer domain lora meta preferences active federated curriculum adaptation',
 games:'controltask bandits rl policy imitation world selfplay lab-game decision',
 compression:'small size compactarch distill quant pruning factorweights binary',
 devices:'tinyml mac export deploy',
 search:'rag lab-search',
 agents:'agents symbolic hybrids systems',
 science:'science pinn neuraloperator snn quantum lab-science',
 math:'math arrays derivatives optimization svm symbolicreg evolution hdc classic',
 experiments:'practice projects repro metrics generalization automl tasks'
};
const byTopic={};for(const [family,ids] of Object.entries(members))for(const id of ids.split(' '))byTopic[id]=family;
const byFamily=new Map(families.map(([id,label],i)=>[id,{id,label,index:i}]));
const source='art/topic-icons.png';
function familyFor(id){return byFamily.get(byTopic[id])||byFamily.get('experiments');}
function render(id,{decorative=false}={}){
 const family=familyFor(id),i=family.index,col=i%6,row=Math.floor(i/6);
 // The image generator's content has four distinct horizontal bands. These
 // viewports retain the complete illustrations instead of cutting into bubbles.
 const y=[28,282,514,748][row],height=[252,230,230,260][row];
 const accessible=decorative?'aria-hidden="true"':`role="img" aria-label="${family.label}"`;
 return `<span class="topic-art" data-art="${family.id}" title="${family.label}" ${accessible}><svg viewBox="${col*256} ${y} 256 ${height}" width="88" height="88" focusable="false" aria-hidden="true"><image href="${source}" width="1536" height="1024"/></svg></span>`;
}
const api={families,byTopic,familyFor,render};
if(typeof module!=='undefined'&&module.exports)module.exports=api;else scope.AtlasArt=api;
})(typeof window!=='undefined'?window:globalThis);
