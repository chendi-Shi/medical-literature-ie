/* Research results come only from the frozen protocol and actual run artifacts. */
(() => {
  const names={logistic:'TF-IDF · Logistic',linear_svm:'TF-IDF · Linear SVM',ce:'BGE · CE',rdrop:'BGE · R-Drop',fgm:'BGE · FGM',rdrop_fgm:'BGE · R-Drop + FGM'};
  const modes={clean_subset:'同组原文',punctuation:'插入标点',char_delete_5pct:'删除 5% 字符',tail_truncate_30pct:'尾部截断 30%'};
  const variants=Object.keys(names);
  const meanStd=x=>x?`${(100*x.mean).toFixed(2)}${x.std===null?'':` ± ${(100*x.std).toFixed(2)}`}`:'—';
  const pp=x=>Number.isFinite(x)?`${x>=0?'+':''}${(100*x).toFixed(2)}`:'—';
  let busy=false, snapshot=null;
  function openRun(id){document.querySelector('[data-view="experiments"]').click();showRun(id).catch(e=>toast(e.message));}
  function bindRows(){document.querySelectorAll('#study [data-run]').forEach(el=>{
    if(el.tagName==='TR'){
      const cell=el.querySelector('td');
      const control=document.createElement('button');control.type='button';control.className='study-open';
      control.innerHTML=cell.innerHTML;control.onclick=event=>{event.stopPropagation();openRun(el.dataset.run)};
      cell.replaceChildren(control);
    }
    el.onclick=()=>openRun(el.dataset.run);el.onkeydown=e=>{if(e.target===el&&e.key==='Enter')openRun(el.dataset.run)};
  })}
  function progressTable(entries,report){
    return `<div class="table-scroll"><table class="study-table"><thead><tr><th>方法</th><th>种子 / 状态</th><th>验证 Macro-F1</th><th>训练耗时</th></tr></thead><tbody>${variants.map(v=>{
      const group=entries.filter(x=>x.variant===v),a=report?.aggregates.find(x=>x.variant===v);
      const expected=['logistic','linear_svm'].includes(v)?[42]:[42,43,44];
      const pills=expected.map(s=>{const x=group.find(y=>y.seed===s);return `<span class="seed-pill ${x?.status==='completed'?'done':x?.status==='running'?'busy':''}">${s} ${x?.evaluation_status==='completed'?'✓ 已评测':x?.status==='completed'?'✓ 已训练':x?.status==='running'?`E${x.current_epoch||1} ${x.batches_done||0}/${x.batches_per_epoch||625}`:'待启动'}</span>`}).join('');
      const finished=group.filter(x=>x.status==='completed'&&Number.isFinite(x.validation_macro_f1));
      const val=a?meanStd(a.validation_macro_f1):finished.length?`${pct(finished.reduce((s,x)=>s+x.validation_macro_f1,0)/finished.length)} <small>(${finished.length}/${expected.length})</small>`:'—';
      const seconds=a?a.training_seconds.mean:finished.length?finished.reduce((s,x)=>s+(x.seconds||0),0)/finished.length:null;
      return `<tr ${group.length?`data-run="${esc(group[0].run)}" tabindex="0"`:''}><td><strong>${names[v]}</strong></td><td>${pills}</td><td>${val}</td><td>${seconds===null?'—':seconds.toFixed(1)+' s'}</td></tr>`;
    }).join('')}</tbody></table></div>`;
  }
  function intervals(comparisons){
    const min=Math.min(-.1,...comparisons.map(x=>100*x.ci95[0])),max=Math.max(.1,...comparisons.map(x=>100*x.ci95[1]));
    const pad=Math.max(.1,(max-min)*.15),lo=min-pad,hi=max+pad;
    const x=v=>170+(100*v-lo)/(hi-lo)*370;
    return `<svg viewBox="0 0 580 190" class="comparison-chart" role="img" aria-label="相对 CE 的配对 Macro-F1 差值与百分之九十五置信区间"><line x1="${x(0)}" y1="15" x2="${x(0)}" y2="137" stroke="#91a399" stroke-dasharray="4 4"/>${comparisons.map((c,i)=>{const y=36+i*42;return `<text x="5" y="${y+4}" font-size="12" fill="#172c27">${names[c.candidate].replace('BGE · ','')}</text><line x1="${x(c.ci95[0])}" x2="${x(c.ci95[1])}" y1="${y}" y2="${y}" stroke="#397455" stroke-width="2"/><circle cx="${x(c.delta_macro_f1)}" cy="${y}" r="4" fill="#24634d"/>`}).join('')}<line x1="170" x2="540" y1="145" y2="145" stroke="#dce4df"/>${[lo,0,hi].map(v=>`<text x="${x(v/100)}" y="163" text-anchor="middle" font-size="10" fill="#687b74">${v.toFixed(2)}</text>`).join('')}<text x="355" y="186" text-anchor="middle" font-size="11" fill="#687b74">Δ Macro-F1（百分点）· 固定三种训练种子</text></svg>`;
  }
  function results(report){
    if(!report){$('studyResults').innerHTML='<div class="panel"><h2>测试结果待统一评测</h2><p class="study-empty">全部 14 个配方完成训练后，再统一评测完整测试集、校准置信度和计算配对区间。这里不会用验证分数代替测试结果。</p></div>';return}
    const chosen=report.aggregates.find(x=>x.variant===report.selected_by_validation);
    $('studyResults').innerHTML=`<div class="panel"><div class="study-title"><h2>完整测试集对照</h2><span class="badge">${report.test_samples.toLocaleString()} 条 · 10 类</span></div><p class="caption">Macro-F1 / Accuracy / ECE 单位为 %，± 为训练种子间样本标准差；稀疏基线仅运行 1 个种子。点击方法查看固定种子 42 的实验。</p><div class="study-results-summary">按验证集均值选择：<strong>${names[chosen.variant]}</strong> · 测试 Macro-F1 <strong>${meanStd(chosen.test_macro_f1)}%</strong></div><div class="table-scroll"><table class="study-table"><thead><tr><th>方法</th><th>种子数</th><th>验证 F1</th><th>测试 F1</th><th>Accuracy</th><th>ECE 原始 → 校准</th><th>训练秒 / 次</th></tr></thead><tbody>${report.aggregates.map(a=>`<tr class="${a.variant===chosen.variant?'picked':''}" data-run="${esc(a.runs[0])}" tabindex="0"><td><strong>${names[a.variant]}</strong>${a.variant===chosen.variant?'<span class="method-tag">验证集选择</span>':''}</td><td>${a.n_seeds}</td><td>${meanStd(a.validation_macro_f1)}</td><td><strong>${meanStd(a.test_macro_f1)}</strong></td><td>${meanStd(a.test_accuracy)}</td><td>${(100*a.uncalibrated_ece.mean).toFixed(2)} → ${(100*a.calibrated_ece.mean).toFixed(2)}</td><td>${a.training_seconds.mean.toFixed(1)}</td></tr>`).join('')}</tbody></table></div><p class="caption">单次训练时间包含验证和 checkpoint 保存；设备为本机 RTX 3050 Ti / CPU。相同样本与优化器步数，额外正则化计算量不同。</p></div>
      <div class="study-grid-two"><div class="panel"><h2>因子消融 · 2 × 2</h2><div class="table-scroll"><table class="study-table"><thead><tr><th>R-Drop</th><th>FGM</th><th>测试 F1</th><th>相对 CE / 95% CI（百分点）</th></tr></thead><tbody>${report.aggregates.filter(a=>!['logistic','linear_svm'].includes(a.variant)).map(a=>{const c=report.paired_comparisons.find(x=>x.candidate===a.variant);return `<tr><td>${a.variant.includes('rdrop')?'✓':'—'}</td><td>${a.variant.includes('fgm')?'✓':'—'}</td><td>${meanStd(a.test_macro_f1)}</td><td>${c?`${pp(c.delta_macro_f1)} [${pp(c.ci95[0])}, ${pp(c.ci95[1])}]`:'参考'}</td></tr>`}).join('')}</tbody></table></div><p class="caption">每类配对重采样 400 次，三种训练种子固定；区间未覆盖训练种子总体不确定性，未做多重比较校正。</p></div><div class="panel"><h2>改善幅度与不确定性</h2>${intervals(report.paired_comparisons)}<p class="caption">横线为条件 95% 区间；跨过零线时，本实验不足以确认对应改善。</p></div></div>
      <div class="panel"><h2>鲁棒性压力测试</h2><p class="caption">每类固定 200 条，共 2,000 条；所有方法和扰动共用同组文本。下行为相对同组原文的变化（百分点）。</p><div class="table-scroll"><table class="study-table"><thead><tr><th>方法</th>${Object.values(modes).map(m=>`<th>${m}</th>`).join('')}</tr></thead><tbody>${report.aggregates.map(a=>`<tr><td><strong>${names[a.variant]}</strong></td>${Object.keys(modes).map(m=>`<td class="stress-cell">${(100*a.robustness[m].macro_f1_mean).toFixed(2)}%<small>${m==='clean_subset'?'参考':pp(a.robustness[m].delta_mean)+' pp'}</small></td>`).join('')}</tr>`).join('')}</tbody></table></div><div class="note">删除和截断可能改变语义；沿用原标签，未人工复标。这是合成扰动压力测试，不能代表自然分布外泛化。</div></div>
      <div class="panel"><h2>逐次证据与部署取舍</h2><div class="table-scroll"><table class="study-table"><thead><tr><th>方法 / 种子</th><th>测试 F1</th><th>峰值显存 MiB</th><th>吞吐 文本/秒</th><th>选择性准确率 / 覆盖率</th></tr></thead><tbody>${report.details.map(d=>`<tr data-run="${esc(d.run)}" tabindex="0"><td>${names[d.variant]} / ${d.seed}</td><td>${pct(d.test_macro_f1)}</td><td>${d.peak_cuda_memory_mb===null?'—':d.peak_cuda_memory_mb?.toFixed(0)||'—'}</td><td>${d.test_texts_per_second.toFixed(0)}</td><td>${pct(d.selective.test_accuracy)} / ${pct(d.selective.test_coverage)}</td></tr>`).join('')}</tbody></table></div><p class="caption">吞吐为本机单次完整测试打分，不含模型加载；稀疏基线在 CPU，神经模型在 GPU，非服务并发基准。拒判阈值在验证集按 80% 覆盖率确定，测试覆盖率可变化。</p></div>`;
  }
  async function refreshStudy(){
    if(busy)return;busy=true;
    try{
      const d=await api('/api/research');snapshot=d;
      if(!d.protocol){$('studyBody').innerHTML='<div class="panel">尚未准备研究数据。运行 python -m scripts.prepare_research。</div>';return}
      const p=d.protocol,s=d.state,e=d.experiments,report=d.report;
      const done=e.filter(x=>x.status==='completed').length,evaluated=e.filter(x=>x.evaluation_status==='completed').length;
      const phase={training:'训练中',evaluating:'统一评测中',completed:'全部完成',failed:'运行失败'}[s.phase]||'准备中';
      $('studyPhase').textContent=phase;$('studyBody').innerHTML=`<div class="study-kpis"><div><small>固定训练预算</small><strong>10,000</strong><small>每类 1,000 · 10 个类别</small></div><div><small>独立验证 / 测试</small><strong>9,991 / 10,000</strong><small>保留全部来源测试行</small></div><div><small>已完成训练</small><strong>${done} / 14</strong><small>2 个稀疏基线 + 4 × 3 神经实验</small></div><div><small>已完成统一评测</small><strong>${evaluated} / 14</strong><small>完整测试 + 4 组压力测试</small></div></div>
        <div class="study-layout"><div class="panel"><div class="study-title"><h2>冻结配方 · 实验进度</h2><span class="study-status">${phase}</span></div><div class="study-progress"><i style="width:${100*(done+evaluated)/28}%"></i></div><p class="caption">${s.phase==='completed'?'全部实验已训练、评测并汇总。':s.phase==='evaluating'?`当前评测：${names[s.active_variant]} / seed ${s.active_seed}`:`当前训练：${names[s.active_variant]||'等待启动'} / seed ${s.active_seed??'—'}`} ${s.error?esc(s.error):''}</p>${progressTable(e,report)}<p class="caption">训练未齐全时仅显示已完成种子的验证均值，不作为最终结论。点击行查看该方法 seed 42 的 checkpoint 和错误样本。</p></div><div class="panel"><div class="section-head">FROZEN PROTOCOL</div><h2>一个问题，统一对照</h2><p class="study-note">${esc(p.question)}</p><dl class="protocol-list"><dt>模型</dt><dd>BGE-small-zh · 约 24M 参数</dd><dt>训练</dt><dd>2 epochs · batch 16 × 累积 2<br>LR 3e−5 · cosine · FP16</dd><dt>正则化</dt><dd>R-Drop α=0.5<br>FGM ε=0.5 · 对抗损失权重 0.5</dd><dt>随机种子</dt><dd>42 / 43 / 44</dd><dt>选模</dt><dd>验证 Macro-F1 选择 checkpoint；方法按验证均值选择</dd><dt>概率校准</dt><dd>验证集拟合单温度；与选模共用验证集</dd></dl><details><summary>查看冻结协议与数据指纹</summary><pre class="log">${esc(JSON.stringify(p,null,2))}</pre></details></div></div>
        <div class="panel"><div class="study-title"><h2>真实数据 · 可追溯审计</h2><span class="badge">第三方 THUCNews 标题子集</span></div><p class="study-note">来源 200,000 条中文新闻标题，原划分 180,000 / 10,000 / 10,000。规范化后清洗训练池为 178,707 条；首轮固定抽取 10,000 条，未使用的 168,707 条不参与本轮训练。验证集移除与测试重叠的 9 条。</p><p class="study-note">训练去除跨划分重叠、同标签重复和冲突标签组；来源验证 / 测试的内部重复与冲突保留并披露（验证 5 组、测试 10 组重复）。本轮未做近重复检测。</p><div class="study-legend">${p.labels.map(x=>`<span>${esc(x)}</span>`).join('')}</div><a class="study-source" target="_blank" rel="noopener" href="https://github.com/${p.source.repository}/tree/${p.source.commit}">来源固定提交 ${p.source.commit.slice(0,12)} →</a><div class="study-fingerprint">DATA SHA256 / ${p.dataset_sha256}</div><details><summary>查看完整数据审计</summary><pre class="log">${esc(JSON.stringify(d.dataset,null,2))}</pre></details></div>`;
      results(report);bindRows();
      const active=['training','evaluating'].includes(s.phase);$('trainButton').disabled=active||runs.some(x=>['running','queued'].includes(x.status));$('suiteLock').classList.toggle('hidden',!active);
    }catch(err){$('studyPhase').textContent='连接失败';console.error(err)}finally{busy=false}
  }
  window.researchPhase=()=>snapshot?.state?.phase;
  $('studyRefresh').onclick=refreshStudy;
  refreshStudy();setInterval(()=>{if(document.visibilityState==='visible'&&snapshot?.state?.phase!=='completed')refreshStudy()},4000);
})();
