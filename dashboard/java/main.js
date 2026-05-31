/* VIC Fuel Forecast Dashboard - main.js
   CL04_G04 · Swinburne COS40007 AI Engineering */

/* CONFIG  */
const GITHUB_ORG    = 'COS40007-2026-Classrooms';
const GITHUB_REPO   = 'CL04_G04';
const GITHUB_BRANCH = 'main';
const RAW = `https://raw.githubusercontent.com/${GITHUB_ORG}/${GITHUB_REPO}/${GITHUB_BRANCH}`;
/*const MAPBOX_TOKEN  =''; PLEASE DO NOT TOUCH THIS OR I WILL FIND YOU*/
const MAPBOX_TOKEN = window.MAP_API || '__MAP_API__';
const C = {
  c1:'#0B7F8C', c2:'#2CBFBF', c3:'#F2A413', c4:'#D96704', c5:'#BF3604',
  bdr:'#E2E8ED', txt:'#0A0A0A', txt3:'#718096'
};
const ZONE_C = {
  inner_metro:  C.c1,
  middle_metro: C.c3,
  regional_vic: C.c5,
  rural_remote: '#888'
};

let suburbData  = null;
let forecastData = null;
let mapInstance = null;
let curMapFuel   = 'ulp91';
let curChartFuel = 'all'; /* FIX 4: shared chart fuel filter */

/*  POSTCODE CENTROIDS  
I used AI help in getting the coordinates */

const PC = {
  '3000':[-37.8136,144.9631],'3002':[-37.8153,144.9842],'3003':[-37.7987,144.9444],
  '3006':[-37.8230,144.9575],'3008':[-37.8178,144.9394],'3011':[-37.8007,144.8997],
  '3012':[-37.8275,144.8603],'3013':[-37.8156,144.8844],'3015':[-37.8530,144.8832],
  '3016':[-37.8623,144.8966],'3018':[-37.8694,144.8272],'3019':[-37.8432,144.8439],
  '3020':[-37.7931,144.8289],'3021':[-37.7718,144.8264],'3022':[-37.7598,144.7994],
  '3023':[-37.7765,144.7694],'3024':[-37.9272,144.7019],'3025':[-37.8643,144.8055],
  '3027':[-37.8656,144.7485],'3028':[-37.8810,144.7002],'3029':[-37.8946,144.7003],
  '3030':[-37.9067,144.7461],'3031':[-37.7889,144.9325],'3032':[-37.7634,144.9158],
  '3033':[-37.7537,144.8820],'3034':[-37.7280,144.8262],'3036':[-37.7427,144.8021],
  '3037':[-37.7228,144.7774],'3038':[-37.7178,144.8168],'3039':[-37.7700,144.9194],
  '3040':[-37.7375,144.9216],'3041':[-37.7239,144.9305],'3042':[-37.7265,144.8757],
  '3043':[-37.7008,144.8861],'3044':[-37.7058,144.9418],'3046':[-37.6987,144.9486],
  '3047':[-37.6777,144.9499],'3048':[-37.6733,144.9325],'3049':[-37.6671,144.9141],
  '3058':[-37.7430,144.9671],'3063':[-37.5951,144.9445],'3064':[-37.5962,144.9235],
  '3072':[-37.7512,145.0105],'3073':[-37.7330,145.0119],'3074':[-37.7200,145.0174],
  '3075':[-37.7057,145.0210],'3076':[-37.6762,145.0131],'3082':[-37.6869,145.0587],
  '3083':[-37.7085,145.0720],'3084':[-37.7192,145.0906],'3106':[-37.7835,145.1024],
  '3130':[-37.8162,145.1554],'3134':[-37.7950,145.2292],'3136':[-37.8066,145.2772],
  '3149':[-37.8713,145.1322],'3150':[-37.8787,145.1617],'3152':[-37.8748,145.2220],
  '3155':[-37.8832,145.2839],'3156':[-37.8770,145.2647],'3170':[-37.9191,145.1614],
  '3173':[-37.9680,145.1782],'3174':[-37.9469,145.1872],'3175':[-37.9885,145.2091],
  '3177':[-37.9564,145.2398],'3178':[-37.9262,145.2398],'3195':[-38.0111,145.1075],
  '3199':[-38.1450,145.1249],'3201':[-38.1089,145.1555],'3216':[-38.1681,144.3814],
  '3217':[-38.2001,144.3498],'3280':[-38.3793,142.4851],'3335':[-37.6636,144.5527],
  '3336':[-37.6781,144.5268],'3337':[-37.6913,144.5553],'3338':[-37.6756,144.5956],
  '3340':[-37.7043,144.4262],'3350':[-37.5622,143.8502],'3429':[-37.5762,144.7286],
  '3500':[-34.1851,142.1614],'3550':[-36.7580,144.2793],'3551':[-36.7956,144.3076],
  '3630':[-36.3805,145.4049],'3677':[-36.3635,146.3048],'3690':[-36.1213,146.8922],
  '3750':[-37.6558,145.1071],'3751':[-37.6239,145.1527],'3752':[-37.6112,145.1374],
  '3754':[-37.5942,145.1748],'3755':[-37.5785,145.1349],'3756':[-37.4192,144.9573],
  '3800':[-37.9132,145.1339],'3802':[-37.9480,145.2398],'3803':[-37.9890,145.2719],
  '3804':[-38.0237,145.2962],'3805':[-38.0514,145.2748],'3806':[-38.0370,145.3492],
  '3810':[-38.0716,145.4918],'3820':[-38.1644,145.9334],'3825':[-38.1738,146.2683],
  '3840':[-38.1993,146.5276],'3842':[-38.2451,146.4196],'3844':[-38.2053,146.5420],
  '3850':[-38.1048,147.0659],'3875':[-37.8399,147.6151],'3910':[-38.1697,145.1830],
  '3975':[-38.1142,145.3286],'3976':[-38.1529,145.2777],'3977':[-38.1254,145.3082],
  '3978':[-38.1041,145.2612],'3980':[-38.1991,145.5212],
};

/*  TOOLTIP  */
const tip = document.getElementById('tip');
function showTip(html, e) { tip.innerHTML = html; tip.style.display = 'block'; moveTip(e); }
function moveTip(e) { tip.style.left = (e.clientX + window.scrollX + 12) + 'px'; tip.style.top = (e.clientY + window.scrollY - 10) + 'px'; }
function hideTip() { tip.style.display = 'none'; }

/*  LOAD DATA  */
async function loadData() {
  try {
    const [s, f] = await Promise.all([
      fetch(`${RAW}/reports/suburb_price_forecast.json`),
      fetch(`${RAW}/reports/price_forecast.json`)
    ]);
    suburbData   = await s.json();
    forecastData = await f.json();
  } catch {
    suburbData   = demoSuburbs();
    forecastData = demoForecast();
  }
  const gen = new Date(suburbData.generated_at || Date.now());
  document.getElementById('lup').textContent = 'Updated ' + gen.toLocaleDateString('en-AU', {
    day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit'
  });
  rSum();
  initSearch();
  initMap();
  rGroupedBar(curChartFuel);
  rBollinger(curChartFuel);
  rScatter(curChartFuel);
  rTop3();
}

/*  UTILS  */
function gs(id) { document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
function setFB(b, fuel) {
  document.querySelectorAll('.ftog button').forEach(x => x.classList.remove('active'));
  b.classList.add('active');
  curMapFuel = fuel;
  buildMapLayer(fuel);
}

/* FIX 4: Chart fuel filter — applies to all 3 analytics charts */
function setChartFuel(btn, fuel) {
  document.querySelectorAll('.chart-ftog button').forEach(x => x.classList.remove('active'));
  btn.classList.add('active');
  curChartFuel = fuel;
  ['chart-grouped','chart-bollinger','chart-scatter'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '';
  });
  rGroupedBar(fuel);
  rBollinger(fuel);
  rScatter(fuel);
}

/*  SUMMARY CARDS  */
function rSum() {
  const sf = forecastData.state_forecasts || suburbData.state_forecasts || {};
  document.getElementById('sg').innerHTML = ['ulp91', 'ulp95', 'diesel'].map(k => {
    const d   = sf[k] || {};
    const chg = d.change_cpl || 0;
    const dir = chg < -.5 ? 'dn' : chg > .5 ? 'up' : 'fl';
    const arr = dir === 'dn' ? '↓' : dir === 'up' ? '↑' : '→';
    const cls = { ulp91: 'u91', ulp95: 'u95', diesel: 'dsl' }[k];
    const lbl = { ulp91: 'Regular Unleaded 91', ulp95: 'Premium Unleaded 95', diesel: 'Diesel' }[k];
    return `<div class="fc ${cls}">
      <div class="cl">${lbl}</div>
      <div style="display:flex;align-items:baseline;gap:7px;margin-bottom:8px">
        <div class="cp">${(d.current_cpl || 0).toFixed(1)}</div>
        <div class="cu">cpl now</div>
      </div>
      <div class="cr">
        <div>
          <div class="cfl">Next week forecast</div>
          <div class="cfv">${(d.forecast_cpl || 0).toFixed(1)} cpl</div>
        </div>
        <div class="cc ${dir}">${arr} ${chg > 0 ? '+' : ''}${chg.toFixed(1)}</div>
      </div>
    </div>`;
  }).join('');
}

/*  SEARCH  */
function initSearch() {
  const inp  = document.getElementById('si');
  const res  = document.getElementById('sr');
  const subs = suburbData.suburbs || [];

  function getMatches(q) {
    return subs.filter(s =>
      s.suburb_name.toLowerCase().includes(q) || s.postcode.includes(q)
    ).slice(0, 8);
  }

  inp.addEventListener('input', () => {
    const q = inp.value.trim().toLowerCase();
    if (q.length < 2) { res.style.display = 'none'; return; }
    const m = getMatches(q);
    if (!m.length) { res.style.display = 'none'; return; }
    res.innerHTML = m.map(s => `
      <div class="sri" onclick="selectSub('${s.postcode}','${s.suburb_name}')">
        <div>
          <div class="srn">${s.suburb_name}</div>
          <div class="srp">${s.postcode} · ${(s.zone || '').replace('_', ' ')}</div>
        </div>
        <div class="srv">${s.fuels?.ulp91?.forecast_cpl?.toFixed(1) || '-'} cpl</div>
      </div>`).join('');
    res.style.display = 'block';
  });

  /* FIX 1: Enter key zooms to first/exact match */
  inp.addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    const q = inp.value.trim().toLowerCase();
    if (!q) return;
    const m = getMatches(q);
    if (m.length) {
      const exact = m.find(s => s.suburb_name.toLowerCase() === q || s.postcode === q) || m[0];
      selectSub(exact.postcode, exact.suburb_name);
    }
    res.style.display = 'none';
  });

  document.addEventListener('click', e => { if (!e.target.closest('#ss')) res.style.display = 'none'; });
}

function selectSub(pc, name) {
  document.getElementById('si').value = name;
  document.getElementById('sr').style.display = 'none';
  flyTo(pc);
}

function flyTo(pc) {
  const c = PC[pc];
  if (!c || !mapInstance) return;
  mapInstance.flyTo({ center: [c[1], c[0]], zoom: 12, duration: 1200 });
  document.getElementById('map').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/*  MAP  */
function initMap() {
  if (MAPBOX_TOKEN === '__MAP_API__' || typeof mapboxgl === 'undefined') {
    document.getElementById('map').innerHTML = `
      <div style="height:520px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;color:#718096;font-size:13px">
        <div style="font-size:42px;opacity:.25">🗺</div>
        <div style="text-align:center">
          <strong style="color:#0B7F8C;display:block;margin-bottom:6px">Mapbox map</strong>
          Requires MAP_API - injected at deploy time
        </div>
      </div>`;
    return;
  }
  mapboxgl.accessToken = MAPBOX_TOKEN;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  mapInstance = new mapboxgl.Map({
    container: 'map',
    style: isDark ? 'mapbox://styles/mapbox/dark-v11' : 'mapbox://styles/mapbox/light-v11',
    center: [144.9631, -37.8136],
    zoom: 11,
    maxBounds: [[140, -40], [151, -33.5]]
  });
  mapInstance.addControl(new mapboxgl.NavigationControl(), 'top-right');
  mapInstance.on('load', () => buildMapLayer('ulp91'));
}

function buildMapLayer(fuel) {
  if (!mapInstance) return;
  const subs = (suburbData.suburbs || []).map(s => {
    const c  = PC[s.postcode];
    if (!c) return null;
    const fd = s.fuels?.[fuel] || {};
    return {
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [c[1], c[0]] },
      properties: {
        postcode:    s.postcode,
        suburb_name: s.suburb_name,
        zone:        s.zone,
        forecast:    fd.forecast_cpl || 0,
        u91:         (s.fuels?.ulp91  || {}).forecast_cpl || 0,
        u95:         (s.fuels?.ulp95  || {}).forecast_cpl || 0,
        dsl:         (s.fuels?.diesel || {}).forecast_cpl || 0,
      }
    };
  }).filter(Boolean);

  const prices = subs.map(f => f.properties.forecast).filter(v => v > 0);
  const mn  = Math.min(...prices);
  const mx  = Math.max(...prices);
  const mid = (mn + mx) / 2;

  document.getElementById('mln').textContent = mn.toFixed(0);
  document.getElementById('mlx').textContent = mx.toFixed(0);

  const expr = ['interpolate', ['linear'], ['get', 'forecast'],
    mn, '#2CBFBF', mid, '#F2A413', mx, '#BF3604'];

  if (mapInstance.getSource('s')) {
    mapInstance.getSource('s').setData({ type: 'FeatureCollection', features: subs });
    mapInstance.setPaintProperty('sc', 'circle-color', expr);
  } else {
    mapInstance.addSource('s', { type: 'geojson', data: { type: 'FeatureCollection', features: subs } });
    mapInstance.addLayer({
      id: 'sc', type: 'circle', source: 's',
      paint: {
        'circle-radius':        ['interpolate', ['linear'], ['zoom'], 6, 5, 11, 13],
        'circle-color':         expr,
        'circle-opacity':       .85,
        'circle-stroke-width':  1.5,
        'circle-stroke-color':  'white',
        'circle-stroke-opacity': .6
      }
    });
    const popup = new mapboxgl.Popup({ closeButton: false, maxWidth: '210px' });
    mapInstance.on('mouseenter', 'sc', e => {
      mapInstance.getCanvas().style.cursor = 'pointer';
      const p = e.features[0].properties;
      popup.setLngLat(e.lngLat).setHTML(`
        <div class="pn">${p.suburb_name}</div>
        <div class="pp">${p.postcode}</div>
        <div class="pr"><span class="prl">ULP91</span><span class="prv" style="color:#0B7F8C">${p.u91.toFixed(1)} cpl</span></div>
        <div class="pr"><span class="prl">ULP95</span><span class="prv" style="color:#9A6800">${p.u95.toFixed(1)} cpl</span></div>
        <div class="pr"><span class="prl">Diesel</span><span class="prv" style="color:#BF3604">${p.dsl.toFixed(1)} cpl</span></div>
        <span class="pz">${(p.zone || '').replace(/_/g, ' ')}</span>
      `).addTo(mapInstance);
    });
    mapInstance.on('mouseleave', 'sc', () => { mapInstance.getCanvas().style.cursor = ''; popup.remove(); });

    /* FIX 2: Click keeps popup pinned until next click elsewhere */
    let pinnedPopup = null;
    mapInstance.on('click', 'sc', e => {
      if (pinnedPopup) { pinnedPopup.remove(); pinnedPopup = null; }
      const p = e.features[0].properties;
      pinnedPopup = new mapboxgl.Popup({ closeButton: true, maxWidth: '220px' })
        .setLngLat(e.lngLat)
        .setHTML(`
          <div class="pn">${p.suburb_name}</div>
          <div class="pp">${p.postcode}</div>
          <div class="pr"><span class="prl">ULP91 forecast</span><span class="prv" style="color:#0B7F8C">${Number(p.u91).toFixed(1)} cpl</span></div>
          <div class="pr"><span class="prl">ULP95 forecast</span><span class="prv" style="color:#9A6800">${Number(p.u95).toFixed(1)} cpl</span></div>
          <div class="pr"><span class="prl">Diesel forecast</span><span class="prv" style="color:#BF3604">${Number(p.dsl).toFixed(1)} cpl</span></div>
          <span class="pz">${(p.zone || '').replace(/_/g, ' ')}</span>
        `)
        .addTo(mapInstance);
    });
    mapInstance.on('click', e => {
      if (!mapInstance.queryRenderedFeatures(e.point, { layers: ['sc'] }).length) {
        if (pinnedPopup) { pinnedPopup.remove(); pinnedPopup = null; }
      }
    });
  }
}

/* D3 CHARTS*/

/*  1. GROUPED BAR: Current vs Forecast  */
function rGroupedBar(fuelFilter) { fuelFilter = fuelFilter || curChartFuel;
  const sf     = forecastData.state_forecasts || suburbData.state_forecasts || {};
  const allFuels = ['ulp91', 'ulp95', 'diesel'];
  const fuels = fuelFilter === 'all' ? allFuels : [fuelFilter];
  const labels = { ulp91: 'ULP91', ulp95: 'ULP95', diesel: 'Diesel' };
  const data   = fuels.map(k => ({
    fuel:     labels[k],
    current:  (sf[k] || {}).current_cpl  || 0,
    forecast: (sf[k] || {}).forecast_cpl || 0,
    change:   (sf[k] || {}).change_cpl   || 0
  }));

  const W  = document.getElementById('chart-grouped').offsetWidth || 500;
  const H  = 260;
  const mg = { top: 10, right: 20, bottom: 40, left: 50 };
  const iW = W - mg.left - mg.right;
  const iH = H - mg.top  - mg.bottom;

  const x0 = d3.scaleBand().domain(data.map(d => d.fuel)).range([0, iW]).paddingInner(.25).paddingOuter(.1);
  const x1 = d3.scaleBand().domain(['current', 'forecast']).range([0, x0.bandwidth()]).padding(.05);
  const allVals = data.flatMap(d => [d.current, d.forecast]);
  const y  = d3.scaleLinear().domain([d3.min(allVals) * .98, d3.max(allVals) * 1.02]).range([iH, 0]);

  const svg = d3.select('#chart-grouped').append('svg')
    .attr('width', W).attr('height', H).attr('class', 'd3-chart');
  const g = svg.append('g').attr('transform', `translate(${mg.left},${mg.top})`);

  g.append('g').attr('class', 'd3-grid')
    .call(d3.axisLeft(y).tickSize(-iW).tickFormat(''))
    .select('.domain').remove();

  const fuelColors = { ULP91: C.c1, ULP95: C.c3, Diesel: C.c5 };

  data.forEach(d => {
    const col = fuelColors[d.fuel];
    const grp = g.append('g').attr('transform', `translate(${x0(d.fuel)},0)`);

    grp.append('rect')
      .attr('x', x1('current')).attr('y', y(d.current))
      .attr('width', x1.bandwidth()).attr('height', iH - y(d.current))
      .attr('fill', col).attr('opacity', .3).attr('rx', 3)
      .on('mouseover', e => showTip(`<strong>${d.fuel}</strong>Current: ${d.current.toFixed(1)} cpl`, e))
      .on('mousemove', moveTip).on('mouseout', hideTip);

    grp.append('rect')
      .attr('x', x1('forecast')).attr('y', y(d.forecast))
      .attr('width', x1.bandwidth()).attr('height', iH - y(d.forecast))
      .attr('fill', col).attr('opacity', .9).attr('rx', 3)
      .on('mouseover', e => showTip(`<strong>${d.fuel}</strong>Forecast: ${d.forecast.toFixed(1)} cpl<br>Change: ${d.change > 0 ? '+' : ''}${d.change.toFixed(1)} cpl`, e))
      .on('mousemove', moveTip).on('mouseout', hideTip);

    const chgColor = d.change < 0 ? '#1A7A4A' : '#B85C00';
    grp.append('text')
      .attr('x', x0.bandwidth() / 2).attr('y', y(Math.max(d.current, d.forecast)) - 6)
      .attr('text-anchor', 'middle').attr('font-size', 11).attr('font-weight', 600)
      .attr('fill', chgColor)
      .text(`${d.change > 0 ? '+' : ''}${d.change.toFixed(1)}`);
  });

  g.append('g').attr('class', 'd3-axis').attr('transform', `translate(0,${iH})`)
    .call(d3.axisBottom(x0).tickSize(0)).select('.domain').remove();
  g.append('g').attr('class', 'd3-axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(v => v.toFixed(0)));
  g.append('text').attr('x', -iH / 2).attr('y', -36).attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', C.txt3).text('cents per litre');
}

/*  2. BOLLINGER BANDS: Suburb price distribution  */
function rBollinger(fuelFilter) { fuelFilter = fuelFilter || curChartFuel;
  const subs      = suburbData.suburbs || [];
  const allFuels  = ['ulp91', 'ulp95', 'diesel'];
  const fuels     = fuelFilter === 'all' ? allFuels : [fuelFilter];
  const fuelLabels = { ulp91: 'ULP91', ulp95: 'ULP95', diesel: 'Diesel' };
  const fuelCols   = { ulp91: C.c1,    ulp95: C.c3,    diesel: C.c5    };

  const bands = fuels.map(k => {
    const vals = subs.map(s => s.fuels?.[k]?.forecast_cpl).filter(v => v && v > 0).sort(d3.ascending);
    return {
      fuel:   fuelLabels[k],
      key:    k,
      min:    d3.min(vals),
      q1:     d3.quantile(vals, .25),
      median: d3.median(vals),
      q3:     d3.quantile(vals, .75),
      max:    d3.max(vals)
    };
  });

  const W  = document.getElementById('chart-bollinger').offsetWidth || 500;
  const H  = 260;
  const mg = { top: 10, right: 20, bottom: 40, left: 50 };
  const iW = W - mg.left - mg.right;
  const iH = H - mg.top  - mg.bottom;

  const x       = d3.scaleBand().domain(fuels.map(k => fuelLabels[k])).range([0, iW]).padding(.4);
  const allVals = bands.flatMap(d => [d.min, d.max]);
  const y       = d3.scaleLinear().domain([d3.min(allVals) * .97, d3.max(allVals) * 1.02]).range([iH, 0]);

  const svg = d3.select('#chart-bollinger').append('svg')
    .attr('width', W).attr('height', H).attr('class', 'd3-chart');
  const g = svg.append('g').attr('transform', `translate(${mg.left},${mg.top})`);

  g.append('g').attr('class', 'd3-grid')
    .call(d3.axisLeft(y).tickSize(-iW).tickFormat(''))
    .select('.domain').remove();

  bands.forEach(d => {
    const cx  = x(d.fuel) + x.bandwidth() / 2;
    const col = fuelCols[d.key];

    // Whisker
    g.append('line')
      .attr('x1', cx).attr('x2', cx)
      .attr('y1', y(d.min)).attr('y2', y(d.max))
      .attr('stroke', col).attr('stroke-width', 1.5).attr('stroke-dasharray', '3,3').attr('opacity', .5);

    // IQR box
    g.append('rect')
      .attr('x', x(d.fuel)).attr('y', y(d.q3))
      .attr('width', x.bandwidth()).attr('height', y(d.q1) - y(d.q3))
      .attr('fill', col).attr('opacity', .18).attr('rx', 4);

    // Median line
    g.append('line')
      .attr('x1', x(d.fuel)).attr('x2', x(d.fuel) + x.bandwidth())
      .attr('y1', y(d.median)).attr('y2', y(d.median))
      .attr('stroke', col).attr('stroke-width', 2.5);

    // Min/Max caps
    [d.min, d.max].forEach(v => {
      g.append('line')
        .attr('x1', cx - 8).attr('x2', cx + 8)
        .attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', col).attr('stroke-width', 1.5).attr('opacity', .6);
    });

    // Median label
    g.append('text')
      .attr('x', cx).attr('y', y(d.median) - 7)
      .attr('text-anchor', 'middle').attr('font-size', 10).attr('font-weight', 600).attr('fill', col)
      .text(d.median?.toFixed(1));

    // Hover target
    g.append('rect')
      .attr('x', x(d.fuel) - 8).attr('y', y(d.max) - 4)
      .attr('width', x.bandwidth() + 16).attr('height', y(d.min) - y(d.max) + 8)
      .attr('fill', 'transparent')
      .on('mouseover', e => showTip(`<strong>${d.fuel}</strong>Max: ${d.max?.toFixed(1)} cpl<br>Q3: ${d.q3?.toFixed(1)} cpl<br>Median: ${d.median?.toFixed(1)} cpl<br>Q1: ${d.q1?.toFixed(1)} cpl<br>Min: ${d.min?.toFixed(1)} cpl`, e))
      .on('mousemove', moveTip).on('mouseout', hideTip);
  });

  g.append('g').attr('class', 'd3-axis').attr('transform', `translate(0,${iH})`)
    .call(d3.axisBottom(x).tickSize(0)).select('.domain').remove();
  g.append('g').attr('class', 'd3-axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(v => v.toFixed(0)));
  g.append('text').attr('x', -iH / 2).attr('y', -36).attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', C.txt3).text('cents per litre');

  document.getElementById('bb-legend').innerHTML = `
    <div class="cli"><div style="width:28px;height:3px;background:#888"></div>Range</div>
    <div class="cli"><div style="width:16px;height:10px;background:var(--c2);opacity:.3;border-radius:2px"></div>IQR</div>
    <div class="cli"><div style="width:16px;height:2.5px;background:var(--c2)"></div>Median</div>`;
}

/*  3. SCATTER: ULP91 forecast vs petrol vehicle count  */
function rScatter(fuelFilter) { fuelFilter = fuelFilter || curChartFuel;
  const fuel = (fuelFilter === 'all') ? 'ulp91' : fuelFilter;
  const vehKey = fuel === 'diesel' ? 'diesel_vehicles' : 'petrol_vehicles';
  const subs = (suburbData.suburbs || [])
    .filter(s => s.fuels?.[fuel]?.forecast_cpl && (s[vehKey] || s.petrol_vehicles) > 0)
    .slice(0, 120);

  const W  = document.getElementById('chart-scatter').offsetWidth || 500;
  const H  = 260;
  const mg = { top: 10, right: 20, bottom: 44, left: 54 };
  const iW = W - mg.left - mg.right;
  const iH = H - mg.top  - mg.bottom;

  const vehField = fuel === 'diesel' ? 'diesel_vehicles' : 'petrol_vehicles';
  const x     = d3.scaleLinear().domain([0, d3.max(subs, d => (d[vehField]||d.petrol_vehicles||0)) * 1.05]).range([0, iW]);
  const yVals = subs.map(d => d.fuels[fuel].forecast_cpl);
  const y     = d3.scaleLinear().domain([d3.min(yVals) * .98, d3.max(yVals) * 1.02]).range([iH, 0]);

  const svg = d3.select('#chart-scatter').append('svg')
    .attr('width', W).attr('height', H).attr('class', 'd3-chart');
  const g = svg.append('g').attr('transform', `translate(${mg.left},${mg.top})`);

  g.append('g').attr('class', 'd3-grid')
    .call(d3.axisLeft(y).tickSize(-iW).tickFormat(''))
    .select('.domain').remove();

  // Regression line
  const xm  = d3.mean(subs, d => d[vehField]||d.petrol_vehicles||0);
  const ym  = d3.mean(subs, d => d.fuels[fuel].forecast_cpl);
  const num = d3.sum(subs, d => ((d[vehField]||d.petrol_vehicles||0) - xm) * (d.fuels[fuel].forecast_cpl - ym));
  const den = d3.sum(subs, d => ((d[vehField]||d.petrol_vehicles||0) - xm) ** 2);
  const slope = num / den;
  const intercept = ym - slope * xm;
  const x1r = d3.min(subs, d => d[vehField]||d.petrol_vehicles||0);
  const x2r = d3.max(subs, d => d[vehField]||d.petrol_vehicles||0);

  g.append('line')
    .attr('x1', x(x1r)).attr('y1', y(slope * x1r + intercept))
    .attr('x2', x(x2r)).attr('y2', y(slope * x2r + intercept))
    .attr('stroke', C.txt3).attr('stroke-width', 1).attr('stroke-dasharray', '4,4').attr('opacity', .6);

  // Dots
  const fuelLabel = { ulp91:'ULP91', ulp95:'ULP95', diesel:'Diesel' }[fuel];
  subs.forEach(d => {
    const col = ZONE_C[d.zone] || '#888';
    const vehCount = d[vehField] || d.petrol_vehicles || 0;
    g.append('circle')
      .attr('cx', x(vehCount)).attr('cy', y(d.fuels[fuel].forecast_cpl))
      .attr('r', 4).attr('fill', col).attr('opacity', .72)
      .attr('stroke', '#fff').attr('stroke-width', .8)
      .style('cursor', 'pointer')
      .on('mouseover', e => showTip(`<strong>${d.suburb_name}</strong>${d.postcode}<br>${fuelLabel}: ${d.fuels[fuel].forecast_cpl.toFixed(1)} cpl<br>Vehicles: ${vehCount.toLocaleString()}`, e))
      .on('mousemove', moveTip).on('mouseout', hideTip)
      .on('click', () => flyTo(d.postcode));
  });

  g.append('g').attr('class', 'd3-axis').attr('transform', `translate(0,${iH})`)
    .call(d3.axisBottom(x).ticks(5).tickFormat(v => v >= 1000 ? (v / 1000).toFixed(0) + 'k' : v));
  g.append('g').attr('class', 'd3-axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(v => v.toFixed(0)));
  g.append('text').attr('x', iW / 2).attr('y', iH + 36)
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', C.txt3).text(fuel === 'diesel' ? 'Diesel vehicles registered' : 'Petrol vehicles registered');
  g.append('text').attr('x', -iH / 2).attr('y', -40).attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', C.txt3).text(fuelLabel + ' forecast (cpl)');
}

/*  TOP 3  */
function rTop3() {
  const sf   = forecastData.state_forecasts || suburbData.state_forecasts || {};
  const subs = suburbData.suburbs || [];
  document.getElementById('t3g').innerHTML = ['ulp91', 'ulp95', 'diesel'].map(k => {
    const sorted = [...subs].filter(s => s.fuels?.[k]?.forecast_cpl)
      .sort((a, b) => a.fuels[k].forecast_cpl - b.fuels[k].forecast_cpl).slice(0, 3);
    const avg = (sf[k] || {}).forecast_cpl || 0;
    const cls = { ulp91: 'u91', ulp95: 'u95', diesel: 'dsl' }[k];
    const lbl = { ulp91: 'Cheapest ULP91', ulp95: 'Cheapest ULP95', diesel: 'Cheapest Diesel' }[k];
    return `<div class="t3c">
      <div class="t3h"><span class="t3b ${cls}">${k.toUpperCase()}</span><span class="t3tl">${lbl} next week</span></div>
      ${sorted.map((s, i) => {
        const fc = s.fuels[k].forecast_cpl;
        const sv = avg - fc;
        return `<div class="t3i" onclick="flyTo('${s.postcode}')">
          <div class="rb r${i + 1}">${i + 1}</div>
          <div class="t3s">
            <div class="t3sn">${s.suburb_name}</div>
            <div class="t3sz">${s.postcode} · ${(s.zone || '').replace('_', ' ')}</div>
          </div>
          <div style="text-align:right">
            <div class="t3pv">${fc.toFixed(1)}</div>
            <div class="t3pu">cpl</div>
            ${sv > 0 ? `<div class="t3sv">↓ ${sv.toFixed(1)} below avg</div>` : ''}
          </div>
        </div>`;
      }).join('')}
    </div>`;
  }).join('');
}

/*  DEMO DATA  */
function demoForecast() {
  return { generated_at: new Date().toISOString(), state_forecasts: {
    ulp91:  { current_cpl: 180.9, forecast_cpl: 171.9, change_cpl: -9.0, forecast_date: '2026-05-08', week_end: '2026-05-14' },
    ulp95:  { current_cpl: 195.9, forecast_cpl: 188.9, change_cpl: -7.0, forecast_date: '2026-05-08', week_end: '2026-05-14' },
    diesel: { current_cpl: 253.5, forecast_cpl: 258.4, change_cpl:  4.9, forecast_date: '2026-05-08', week_end: '2026-05-14' }
  }};
}

function demoSuburbs() {
  const rows = [
    ['3023','Deer Park','inner_metro',173.9,163.9,188.9,178.9,246.5,251.4,39034,10599],
    ['3338','Melton','middle_metro',174.9,164.9,189.9,179.9,247.5,252.4,25662,7487],
    ['3429','Sunbury','middle_metro',174.9,164.9,189.9,179.9,247.5,252.4,26796,9400],
    ['3029','Hoppers Crossing','inner_metro',179.9,169.9,194.9,184.9,250.5,255.4,91799,25678],
    ['3350','Ballarat','middle_metro',176.9,166.9,191.9,181.9,248.5,253.4,42968,15497],
    ['3216','Geelong','middle_metro',177.2,167.2,192.2,182.2,249.5,254.4,36102,11380],
    ['3550','Bendigo','regional_vic',181.3,171.3,196.3,186.3,253.5,258.4,27567,11441],
    ['3500','Mildura','middle_metro',195.9,185.9,210.9,200.9,263.5,268.4,22327,12065],
    ['3630','Shepparton','regional_vic',179.9,169.9,194.9,184.9,251.5,256.4,21916,11140],
    ['3690','Wodonga','regional_vic',180.9,170.9,195.9,185.9,252.5,257.4,24976,13841],
    ['3030','Point Cook','inner_metro',180.8,171.8,195.8,185.8,253.4,258.3,71566,19663],
    ['3977','Cranbourne South','regional_vic',179.9,169.9,194.9,184.9,252.5,257.4,71613,22795],
    ['3064','Craigieburn','inner_metro',179.9,169.9,194.9,184.9,252.5,257.4,77467,20588],
    ['3175','Dandenong','inner_metro',181.7,171.9,196.7,186.9,254.3,258.4,40105,21589],
    ['3844','Traralgon','regional_vic',177.4,168.4,192.4,183.4,251.0,255.9,22253,11531],
    ['3810','Pakenham','regional_vic',177.9,168.9,192.9,183.9,251.5,256.4,35268,13865],
    ['3805','Narre Warren South','regional_vic',177.4,168.4,192.4,183.4,251.0,255.9,35803,10141],
    ['3806','Berwick','regional_vic',176.8,167.8,191.8,182.8,250.4,255.3,33992,9757],
    ['3136','Croydon','inner_metro',178.3,169.3,193.3,184.3,251.9,256.8,28353,7443],
    ['3073','Reservoir','inner_metro',178.9,169.9,193.9,184.9,252.5,257.4,29403,5830],
  ];
  const sf = {
    ulp91:  { current_cpl: 180.9, forecast_cpl: 171.9, change_cpl: -9.0 },
    ulp95:  { current_cpl: 195.9, forecast_cpl: 188.9, change_cpl: -7.0 },
    diesel: { current_cpl: 253.5, forecast_cpl: 258.4, change_cpl:  4.9 }
  };
  return { generated_at: new Date().toISOString(), state_forecasts: sf,
    suburbs: rows.map(d => ({
      postcode: d[0], suburb_name: d[1], zone: d[2],
      petrol_vehicles: d[10], diesel_vehicles: d[11],
      fuels: {
        ulp91:  { current_cpl: d[3], forecast_cpl: d[4], change_cpl: d[4] - d[3] },
        ulp95:  { current_cpl: d[5], forecast_cpl: d[6], change_cpl: d[6] - d[5] },
        diesel: { current_cpl: d[7], forecast_cpl: d[8], change_cpl: d[8] - d[7] }
      }
    }))
  };
}

/*  BOOT - extended at bottom of file  */
/* window.addEventListener('load', loadData); */

/* DARK MODE */
function initDarkMode() {
  if (localStorage.getItem('darkMode') === 'true')
    document.documentElement.setAttribute('data-theme', 'dark');
  updateDarkIcon();
}

function toggleDark() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  if (isDark) {
    document.documentElement.removeAttribute('data-theme');
    localStorage.setItem('darkMode', 'false');
    if (mapInstance) mapInstance.setStyle('mapbox://styles/mapbox/light-v11');
  } else {
    document.documentElement.setAttribute('data-theme', 'dark');
    localStorage.setItem('darkMode', 'true');
    if (mapInstance) mapInstance.setStyle('mapbox://styles/mapbox/dark-v11');
  }
  updateDarkIcon();
  // Rebuild D3 charts after style transition so CSS vars are re-read
  setTimeout(() => {
    ['chart-grouped','chart-bollinger','chart-scatter','chart-history']
      .forEach(id => { const el = document.getElementById(id); if (el) el.innerHTML = ''; });
    rGroupedBar(curChartFuel); rBollinger(curChartFuel); rScatter(curChartFuel);
    if (window._actualsData) rPriceHistory(window._actualsData, window._forecastData);
  }, 350);
}

function updateDarkIcon() {
  const btn = document.getElementById('darkToggle');
  if (!btn) return;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  btn.textContent = isDark ? '☼' : '࣪ ִֶָ☾.';
  btn.title = isDark ? 'Switch to light mode' : 'Switch to dark mode';
}

/* PRICE HISTORY LINE CHART */
async function loadPriceHistory() {
  let rows = [];
  try {
    const res  = await fetch(`${RAW}/data/actuals_history.csv`);
    const text = await res.text();
    const lines = text.trim().split('\n');
    const header = lines[0].split(',');
    rows = lines.slice(1).map(l => {
      const v = l.split(',');
      return {
        date:   new Date(v[0]),
        ulp91:  parseFloat(v[1]) || null,
        ulp95:  parseFloat(v[2]) || null,
        diesel: parseFloat(v[3]) || null,
      };
    }).filter(r => r.date && !isNaN(r.date));
  } catch {
    rows = demoHistory();
  }

  // Take last 12 weeks
  rows = rows.slice(-12);

  // Pull forecast point from forecastData
  const sf = forecastData?.state_forecasts || {};
  const lastDate = rows.length ? rows[rows.length - 1].date : new Date();
  const fcDate   = new Date(lastDate.getTime() + 7 * 24 * 60 * 60 * 1000);
  const fcPoint  = {
    date:   fcDate,
    ulp91:  sf.ulp91?.forecast_cpl  || null,
    ulp95:  sf.ulp95?.forecast_cpl  || null,
    diesel: sf.diesel?.forecast_cpl || null,
    isForecast: true,
  };

  // Cache for dark mode rebuild
  window._actualsData  = rows;
  window._forecastData = fcPoint;

  rPriceHistory(rows, fcPoint);
}

function rPriceHistory(rows, fcPoint) {
  const el = document.getElementById('chart-history');
  if (!el) return;
  el.innerHTML = '';

  const allRows  = [...rows, fcPoint].filter(Boolean);
  const W  = el.offsetWidth || 700;
  const H  = 280;
  const mg = { top: 16, right: 60, bottom: 48, left: 54 }; /* FIX 3: extra right margin for forecast dot */
  const iW = W - mg.left - mg.right;
  const iH = H - mg.top  - mg.bottom;

  const fuels  = ['ulp91', 'ulp95', 'diesel'];
  const colors = { ulp91: '#0B7F8C', ulp95: '#F2A413', diesel: '#BF3604' };
  const labels = { ulp91: 'ULP91', ulp95: 'ULP95', diesel: 'Diesel' };

  /* FIX 3b: explicitly include fcPoint in domain so dashed line is never cut off */
  const allDates = allRows.map(d => d.date).filter(Boolean);
  const x = d3.scaleTime()
    .domain([d3.min(allDates), d3.max(allDates)])
    .range([0, iW]);

  const allVals = allRows.flatMap(r => fuels.map(f => r[f])).filter(v => v && !isNaN(v));
  const y = d3.scaleLinear()
    .domain([d3.min(allVals) * .975, d3.max(allVals) * 1.025])
    .range([iH, 0]);

  const svg = d3.select('#chart-history').append('svg')
    .attr('width', W).attr('height', H).attr('class', 'd3-chart')
    .style('overflow', 'visible');

  const g = svg.append('g').attr('transform', `translate(${mg.left},${mg.top})`);

  // Grid lines
  g.append('g').attr('class', 'd3-grid')
    .call(d3.axisLeft(y).tickSize(-iW).tickFormat(''))
    .call(ax => ax.select('.domain').remove());

  // Excise cut shading (Apr 1 – Jun 30 2026)
  const exciseS = new Date('2026-04-01');
  const exciseE = new Date('2026-06-30');
  const xs = Math.max(0, x(exciseS));
  const xe = Math.min(iW, x(exciseE));
  if (xe > xs) {
    g.append('rect')
      .attr('x', xs).attr('y', 0)
      .attr('width', xe - xs).attr('height', iH)
      .attr('fill', '#F2A413').attr('opacity', .07);
    g.append('text')
      .attr('x', xs + (xe - xs) / 2).attr('y', iH - 6)
      .attr('text-anchor', 'middle').attr('font-size', 9)
      .attr('fill', '#D96704').attr('opacity', .7)
      .text('Excise cut -32cpl');
  }

  // Draw lines for each fuel
  fuels.forEach(fuel => {
    const actuals  = rows.filter(r => r[fuel] !== null && !isNaN(r[fuel]));
    const lastActual = actuals[actuals.length - 1];

    // Solid line - actuals only
    const line = d3.line()
      .defined(d => d[fuel] !== null && !isNaN(d[fuel]))
      .x(d => x(d.date))
      .y(d => y(d[fuel]))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(actuals)
      .attr('fill', 'none')
      .attr('stroke', colors[fuel])
      .attr('stroke-width', 2.2)
      .attr('d', line);

    // Area fill under actuals
    const area = d3.area()
      .defined(d => d[fuel] !== null && !isNaN(d[fuel]))
      .x(d => x(d.date))
      .y0(iH)
      .y1(d => y(d[fuel]))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(actuals)
      .attr('fill', colors[fuel])
      .attr('opacity', .06)
      .attr('d', area);

    // Dashed line from last actual to forecast point
    if (lastActual && fcPoint?.[fuel]) {
      g.append('line')
        .attr('x1', x(lastActual.date)).attr('y1', y(lastActual[fuel]))
        .attr('x2', x(fcPoint.date)).attr('y2', y(fcPoint[fuel]))
        .attr('stroke', colors[fuel])
        .attr('stroke-width', 1.8)
        .attr('stroke-dasharray', '5,4')
        .attr('opacity', .65);

      // Forecast endpoint dot
      g.append('circle')
        .attr('cx', x(fcPoint.date)).attr('cy', y(fcPoint[fuel]))
        .attr('r', 5).attr('fill', colors[fuel]).attr('opacity', .8)
        .attr('stroke', 'var(--surf)').attr('stroke-width', 1.8);
    }

    // Data point dots on actuals
    actuals.forEach(d => {
      g.append('circle')
        .attr('cx', x(d.date)).attr('cy', y(d[fuel]))
        .attr('r', 3.5)
        .attr('fill', colors[fuel])
        .attr('stroke', 'var(--surf)')
        .attr('stroke-width', 1.5)
        .attr('opacity', .85);
    });
  });

  // Axes
  g.append('g').attr('class', 'd3-axis')
    .attr('transform', `translate(0,${iH})`)
    .call(d3.axisBottom(x)
      .ticks(Math.min(allRows.length, 8))
      .tickFormat(d3.timeFormat('%-d %b')))
    .call(ax => ax.select('.domain').remove())
    .selectAll('text')
    .attr('transform', 'rotate(-30)')
    .style('text-anchor', 'end');

  g.append('g').attr('class', 'd3-axis')
    .call(d3.axisLeft(y).ticks(5).tickFormat(v => v.toFixed(0)));

  g.append('text')
    .attr('x', -iH / 2).attr('y', -40)
    .attr('transform', 'rotate(-90)')
    .attr('text-anchor', 'middle')
    .attr('font-size', 11).attr('fill', 'var(--txt3)')
    .text('cents per litre');

  // ── Crosshair + hover tooltip ──
  const crossV = g.append('line')
    .attr('class', 'ph-crosshair')
    .attr('y1', 0).attr('y2', iH)
    .style('display', 'none');

  const hoverDots = {};
  fuels.forEach(fuel => {
    hoverDots[fuel] = g.append('circle')
      .attr('r', 5).attr('fill', colors[fuel])
      .attr('stroke', 'var(--surf)').attr('stroke-width', 2)
      .style('display', 'none').style('pointer-events', 'none');
  });

  // Invisible overlay for mouse events
  const bisect = d3.bisector(d => d.date).left;

  g.append('rect')
    .attr('width', iW).attr('height', iH)
    .attr('fill', 'transparent')
    .on('mousemove', function(event) {
      const [mx] = d3.pointer(event, this);
      const xDate = x.invert(mx);
      // Find nearest actual data point
      const idx = bisect(rows, xDate);
      const d0  = rows[Math.max(0, idx - 1)];
      const d1  = rows[Math.min(rows.length - 1, idx)];
      if (!d0 && !d1) return;
      const d   = !d1 ? d0 : !d0 ? d1 :
        (xDate - d0.date > d1.date - xDate ? d1 : d0);
      if (!d) return;

      const cx = x(d.date);
      crossV.attr('x1', cx).attr('x2', cx).style('display', null);

      fuels.forEach(fuel => {
        if (d[fuel]) {
          hoverDots[fuel]
            .attr('cx', cx).attr('cy', y(d[fuel]))
            .style('display', null);
        }
      });

      const dateStr = d3.timeFormat('%-d %b %Y')(d.date);
      showTip(
        `<strong>${dateStr}</strong>` +
        fuels.filter(f => d[f]).map(f =>
          `<span style="color:${colors[f]}">■</span> ${labels[f]}: ${d[f].toFixed(1)} cpl`
        ).join('<br>'),
        event
      );
    })
    .on('mouseleave', () => {
      crossV.style('display', 'none');
      fuels.forEach(f => hoverDots[f].style('display', 'none'));
      hideTip();
    });
}

function demoHistory() {
  // Generate 12 weeks of plausible demo data working backwards from known values
  const base = { ulp91: 180.9, ulp95: 195.9, diesel: 253.5 };
  const rows = [];
  for (let i = 11; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i * 7);
    // Add some realistic weekly variation
    const noise = (Math.sin(i * 1.3) * 8) + (Math.cos(i * 0.7) * 4);
    rows.push({
      date:   d,
      ulp91:  Math.round((base.ulp91  + noise + (i > 6 ? 15 : 0)) * 10) / 10,
      ulp95:  Math.round((base.ulp95  + noise + (i > 6 ? 15 : 0)) * 10) / 10,
      diesel: Math.round((base.diesel + noise * 0.4 + (i > 6 ? 8 : 0)) * 10) / 10,
    });
  }
  return rows;
}

/* MODEL METRICS TABLE */
async function loadModelMetrics() {
  let metrics = null, pipeline = null;
  try {
    const [mr, pr] = await Promise.all([
      fetch(`${RAW}/reports/price_forecast.json`),
      fetch(`${RAW}/data/pipeline_metrics.json`)
    ]);
    metrics  = await mr.json();
    pipeline = await pr.json();
  } catch {
    metrics  = demoMetrics();
    pipeline = demoPipeline();
  }

  const el = document.getElementById('metrics-table');
  if (!el) return;

  const fuels      = ['ulp91', 'ulp95', 'diesel'];
  const fuelLabels = { ulp91: 'ULP91', ulp95: 'ULP95', diesel: 'Diesel' };
  const mData      = metrics?.metrics || {};
  const genAt      = metrics?.generated_at ? new Date(metrics.generated_at) : new Date();
  const retrained  = pipeline?.retrain_success;
  const threshold  = pipeline?.threshold_passed;

  function confidence(mape) {
    if (!mape) return { pct: 0, str: '-' };
    const score = Math.max(0, Math.min(100, 100 - (mape * 10)));
    return { pct: score, str: score.toFixed(0) + '%' };
  }
  function confColor(pct) {
    return pct > 85 ? '#2CBFBF' : pct > 70 ? '#F2A413' : '#BF3604';
  }
  function mapeStyle(mape) {
    if (!mape) return '';
    if (mape < 1.5) return 'style="color:#1A7A4A;font-weight:600"';
    if (mape < 2.5) return 'style="color:#9A6800;font-weight:600"';
    return 'style="color:#BF3604;font-weight:600"';
  }

  el.innerHTML = `
    <div class="mi-header">
      <div>
        <div class="mi-title">Model accuracy - Ridge Regression (α=1.0)</div>
        <div class="mi-sub">Last retrain: ${genAt.toLocaleDateString('en-AU',{day:'numeric',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'})}</div>
      </div>
      <div class="mi-badges">
        <span class="mi-badge ${threshold ? 'pass' : 'fail'}">${threshold ? '✓ Threshold passed' : '✗ Threshold failed'}</span>
        <span class="mi-badge ${retrained ? 'pass' : 'neutral'}">${retrained ? '↻ Retrained this run' : '→ Models unchanged'}</span>
      </div>
    </div>
    <div style="overflow-x:auto">
      <table class="mi-table">
        <thead>
          <tr>
            <th>Fuel</th>
            <th>MAPE %</th>
            <th>MAE (cpl)</th>
            <th>CV R²</th>
            <th>F1 Score</th>
            <th>Confidence</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          ${fuels.map(k => {
            const h   = mData[k]?.holdout || {};
            const cv  = mData[k]?.cv      || {};
            const conf = confidence(h.mape);
            const statusLabel = !h.mape ? '-' : h.mape < 1.5 ? '✓ Excellent' : h.mape < 2.5 ? '~ Good' : '△ Review';
            const statusCls   = !h.mape ? '' : h.mape < 1.5 ? 'pass' : h.mape < 2.5 ? 'warn' : 'fail';
            const fuelCls     = k === 'ulp91' ? 'u91' : k === 'ulp95' ? 'u95' : 'dsl';
            return `<tr>
              <td><span class="mi-fuel ${fuelCls}">${fuelLabels[k]}</span></td>
              <td ${mapeStyle(h.mape)}>${h.mape ? h.mape.toFixed(2) + '%' : '-'}</td>
              <td>${h.mae    ? h.mae.toFixed(2)    : '-'}</td>
              <td>${cv.r2_mean ? cv.r2_mean.toFixed(4) : '-'}</td>
              <td>${h.f1     ? h.f1.toFixed(4)    : '-'}</td>
              <td>
                <div class="mi-conf-wrap">
                  <div class="mi-conf-bar" style="width:${conf.pct}%;background:${confColor(conf.pct)}"></div>
                  <span>${conf.str}</span>
                </div>
              </td>
              <td><span class="mi-status ${statusCls}">${statusLabel}</span></td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>`;
}
/* this has been used for extreme measures only as well as was the initial testing for UI we left it intentially for full fall back*/
function demoMetrics() {
  return { generated_at: new Date().toISOString(), metrics: {
    ulp91:  { holdout: { mape: 1.77, mae: 3.5, r2: 0.37, f1: 0.9794 }, cv: { mape_mean: 1.78, r2_mean: 0.8014 } },
    ulp95:  { holdout: { mape: 1.18, mae: 2.6, r2: 0.32, f1: 0.9667 }, cv: { mape_mean: 1.34, r2_mean: 0.7932 } },
    diesel: { holdout: { mape: 1.32, mae: 3.1, r2: 0.58, f1: 0.9667 }, cv: { mape_mean: 1.25, r2_mean: 0.8655 } },
  }};
}
function demoPipeline() {
  return { threshold_passed: true, retrain_success: true, run_timestamp: new Date().toISOString() };
}

/* BOOT ; replaces the original window load listener */
window.removeEventListener('load', loadData);
window.addEventListener('load', async () => {
  initDarkMode();
  await loadData(); // loads suburb + forecast data, renders all charts
  loadPriceHistory(); // async -renders after actuals csv loads
  loadModelMetrics(); // async - renders after json loads
});
